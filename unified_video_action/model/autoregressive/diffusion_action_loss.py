import torch
import torch.nn as nn
from einops import rearrange

from unified_video_action.model.autoregressive.diffusion import create_diffusion
from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN


class DiffActLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        n_frames=4,
        act_diff_training_steps=1000,
        act_diff_testing_steps="100",
        act_model_type="conv_fc",
        **kwargs
    ):
        super(DiffActLoss, self).__init__()
        self.in_channels = target_channels
        self.n_frames = n_frames

        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = kwargs["language_emb_model_type"]

        self.act_model_type = act_model_type

        if self.act_model_type == "conv_fc":
            self.w = 16
            self.h = 16
            self.num_frames = 4
            self.num_actions = 16

            # Single convolutional layer for spatial processing
            self.conv = nn.Sequential(
                nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to a fixed spatial size of 4x4
            )

            # Fully connected layer for action latent prediction
            self.fc = nn.Sequential(
                nn.Linear(z_channels * 4 * 4, z_channels),
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),  # Predict latents for all actions
            )

            self.interpolate = nn.Linear(self.num_frames, self.num_actions)

            self.refine = nn.Sequential(
                nn.Linear(z_channels, z_channels),
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),
            )

        elif self.act_model_type == "conv_ori":
            self.w = 16
            self.h = 16
            self.conv_transpose3d = nn.ConvTranspose3d(
                in_channels=z_channels,
                out_channels=z_channels,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
            )
            self.avg_pool = nn.AvgPool3d(kernel_size=(1, self.w, self.h))
            print("use original conv layers in action prediction")

        else:
            raise NotImplementedError

        if self.language_emb_model == "clip" and self.language_emb_model_type == 2:
            self.text_proj_cond = nn.Linear(512, z_channels, bias=True)

            self.net = SimpleMLPAdaLN(
                in_channels=target_channels,
                model_channels=width,
                out_channels=target_channels * 2,  # for vlb loss
                z_channels=z_channels * 2,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
            )
        else:
            self.net = SimpleMLPAdaLN(
                in_channels=target_channels,
                model_channels=width,
                out_channels=target_channels * 2,  # for vlb loss
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
            )

        self.train_diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="cosine",
            diffusion_steps=act_diff_training_steps,
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=act_diff_testing_steps, noise_schedule="cosine"
        )

    def forward(self, target, z, task_mode=None, text_latents=None):
        bsz, seq_len, _ = target.shape

        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z)

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )  # [2, 4, 256, 768]
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )  # [2, 768, 4, 16, 16]
            z = self.conv_transpose3d(z)  # [2, 768, 16, 16, 16]
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
        else:
            raise NotImplementedError

        target = target.reshape(bsz * seq_len, -1)  # [2, 768, 16, 1, 1]
        z = z.reshape(bsz * seq_len, -1)  # [2, 16, 768]

        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )

        if self.language_emb_model == "clip" and self.language_emb_model_type == 2:
            text_latents = self.text_proj_cond(text_latents)
            text_latents = text_latents.unsqueeze(1).expand(-1, seq_len, -1)
            text_latents = text_latents.reshape(bsz * seq_len, -1)
            z = torch.cat([z, text_latents], dim=-1)

        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs
        )

        action_loss = loss_dict["loss"].reshape(bsz, seq_len)

        total_loss = torch.mean(action_loss)

        return total_loss

    def sample(self, z, temperature=1.0, cfg=1.0, text_latents=None):
        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z)

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )  # [2, 4, 256, 768]
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )  # [2, 768, 4, 16, 16]
            z = self.conv_transpose3d(z)  # [2, 768, 16, 16, 16]
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
        else:
            raise NotImplementedError

        bsz, seq_len, _ = z.shape
        z = rearrange(z, "b t c -> (b t) c")

        if self.language_emb_model == "clip" and self.language_emb_model_type == 2:
            text_latents = self.text_proj_cond(text_latents)
            text_latents = text_latents.unsqueeze(1).expand(-1, seq_len, -1)
            text_latents = text_latents.reshape(bsz * seq_len, -1)
            z = torch.cat([z, text_latents], dim=-1)

        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )

        sampled_token_latent = rearrange(
            sampled_token_latent, "(b t) c -> b t c", b=bsz
        )
        return sampled_token_latent
