import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import functools
import pdb
from unified_video_action.utils.lpips import LPIPS


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(global_step, threshold=0, value=0.0):
    weight = 1.0
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_num_layers=3,
        disc_in_channels=3,
        pixelloss_weight=4.0,
        disc_weight=1.0,
        perceptual_weight=4.0,
        feature_weight=4.0,
        disc_ndf=64,
        disc_loss="hinge",
        timesteps=16,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.s = timesteps
        self.perceptual_loss = LPIPS().eval()
        self.pixel_loss = l1

        self.discriminator_2d = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, ndf=disc_ndf
        ).apply(weights_init)
        self.discriminator_3d = NLayerDiscriminator3D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, ndf=disc_ndf
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss

        self.pixel_weight = pixelloss_weight
        self.gan_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.gan_feat_weight = feature_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        cond=None,
        split="train",
        predicted_indices=None,
    ):

        b, c, _, h, w = inputs.size()
        rec_loss = self.pixel_weight * F.l1_loss(
            inputs.contiguous(), reconstructions.contiguous()
        )

        frame_idx = torch.randint(0, self.s, [b]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, c, 1, h, w)
        inputs_2d = torch.gather(inputs, 2, frame_idx_selected).squeeze(2)
        reconstructions_2d = torch.gather(
            reconstructions, 2, frame_idx_selected
        ).squeeze(2)

        if optimizer_idx == 0:
            if self.perceptual_weight > 0:
                """
                p_loss = self.perceptual_weight * self.perceptual_loss(rearrange(inputs, 'b c t h w -> (b t) c h w').contiguous(),
                                                                       rearrange(reconstructions, 'b c t h w -> (b t) c h w').contiguous()).mean()
                """
                p_loss = (
                    self.perceptual_weight
                    * self.perceptual_loss(
                        inputs_2d.contiguous(), reconstructions_2d.contiguous()
                    ).mean()
                )
            else:
                p_loss = torch.tensor([0.0])

            disc_factor = adopt_weight(
                global_step, threshold=self.discriminator_iter_start
            )
            logits_real_2d, pred_real_2d = self.discriminator_2d(inputs_2d)
            logits_real_3d, pred_real_3d = self.discriminator_3d(inputs.contiguous())
            logits_fake_2d, pred_fake_2d = self.discriminator_2d(reconstructions_2d)
            logits_fake_3d, pred_fake_3d = self.discriminator_3d(
                reconstructions.contiguous()
            )
            g_loss = (
                -disc_factor
                * self.gan_weight
                * (torch.mean(logits_fake_2d) + torch.mean(logits_fake_3d))
            )

            image_gan_feat_loss = 0.0
            video_gan_feat_loss = 0.0

            for i in range(len(pred_real_2d) - 1):
                image_gan_feat_loss += F.l1_loss(
                    pred_fake_2d[i], pred_real_2d[i].detach()
                )
            for i in range(len(pred_real_3d) - 1):
                video_gan_feat_loss += F.l1_loss(
                    pred_fake_3d[i], pred_real_3d[i].detach()
                )

            gan_feat_loss = (
                disc_factor
                * self.gan_feat_weight
                * (image_gan_feat_loss + video_gan_feat_loss)
            )
            return rec_loss + p_loss + g_loss + gan_feat_loss

        if optimizer_idx == 1:

            # second pass for discriminator update
            logits_real_2d, _ = self.discriminator_2d(inputs_2d)
            logits_real_3d, _ = self.discriminator_3d(inputs.contiguous())
            logits_fake_2d, _ = self.discriminator_2d(reconstructions_2d)
            logits_fake_3d, _ = self.discriminator_3d(reconstructions.contiguous())

            disc_factor = adopt_weight(
                global_step, threshold=self.discriminator_iter_start
            )
            d_loss = (
                disc_factor
                * self.gan_weight
                * (
                    self.disc_loss(logits_real_2d, logits_fake_2d)
                    + self.disc_loss(logits_real_3d, logits_fake_3d)
                )
            )

            return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
