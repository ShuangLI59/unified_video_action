from typing import Dict, Tuple
import torch
import sys
import pdb
from einops import rearrange

from unified_video_action.model.autoencoder.autoencoder_vit import ViTAutoencoder 
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.utils.data_utils import process_data
from unified_video_action.utils.perceptual_loss import LPIPSWithDiscriminator


class VAEModel(BaseImagePolicy):
    def __init__(self, 
            autoencoder_path,
            ddconfig,
            optimizer,
            image_resolution,
            train_stage,
            output_dir=None,
            **kwargs):
        super().__init__()
        
        pdb.set_trace()
        self.ddconfig = ddconfig
        if train_stage=='first_stage_triplane' or train_stage=='second_stage_diffusion_transformer':
            first_stage_model = ViTAutoencoder(ddconfig.embed_dim, ddconfig, image_resolution)
            
            if autoencoder_path != 'none':
                print('--------------------------------------------------------------------------------')
                print('Loading vae pretrained model: ', autoencoder_path)
                print('--------------------------------------------------------------------------------')
                # first_stage_model_ckpt = torch.load(autoencoder_path)
                # first_stage_model.load_state_dict(first_stage_model_ckpt)

                ## this one works for missing and unexpected keys, but not for size mismatch
                # first_stage_model_ckpt = torch.load(autoencoder_path)
                # missing_keys, unexpected_keys = first_stage_model.load_state_dict(first_stage_model_ckpt, strict=False)
                # print('Missing keys:', missing_keys)
                # print('Unexpected keys:', unexpected_keys)

                ## this one works for size mismatch
                first_stage_model_ckpt = torch.load(autoencoder_path)
                if 'state_dicts' in first_stage_model_ckpt:
                    if 'model' in first_stage_model_ckpt['state_dicts']:
                        print('Loading model from state_dicts')
                        first_stage_model_ckpt = {k[6:]:v for k,v in first_stage_model_ckpt['state_dicts']['model'].items() if k.startswith('model.')}

                model_state_dict = first_stage_model.state_dict()
                pretrained_state_dict = {k: v for k, v in first_stage_model_ckpt.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
                assert len(model_state_dict) > 0
                assert len(pretrained_state_dict) > 0
                print('--------------------------------------------------------------------------------')
                print('Missing keys or mismatch keys:')
                print([k for k, v in first_stage_model_ckpt.items() if k not in pretrained_state_dict])
                print('--------------------------------------------------------------------------------')
                model_state_dict.update(pretrained_state_dict)
                missing_keys, unexpected_keys = first_stage_model.load_state_dict(model_state_dict, strict=False)
                print('Missing keys:', missing_keys)
                print('Unexpected keys:', unexpected_keys)

            self.model = first_stage_model
            self.criterion = LPIPSWithDiscriminator(disc_start = ddconfig.lossconfig_disc_start,
                                                timesteps = ddconfig.timesteps)
            self.normalizer = LinearNormalizer()
        
        elif train_stage == 'second_stage_unipi_pixel':
            self.model = None
            self.criterion = None
            self.normalizer = None
        else:
            raise ValueError('train_stage not supported')
        
        
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # @property
    # def device(self):
    #     return next(iter(self.model.parameters())).device
    
    def get_optimizer(
            self, 
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=betas
        )
        return optimizer
    
    def forward(self, batch):
        x, _ = process_data(self.ddconfig, batch, self.device, first_stage_data=True)
        x_tilde, vq_loss  = self.model(x)
        ae_loss = self.criterion(vq_loss, x, 
                            rearrange(x_tilde, '(b t) c h w -> b c t h w', b=x.size(0)),
                            optimizer_idx=0,
                            global_step=0)
        return ae_loss

    def generate_images(self, batch):
        # x = process_data(self.ddconfig, batch, self.device, first_stage_data=True, vis_label='first_stage_eval_generated')
        x_tilde, _ = self.model(batch)
        return x_tilde