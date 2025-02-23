from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.model.diffusion.conditional_unet1d import ConditionalUnet1D
from unified_video_action.model.diffusion.mask_generator import LowdimMaskGenerator
from unified_video_action.common.robomimic_config_util import get_robomimic_config
import unified_video_action.model.vision.crop_randomizer as dmvc
from unified_video_action.common.pytorch_util import dict_apply, replace_submodules
from unified_video_action.utils.language_model import extract_text_features
import pdb


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            pretrain_checkpoint=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.debug = kwargs['debug']
        del kwargs['debug']
        self.task_name = kwargs['task_name']
        self.language_emb_model = kwargs['language_emb_model']
        

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        if 'libero' in self.task_name or 'language_table' in self.task_name:
            policy: PolicyAlgo = algo_factory(
                    algo_name=config.algo_name,
                    config=config,
                    obs_key_shapes={'agentview_rgb': obs_key_shapes['agentview_rgb']},
                    ac_dim=action_dim,
                    device='cpu',
                )
        else:
            policy: PolicyAlgo = algo_factory(
                    algo_name=config.algo_name,
                    config=config,
                    obs_key_shapes=obs_key_shapes,
                    ac_dim=action_dim,
                    device='cpu',
                )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            task_name=self.task_name,
            language_emb_model=self.language_emb_model,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        if pretrain_checkpoint is not None:
            pdb.set_trace()


        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        language_goal = kwargs['language_goal']

        # if 'language_goal' in kwargs:
        #     language_goal = kwargs['language_goal']
        # else:
        #     language_goal = None

        del kwargs['language_goal']
        del kwargs['task_name']
        del kwargs['language_emb_model']


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond, language_goal=language_goal)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], language_goal=None, text_model=None, tokenizer=None, text_latents=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet

        # normalize input
        if 'libero' in self.task_name and 'agentview_image' in obs_dict:
            obs_dict['agentview_rgb'] = obs_dict['agentview_image']
            del obs_dict['agentview_image']


        # print('predict_action language_goal: ', language_goal)

        
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None


        if 'libero_goal' in self.kwargs['task_name']:
            max_length = 15
        elif 'libero_10' in self.kwargs['task_name']:
            max_length = 30
        elif 'libero_90' in self.kwargs['task_name']:
            max_length = 30
        elif 'language_table' in self.kwargs['task_name']:
            max_length = 30
            
        if text_latents is None:
            if self.language_emb_model == 'clip':
                text_tokens = tokenizer(language_goal, padding='max_length', max_length=max_length, return_tensors="pt").to(self.device)
                text_latents = extract_text_features(text_model, text_tokens, language_emb_model=self.language_emb_model)
            elif self.language_emb_model == 'flant5':
                text_tokens = torch.LongTensor(tokenizer(language_goal, padding='max_length', max_length=max_length).input_ids).to(self.device)
                text_latents = extract_text_features(text_model, text_tokens, language_emb_model=self.language_emb_model)
            else:
                text_latents = None


        ##############################################################################################################################
        ## test single forward time
        ##############################################################################################################################
        test_inference_speed = False
        # test_inference_speed = True
        if test_inference_speed:
            start_time = time.time()
            test_times = 100
            print('start test_inference_speed')
        else:
            test_times = 1
        
        for i in range(test_times):
            if self.obs_as_global_cond:
                # condition through global feature
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
                # empty data for action
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                # condition through impainting
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)
                # reshape back to B, To, Do
                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:,:To,Da:] = nobs_features
                cond_mask[:,:To,Da:] = True

            # run sampling
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                language_goal=text_latents,
                **self.kwargs)
        
        
        if test_inference_speed:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('elapsed_time: ', elapsed_time/test_times)
        ##############################################################################################################################
        ##############################################################################################################################
        ##############################################################################################################################
            
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, text_model, tokenizer):
        # normalize input
        assert 'valid_mask' not in batch
        
        
        if self.language_emb_model is not None:    
            language_goal = batch['obs']['language']
            del batch['obs']['language']

            if self.language_emb_model == 'clip':
                text_tokens = {'input_ids': language_goal[:,0].long()[:,0], 'attention_mask': language_goal[:,0].long()[:,1]}
                text_latents = extract_text_features(text_model, text_tokens, language_emb_model=self.language_emb_model)
            elif self.language_emb_model == 'flant5':
                assert (language_goal[:,:-1]-language_goal[:,1:]).sum() < 1e-6
                text_tokens = language_goal[:,0].long()
                text_latents = extract_text_features(text_model, text_tokens, language_emb_model=self.language_emb_model)
        else:
            text_latents = None


        nobs = self.normalizer.normalize(batch['obs'])
        # batch['obs']['image']: [64, 16, 3, 96, 96]
        # batch['obs']['agent_pos']: [64, 16, 2]
        # batch['action']: [64, 16, 2]
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])) # n_obs_steps=2
            nobs_features = self.obs_encoder(this_nobs)
            # this_nobs['image']: [128, 3, 96, 96]
            # this_nobs['agent_pos']: [128, 2]
            # nobs_features: [128, 66]

            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # [64, 132]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape) # [64, 16, 2]

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device) # [64, 16, 2]
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond, language_goal=text_latents)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
