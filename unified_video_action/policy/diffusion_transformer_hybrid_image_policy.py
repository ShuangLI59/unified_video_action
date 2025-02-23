from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from unified_video_action.model.diffusion.mask_generator import LowdimMaskGenerator
from unified_video_action.common.robomimic_config_util import get_robomimic_config
import unified_video_action.model.vision.crop_randomizer as dmvc
from unified_video_action.common.pytorch_util import dict_apply, replace_submodules
from unified_video_action.utils.language_model import extract_text_features
import pdb


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            pretrain_checkpoint=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

    
        self.debug = kwargs['debug']
        del kwargs['debug']
        self.task_name = kwargs['task_name']
        self.language_emb_model = kwargs['language_emb_model']
        

        # parse shape_meta # {'action': {'shape': [2]}, 'obs': {'agent_pos': {'shape': [2], 'type': 'low_dim'}, 'image': {'shape': [3, 96, 96], 'type': 'rgb'}}}
        action_shape = shape_meta['action']['shape'] # [2]
        assert len(action_shape) == 1
        action_dim = action_shape[0] # 2
        obs_shape_meta = shape_meta['obs'] # {'agent_pos': {'shape': [2], 'type': 'low_dim'}, 'image': {'shape': [3, 96, 96], 'type': 'rgb'}}
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
        # obs_config: {'low_dim': ['agent_pos'], 'rgb': ['image'], 'depth': [], 'scan': []}
        
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
        # {'agentview_image': [3, 84, 84], 'robot0_eef_pos': [3], 'robot0_eef_quat': [4], 'robot0_eye_in_hand_image': [3, 84, 84], 'robot0_gripper_qpos': [2]}

        
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
        obs_feature_dim = obs_encoder.output_shape()[0] # 66
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim) # 2
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0 # 66

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            task_name=self.task_name,
            language_emb_model=self.language_emb_model,
        )


        self.obs_encoder = obs_encoder
        self.model = model

        
        if pretrain_checkpoint is not None:
            pretrained_model_ckpt = torch.load(pretrain_checkpoint, map_location='cpu')        
            if 'state_dicts' in pretrained_model_ckpt:
                print('Loading pretrained model from state_dicts', pretrain_checkpoint)
                if 'ema_model' in pretrained_model_ckpt['state_dicts']:
                    pretrained_model_ckpt['state_dicts']['ema_model']

                    pretrained_model_ckpt_ = {k[6:]:v for k,v in pretrained_model_ckpt['state_dicts']['ema_model'].items() if k.startswith('model.')} # remove 'model.'
                    missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_model_ckpt_)
                    print('Model Missing keys:', missing_keys)
                    print('Model Unexpected keys:', unexpected_keys)

                    pretrained_model_ckpt_ = {k[12:]:v for k,v in pretrained_model_ckpt['state_dicts']['ema_model'].items() if k.startswith('obs_encoder.')} # remove 'obs_encoder.'
                    missing_keys, unexpected_keys = self.obs_encoder.load_state_dict(pretrained_model_ckpt_)
                    print('Model obs_encoder keys:', missing_keys)
                    print('Model obs_encoder keys:', unexpected_keys)

        print('--------------------------------------------------------------------------------')
        print(f"obs_encoder: total number of parameters: {sum(p.numel() for p in self.obs_encoder.parameters())}") # 11197088
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        print(f"model: total number of parameters: {sum(p.numel() for p in self.model.parameters())}") # 8975362
        print('--------------------------------------------------------------------------------')
        
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim # 66
        self.action_dim = action_dim # 2
        self.n_action_steps = n_action_steps # 8
        self.n_obs_steps = n_obs_steps # 2
        self.obs_as_cond = obs_as_cond # True
        self.pred_action_steps_only = pred_action_steps_only # False
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of **model** trainable parameters: {}M".format(n_params / 1e6))
        
        n_params = sum(p.numel() for p in self.obs_encoder.parameters() if p.requires_grad)
        print("Number of **obs_encoder** trainable parameters: {}M".format(n_params / 1e6))
        
        # Number of **model** trainable parameters: 8.975362M
        # Number of **obs_encoder** trainable parameters: 11.197088M
        
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator) # [56, 10, 2]
    
        # pdb.set_trace()
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
            model_output = model(trajectory, t, cond, language_goal=language_goal)

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

        # print('test')
        # obs_dict['sideview_image'] = obs_dict['sideview_image'].index_select(dim=2, index=torch.tensor([2, 1, 0], device=obs_dict['sideview_image'].device))
        # obs_dict['robot0_eye_in_hand_image'] = obs_dict['robot0_eye_in_hand_image'].index_select(dim=2, index=torch.tensor([2, 1, 0], device=obs_dict['robot0_eye_in_hand_image'].device))

        # images = obs_dict['robot0_eye_in_hand_image']
        # images = np.array((images * 255).cpu()).astype(np.uint8)
        # image = np.transpose(images[0][0], (1, 2, 0))
        # cv2.imwrite(f"vis2/robot0_eye_in_hand_image.png", image)

        

        assert 'past_action' not in obs_dict # not implemented yet

        ## language table image size
        # agentview_rgb: torch.Size([1, 2, 3, 180, 320])

        # normalize input
        if 'libero' in self.task_name and 'agentview_image' in obs_dict:
            obs_dict['agentview_rgb'] = obs_dict['agentview_image']
            del obs_dict['agentview_image']


        # print('predict_action language_goal: ', language_goal)

        # pdb.set_trace()
        # from diffusion_policy.my_utils.my_data_process import write_image
        # write_image({'agentview_image': np.array(obs_dict['agentview_rgb'].cpu())}, cur_step=0, task_name='testing')
        # pdb.set_trace()

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
        cond = None
        cond_data = None
        cond_mask = None
        
        
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


        
        if self.obs_as_cond:
            # nobs['agent_pos']: [56, 2, 2]->[112, 2]
            # nobs['image']: [56, 2, 3, 96, 96]->[112, 3, 96, 96]
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
            nobs_features = self.obs_encoder(this_nobs) # [112, 66]
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1) # [56, 2, 66]
            shape = (B, T, Da) # 56, 10, 2
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

    
        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            language_goal=text_latents,
            **self.kwargs) # [56, 10, 2]
        
        
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da] # [56, 10, 2] -> [56, 10, 2]
        action_pred = self.normalizer['action'].unnormalize(naction_pred) # [56, 10, 2]

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1 # Tp=2, 2-1
            end = start + self.n_action_steps # 1+8
            action = action_pred[:,start:end] # [56, 8, 2]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, text_model, tokenizer):
        # normalize input
        assert 'valid_mask' not in batch

        
        ## language table iamge size
        # batch['obs']['agentview_rgb'].shape # [1, 10, 3, 360, 640]
        
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

        ######################################################################################################################################################
        ## debug
        ## print the tokens
        # if self.language_emb_model == 'clip':
        #     text_tokens = {'input_ids': language_goal[:,0].long()[:,0], 'attention_mask': language_goal[:,0].long()[:,1]}
        #     for item in text_tokens['input_ids']:
        #         print('compute_loss: ', tokenizer.decode(item, skip_special_tokens=True))
        # elif self.language_emb_model == 'flant5':
        #     text_tokens = language_goal[:,0].long()
        #     for item in text_tokens:
        #         print('compute_loss: ', tokenizer.decode(item, skip_special_tokens=True))

        # pdb.set_trace()
        # from diffusion_policy.my_utils.my_data_process import write_image
        # write_image({'agentview_image': np.array(batch['obs']['agentview_rgb'].cpu())}, cur_step=0, task_name='training')
        # pdb.set_trace()
        ######################################################################################################################################################
        
        
        
        nobs = self.normalizer.normalize(batch['obs']) # nobs['image']: [64, 10, 3, 96, 96], nobs['agent_pos']: [64, 10, 2]
        nactions = self.normalizer['action'].normalize(batch['action']) # [64, 10, 2]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1] # 10
        To = self.n_obs_steps # 2

    
        # handle different ways of passing observation
        cond = None
        trajectory = nactions # [64, 10, 2]
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # this_nobs['image']: [128, 3, 96, 96]
            

            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1) # [64, 2, 66]
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device) # [64, 10, 2]
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long() # [64]
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps) # [64, 10, 2]

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond, language_goal=text_latents) # [64, 10, 2]

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

    def forward(self, batch, text_model, tokenizer):
        return self.compute_loss(batch, text_model, tokenizer)
        