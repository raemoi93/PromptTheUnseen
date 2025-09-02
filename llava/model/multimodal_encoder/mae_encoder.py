import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

SANITY_CHECK = False

class MAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # Only load config to avoid memory overhead
            self.cfg_only = None  # DINOv2 does not expose a separate config object easily

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # needs sanity check here.
        if SANITY_CHECK:
            hidden_states = image_forward_outs.hidden_states
            print(f'length of hidden_states: {len(hidden_states)} and we select {self.select_layer} layer')
            print(hidden_states[self.select_layer].shape, 'shape of hidden_states[self.select_layer]')
            print(f'you can turn off sanity check by setting SANITY_CHECK = False in model/multimodal_encoder/dino_encoder.py')
            
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            if SANITY_CHECK:
                print(f'We use patch: From {image_features.shape} -> To {image_features[:, 1:].shape}')
            
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.vision_tower.config.hidden_size

    @property
    def num_patches_per_side(self):
        image_size = getattr(self.vision_tower.config, "image_size", 224)
        patch_size = getattr(self.vision_tower.config, "patch_size", 14)  # Default for ViT
        return image_size // patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
