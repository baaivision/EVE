import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor


class VisionTokenizer(nn.Module):
    def __init__(self, input_size, output_size, vision_tower_name):
        super().__init__()

        self.is_loaded = True
        self.hidden_size = input_size
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        self.patch_stride = self.image_processor.patch_stride
        self.dense_stride = self.image_processor.dense_stride

        self.patch_embedding = nn.Sequential(nn.Conv2d(3, input_size, 
                                                       kernel_size=self.patch_stride, 
                                                       stride=self.patch_stride),
                                             nn.GELU(),
                                             nn.Conv2d(input_size, output_size, 
                                                       kernel_size=self.dense_stride, 
                                                       stride=self.dense_stride))
        self.class_embedding = nn.Parameter(torch.randn(output_size))
        self.split_embedding = nn.Parameter(torch.randn(output_size))

    def forward(self, pixel_values):
                
        patch_embeds = []
        for i in range(len(pixel_values)):
            pixel_value = pixel_values[i].to(dtype=self.dtype)
            patch_embed = self.patch_embedding(pixel_value.unsqueeze(0))[0]
            split_embed = self.split_embedding[:, None, None].repeat(1, patch_embed.shape[1], 1)
            patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)

            class_embed = self.class_embedding[None, :].to(dtype=self.dtype)
            patch_embeds.append(torch.cat([class_embed, patch_embed.flatten(1).transpose(0, 1)], dim=0))

        return patch_embeds
    
    @property
    def dtype(self):
        return self.patch_embedding[0].weight.dtype

    @property
    def device(self):
        return self.patch_embedding[0].weight.device