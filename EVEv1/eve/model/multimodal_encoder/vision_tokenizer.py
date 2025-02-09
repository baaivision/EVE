import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from transformers import CLIPImageProcessor
from eve.constants import IMAGE_TOKEN_INDEX


class LocalAttention(nn.Module):
    def __init__(self, input_size, conv_stride, num_heads=8):
        super().__init__()
        self.conv_stride = conv_stride
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size),
                               nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size),
                                nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)

    def forward(self, features):
        reduce_features = F.avg_pool2d(features, kernel_size=self.conv_stride, stride=self.conv_stride)
        B, C, H, W = features.shape
        _, _, h, w = reduce_features.shape
        N = self.conv_stride ** 2

        reduce_features = reduce_features.flatten(2).transpose(-2, -1)
        patch_q = self.q(reduce_features).reshape(B, h * w, self.num_heads, -1).permute(0, 2, 1, 3).unsqueeze(-2)
        
        features = features.unfold(2, self.conv_stride, self.conv_stride).unfold(3, self.conv_stride, self.conv_stride)
        features = features.contiguous().view(B, C, h * w, self.conv_stride, self.conv_stride)
        patch_kv = self.kv(features.flatten(3).permute(0, 2, 3, 1))
        patch_kv = patch_kv.reshape(B, h * w, N, 2, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)

        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.transpose(1, 2).reshape(B, h * w, -1)

        return reduce_features + self.proj(aggre_features)


class GlobalAttention(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size),
                               nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size),
                                nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)
    
    def forward(self, class_feature, features):

        B, N, C = features.shape
        class_feature = class_feature.repeat(B, 1, 1)

        patch_q, patch_kv = self.q(class_feature), self.kv(features)
        patch_q = patch_q.reshape(B, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = patch_kv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.reshape(B, 1, -1)
        
        return class_feature + self.proj(aggre_features)


class VisionTokenizer(nn.Module):
    def __init__(self, input_size, output_size, vision_tower_name):
        super().__init__()

        self.is_loaded = True
        self.hidden_size = input_size
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        patch_stride, conv_stride = self.image_processor.patch_stride, self.image_processor.conv_stride
        self.patch_stride, self.conv_stride = patch_stride, conv_stride

        self.patch_embedding = nn.Conv2d(3, input_size, kernel_size=patch_stride, stride=patch_stride, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(input_size))
        self.split_embedding = nn.Parameter(torch.randn(input_size))

        self.local_attention = LocalAttention(input_size, conv_stride)
        self.global_attention = GlobalAttention(input_size)

    def forward(self, pixel_values, modules):
        pixel_values, pixel_masks = pixel_values[:, :-1, :, :], pixel_values[:, -1:, :, :]

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.dtype))
        patch_masks = F.avg_pool2d(pixel_masks, kernel_size=self.patch_stride, stride=self.patch_stride)
        assert len(torch.where(patch_masks % 1)[0]) == 0
        
        patch_embeds_, patch_hw_ = [], []
        for i in range(patch_embeds.shape[0]):
            if patch_masks[i, 0].sum() == 0:
                patch_embed = patch_embeds[i, :, :16, :16]
            else:
                nonzero_indices = torch.nonzero(patch_masks[i, 0], as_tuple=False)
                h1, w1 = nonzero_indices[0]
                h2, w2 = nonzero_indices[-1]
                patch_embed = patch_embeds[i, :, h1:h2+1, w1:w2+1]

            H, W = patch_embed.shape[1:]
            h, w = H // self.conv_stride, W // self.conv_stride
            patch_embed = self.local_attention(patch_embed.unsqueeze(0))
            class_embed = self.class_embedding[None, None, :].to(dtype=self.dtype)
            class_embed = self.global_attention(class_embed, patch_embed)[0]

            patch_embed = patch_embed.transpose(-2, -1).reshape(-1, h, w)
            split_embed = self.split_embedding[:, None, None].repeat(1, h, 1)
    
            patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)
            patch_embed = patch_embed.flatten(1).transpose(0, 1)
            patch_embeds_.append(modules(torch.cat([class_embed, patch_embed], dim=0)))
            patch_hw_.append(torch.LongTensor([h, w]).to(self.device))

        return patch_embeds_, patch_hw_
    
    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.patch_embedding.weight.device


class VisionCompressor(nn.Module):
    def __init__(self, clip_layer, processor, llm_size, num_layer, layer_rate=4, num_heads=8):
        super().__init__()

        self.layer_idx = num_layer - torch.arange(0, num_layer + 1, layer_rate)
        self.num_layer = len(self.layer_idx)

        self.clip_layer = clip_layer
        clip_size = clip_layer.hidden_size

        self.norm_layer = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(llm_size), nn.Linear(llm_size, clip_size, bias=False))
            for _ in range(self.num_layer)])

        self.input_layer = nn.Linear(llm_size, clip_size)
        self.output_layer = nn.Linear(clip_size, clip_size)
        self.num_heads = num_heads
        self.scale = clip_size ** -0.5
        
        max_patch = processor.image_size // processor.patch_stride // processor.conv_stride
        max_patch_clip = processor.image_size_clip // processor.patch_stride_clip
        assert max_patch % max_patch_clip == 0

        self.mask_stride_clip = processor.patch_stride_clip
        self.patch_ratio = max_patch // max_patch_clip

    def cross_attention(self, vision_input, vision_outputs):
        N, L, C = vision_outputs.shape

        patch_q = vision_outputs[:, :1].reshape(N, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = vision_outputs[:, 1:].reshape(N, L - 1, self.num_heads, -1).transpose(1, 2)

        cross_attn = (patch_q * self.scale * patch_kv).sum(-1).softmax(dim=-1)
        fuse_ouptuts = (cross_attn.unsqueeze(-1) * patch_kv).sum(-2)

        return vision_input + self.output_layer(fuse_ouptuts.reshape(N, -1))
    
    def compute_mseloss(self, pred_feature, clip_feature):
        loss_func = nn.CosineSimilarity(dim=-1)
        return 1 - loss_func(pred_feature, clip_feature).mean()
        
    def forward(self, input_ids, images, all_features, patch_hw):
        assert ((input_ids == IMAGE_TOKEN_INDEX).sum(-1) > 1).sum() == 0

        B, N, L, D = all_features.shape
        c_dtype, c_device = all_features.dtype, all_features.device

        clip_2d_masks = F.avg_pool2d(images[:, -1:, :, :], 
                                     kernel_size=self.mask_stride_clip, 
                                     stride=self.mask_stride_clip)
        assert len(torch.where(clip_2d_masks % 1)[0]) == 0
        clip_features = self.clip_layer(images[:, :-1, :, :])
        
        mse_loss, count_image = 0, 0
        for i in range(B):
            H, W = patch_hw[i]
            if_exist_image = IMAGE_TOKEN_INDEX in input_ids[i]
            idx_str = torch.where(input_ids[i] == IMAGE_TOKEN_INDEX)[0] if if_exist_image else 0
            idx_end = idx_str + min(N, H * (W + 1) + 1)

            i_features = all_features[i, idx_str:idx_end]
            if if_exist_image:
                i_features = i_features.permute(1, 2, 0)[:, :, 1:]
                i_features = i_features.reshape(L, D, H, W + 1)[:, :, :, :-1]
                if self.patch_ratio > 1:
                    i_features = F.avg_pool2d(i_features, 
                                              kernel_size=self.patch_ratio, 
                                              stride=self.patch_ratio)
                i_features = i_features.flatten(2).permute(2, 0, 1)

            vision_feature = []
            for j in range(self.num_layer):
                vision_feature.append(self.norm_layer[j](i_features[:, self.layer_idx[j]])) 
            vision_feature = self.cross_attention(self.input_layer(i_features[:, -1]),
                                                  torch.stack(vision_feature, dim=1))

            if if_exist_image:
                clip_1d_mask = clip_2d_masks[i, 0].view(-1).bool()
                clip_feature = clip_features[i][clip_1d_mask]

                count_image += 1
                mse_loss += self.compute_mseloss(vision_feature, clip_feature)
            else:
                mse_loss += self.compute_mseloss(vision_feature, vision_feature.detach())

        return mse_loss / max(count_image, 1.0)
    