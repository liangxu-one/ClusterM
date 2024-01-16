import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from module.utils import Embeddings, FuseFormer, ObjectModel, _get_clones, _get_caption_mask_attn_mask

class CaptionModel(nn.Module):
    def __init__(self, config) -> None:
        super(CaptionModel, self).__init__()

        self.head_nums = config.head_nums
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.mask_token_id = config.mask_token_id

        self.generation_length = config.generation_length
        self.img_length = config.img_length
        self.max_length = config.max_length
        self.sentence_nums = config.sentence_nums
        self.encoder_layer_nums = config.encoder_layer_nums
        self.decoder_layer_nums = config.decoder_layer_nums

        self.generator_fun = config.generator_fun

        self.image_encoder = AutoModel.from_pretrained(config.vision_path)
        self.caption_encoder = Embeddings(config)

        decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
                                                dropout = config.dropout, activation = F.relu, batch_first = True)
        self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

        self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
        self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

        self.attn_mask = nn.Parameter(torch.randn((config.max_length, config.max_length)).data, requires_grad = True)

        self.object_model = ObjectModel(config)

    def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

        if img_embed is None:
            img_embedding = self.image_encoder(img)[0]
        else:
            img_embedding = img_embed

        if caption_index is None:
            return img_embedding

        region_img_embedding = self.object_model(img_embedding, updata_centre)

        if caption_index.dim() == 3:
            img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
            region_img_embedding = region_img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, region_img_embedding.size(-2), region_img_embedding.size(-1))
            caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
            caption_index = caption_index.reshape(-1, caption_index.size(-1))

        caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
        padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

        caption_embedding = self.caption_encoder(caption_index)

        mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
        mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
        caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
        caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

        attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
        caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

        fuse_attn_mask = self.attn_mask + self.generator_fun(self.attn_mask.size(0)).to(self.attn_mask.device)
        fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 0)[0]
        fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 1)[0]

        all_out = []
        out = caption_mask_embedding
        for i in range(self.decoder_layer_nums):
            out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), memory_1 = region_img_embedding, attn_mask = fuse_attn_mask)
            all_out.append(out)

        out = out.reshape(out.size(0), -1, 2, out.size(-1))
        out = out[:, :, 1, :]

        pred = self.classify(out)

        if label is None:
            return pred
        else:
            pred = pred.reshape(-1, self.vocab_size)
            label = label.reshape(-1)
            loss = self.loss_fun(pred, label)
            loss = loss / caption_index.size(0)

            if self.decoder_layer_nums > 1:
                loss_kl = 0
                for i in range(self.decoder_layer_nums - 1):
                    out = all_out[i]
                    out = out.reshape(out.size(0), -1, 2, out.size(-1))
                    out_0 = out[:, :, 0, :][:, 1:, :]
                    out_1 = out[:, :, 1, :][:, :-1, :]
                    loss_dist = F.kl_div(out_1.log_softmax(dim = -1), out_0.softmax(dim = -1), reduction = 'none')
                    loss_dist = torch.sum(loss_dist, dim = -1) * padding_mask[:, 1:]
                    loss_dist = torch.sum(loss_dist)
                    loss_kl = loss_kl + loss_dist
                loss_kl = loss_kl / (caption_index.size(0) * (self.decoder_layer_nums - 1))
                loss = loss + loss_kl

            return loss

# # 只使用网格特征
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.attn_mask = nn.Parameter(torch.randn((config.max_length, config.max_length)).data, requires_grad = True)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
#         mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
#         caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
#         caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

#         attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
#         caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

#         fuse_attn_mask = self.attn_mask + self.generator_fun(self.attn_mask.size(0)).to(self.attn_mask.device)
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 0)[0]
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 1)[0]

#         all_out = []
#         out = caption_mask_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), attn_mask = fuse_attn_mask)
#             all_out.append(out)

#         out = out.reshape(out.size(0), -1, 2, out.size(-1))
#         out = out[:, :, 1, :]

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             if self.decoder_layer_nums > 1:
#                 loss_kl = 0
#                 for i in range(self.decoder_layer_nums - 1):
#                     out = all_out[i]
#                     out = out.reshape(out.size(0), -1, 2, out.size(-1))
#                     out_0 = out[:, :, 0, :][:, 1:, :]
#                     out_1 = out[:, :, 1, :][:, :-1, :]
#                     loss_dist = F.kl_div(out_1.log_softmax(dim = -1), out_0.softmax(dim = -1), reduction = 'none')
#                     loss_dist = torch.sum(loss_dist, dim = -1) * padding_mask[:, 1:]
#                     loss_dist = torch.sum(loss_dist)
#                     loss_kl = loss_kl + loss_dist
#                 loss_kl = loss_kl / (caption_index.size(0) * (self.decoder_layer_nums - 1))
#                 loss = loss + loss_kl

#             return loss

# # 只使用区域特征
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.attn_mask = nn.Parameter(torch.randn((config.max_length, config.max_length)).data, requires_grad = True)

#         self.object_model = ObjectModel(config)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         img_embedding = self.object_model(img_embedding, updata_centre)

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
#         mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
#         caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
#         caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

#         attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
#         caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

#         fuse_attn_mask = self.attn_mask + self.generator_fun(self.attn_mask.size(0)).to(self.attn_mask.device)
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 0)[0]
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 1)[0]

#         all_out = []
#         out = caption_mask_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), attn_mask = fuse_attn_mask)
#             all_out.append(out)

#         out = out.reshape(out.size(0), -1, 2, out.size(-1))
#         out = out[:, :, 1, :]

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             if self.decoder_layer_nums > 1:
#                 loss_kl = 0
#                 for i in range(self.decoder_layer_nums - 1):
#                     out = all_out[i]
#                     out = out.reshape(out.size(0), -1, 2, out.size(-1))
#                     out_0 = out[:, :, 0, :][:, 1:, :]
#                     out_1 = out[:, :, 1, :][:, :-1, :]
#                     loss_dist = F.kl_div(out_1.log_softmax(dim = -1), out_0.softmax(dim = -1), reduction = 'none')
#                     loss_dist = torch.sum(loss_dist, dim = -1) * padding_mask[:, 1:]
#                     loss_dist = torch.sum(loss_dist)
#                     loss_kl = loss_kl + loss_dist
#                 loss_kl = loss_kl / (caption_index.size(0) * (self.decoder_layer_nums - 1))
#                 loss = loss + loss_kl

#             return loss

# # 移除mask机制
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.attn_mask = nn.Parameter(torch.randn((config.max_length, config.max_length)).data, requires_grad = True)

#         self.object_model = ObjectModel(config)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         region_img_embedding = self.object_model(img_embedding, updata_centre)

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             region_img_embedding = region_img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, region_img_embedding.size(-2), region_img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         fuse_attn_mask = self.attn_mask + self.generator_fun(self.attn_mask.size(0)).to(self.attn_mask.device)
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 0)[0]
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 1)[0]

#         out = caption_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = caption_mask, tgt_key_padding_mask = ~(padding_mask > 0), memory_1 = region_img_embedding, attn_mask = fuse_attn_mask)

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             return loss

# # 移除KL散度目标函数
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.attn_mask = nn.Parameter(torch.randn((config.max_length, config.max_length)).data, requires_grad = True)

#         self.object_model = ObjectModel(config)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         region_img_embedding = self.object_model(img_embedding, updata_centre)

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             region_img_embedding = region_img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, region_img_embedding.size(-2), region_img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
#         mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
#         caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
#         caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

#         attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
#         caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

#         fuse_attn_mask = self.attn_mask + self.generator_fun(self.attn_mask.size(0)).to(self.attn_mask.device)
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 0)[0]
#         fuse_attn_mask = torch.split(fuse_attn_mask, caption_embedding.size(1), dim = 1)[0]

#         all_out = []
#         out = caption_mask_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), memory_1 = region_img_embedding, attn_mask = fuse_attn_mask)
#             all_out.append(out)

#         out = out.reshape(out.size(0), -1, 2, out.size(-1))
#         out = out[:, :, 1, :]

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             return loss

# # 移除RP_MHA模块
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.object_model = ObjectModel(config)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         region_img_embedding = self.object_model(img_embedding, updata_centre)

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             region_img_embedding = region_img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, region_img_embedding.size(-2), region_img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
#         mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
#         caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
#         caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

#         attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
#         caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

#         all_out = []
#         out = caption_mask_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), memory_1 = region_img_embedding)
#             all_out.append(out)

#         out = out.reshape(out.size(0), -1, 2, out.size(-1))
#         out = out[:, :, 1, :]

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             if self.decoder_layer_nums > 1:
#                 loss_kl = 0
#                 for i in range(self.decoder_layer_nums - 1):
#                     out = all_out[i]
#                     out = out.reshape(out.size(0), -1, 2, out.size(-1))
#                     out_0 = out[:, :, 0, :][:, 1:, :]
#                     out_1 = out[:, :, 1, :][:, :-1, :]
#                     loss_dist = F.kl_div(out_1.log_softmax(dim = -1), out_0.softmax(dim = -1), reduction = 'none')
#                     loss_dist = torch.sum(loss_dist, dim = -1) * padding_mask[:, 1:]
#                     loss_dist = torch.sum(loss_dist)
#                     loss_kl = loss_kl + loss_dist
#                 loss_kl = loss_kl / (caption_index.size(0) * (self.decoder_layer_nums - 1))
#                 loss = loss + loss_kl

#             return loss

# # 移除RP_weight
# class CaptionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super(CaptionModel, self).__init__()

#         self.head_nums = config.head_nums
#         self.hidden_dim = config.hidden_dim
#         self.vocab_size = config.vocab_size
#         self.pad_token_id = config.pad_token_id
#         self.bos_token_id = config.bos_token_id
#         self.eos_token_id = config.eos_token_id
#         self.mask_token_id = config.mask_token_id

#         self.generation_length = config.generation_length
#         self.img_length = config.img_length
#         self.max_length = config.max_length
#         self.sentence_nums = config.sentence_nums
#         self.encoder_layer_nums = config.encoder_layer_nums
#         self.decoder_layer_nums = config.decoder_layer_nums

#         self.generator_fun = config.generator_fun

#         self.image_encoder = AutoModel.from_pretrained(config.vision_path)
#         self.caption_encoder = Embeddings(config)

#         decoder_layer = FuseFormer(d_model = config.hidden_dim, nhead = config.head_nums, 
#                                                 dropout = config.dropout, activation = F.relu, batch_first = True)
#         self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

#         self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
#         self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

#         self.object_model = ObjectModel(config)

#     def forward(self, img, caption_index = None, caption_mask = None, label = None, updata_centre = False, img_embed = None):

#         if img_embed is None:
#             img_embedding = self.image_encoder(img)[0]
#         else:
#             img_embedding = img_embed

#         if caption_index is None:
#             return img_embedding

#         region_img_embedding = self.object_model(img_embedding, updata_centre)

#         if caption_index.dim() == 3:
#             img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
#             region_img_embedding = region_img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, region_img_embedding.size(-2), region_img_embedding.size(-1))
#             caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#             caption_index = caption_index.reshape(-1, caption_index.size(-1))

#         caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
#         padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

#         caption_embedding = self.caption_encoder(caption_index)

#         mask_index = torch.empty((caption_index.size(0), caption_index.size(1) + 1), dtype = caption_index.dtype, device = caption_index.device).fill_(self.mask_token_id)
#         mask_embedding = self.caption_encoder(mask_index)[:, 1:, :]
#         caption_mask_embedding = torch.cat([caption_embedding.unsqueeze(2), mask_embedding.unsqueeze(2)], dim = 2)
#         caption_mask_embedding = caption_mask_embedding.reshape(caption_embedding.size(0), -1, caption_embedding.size(-1))

#         attn_mask = _get_caption_mask_attn_mask(self.generator_fun, caption_index.size(1), caption_index.device)
#         caption_mask_padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, 2).reshape(caption_index.size(0), -1)

#         all_out = []
#         out = caption_mask_embedding
#         for i in range(self.decoder_layer_nums):
#             out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = attn_mask, tgt_key_padding_mask = ~(caption_mask_padding_mask > 0), memory_1 = region_img_embedding, attn_mask = caption_mask)
#             all_out.append(out)

#         out = out.reshape(out.size(0), -1, 2, out.size(-1))
#         out = out[:, :, 1, :]

#         pred = self.classify(out)

#         if label is None:
#             return pred
#         else:
#             pred = pred.reshape(-1, self.vocab_size)
#             label = label.reshape(-1)
#             loss = self.loss_fun(pred, label)
#             loss = loss / caption_index.size(0)

#             if self.decoder_layer_nums > 1:
#                 loss_kl = 0
#                 for i in range(self.decoder_layer_nums - 1):
#                     out = all_out[i]
#                     out = out.reshape(out.size(0), -1, 2, out.size(-1))
#                     out_0 = out[:, :, 0, :][:, 1:, :]
#                     out_1 = out[:, :, 1, :][:, :-1, :]
#                     loss_dist = F.kl_div(out_1.log_softmax(dim = -1), out_0.softmax(dim = -1), reduction = 'none')
#                     loss_dist = torch.sum(loss_dist, dim = -1) * padding_mask[:, 1:]
#                     loss_dist = torch.sum(loss_dist)
#                     loss_kl = loss_kl + loss_dist
#                 loss_kl = loss_kl / (caption_index.size(0) * (self.decoder_layer_nums - 1))
#                 loss = loss + loss_kl

#             return loss