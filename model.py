import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from module.utils import Embeddings, _get_clones

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

        decoder_layer = nn.TransformerDecoderLayer(d_model = config.hidden_dim, nhead = config.head_nums, 
                                                dropout = config.dropout, activation = F.relu, batch_first = True)
        self.decoder = _get_clones(decoder_layer, config.decoder_layer_nums)

        self.classify = nn.Linear(config.hidden_dim, config.vocab_size)
        self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

    def forward(self, img, caption_index = None, caption_mask = None, label = None, img_embed = None):

        if img_embed is None:
            img_embedding = self.image_encoder(img)[0]
        else:
            img_embedding = img_embed

        if caption_index is None:
            return img_embedding

        if caption_index.dim() == 3:
            img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
            caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
            caption_index = caption_index.reshape(-1, caption_index.size(-1))

        caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
        padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

        caption_embedding = self.caption_encoder(caption_index)

        out = caption_embedding
        for i in range(self.decoder_layer_nums):
            out = self.decoder[i](tgt = out, memory = img_embedding, tgt_mask = caption_mask, tgt_key_padding_mask = ~(padding_mask > 0))

        pred = self.classify(out)

        if label is None:
            return pred
        else:
            pred = pred.reshape(-1, self.vocab_size)
            label = label.reshape(-1)
            loss = self.loss_fun(pred, label)
            loss = loss / caption_index.size(0)
            return loss