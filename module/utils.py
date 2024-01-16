import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Any, Union, Callable
from torch import Tensor

def _sample(model, img, caption_index, config):
    isFin = torch.ones_like(caption_index).reshape(-1)
    isUnFin = torch.zeros_like(caption_index).reshape(-1)
    eos_index = torch.tensor([config.eos_token_id]).to(img.device)

    sum_log = torch.zeros(img.size(0)).to(img.device)
    # 计算一次图片编码, 加快解码速度
    img_embed = model(img)
    for i in range(config.generation_length):

        # 若某个句子已经到达结束符, 将其状态设置为已完成
        last_token = caption_index[:, -1]
        flag = torch.where(last_token == eos_index, isFin, isUnFin)

        if torch.sum(flag) == torch.sum(isFin):
            break

        caption_mask = config.generator_fun(caption_index.size(1)).to(img.device).unsqueeze(0).repeat(caption_index.size(0), 1, 1)
        pred = model(img, caption_index, caption_mask, img_embed = img_embed)
        next = pred[:, -1, :]

        # 蒙特卡洛采样
        score = next.softmax(dim = -1)
        sample_index = torch.multinomial(score, 1)
        # 取出采样概率
        logits = next.log_softmax(dim = -1)
        logits = torch.gather(logits, dim = -1, index = sample_index)
        logits = logits.reshape(-1)

        # 若某个句子到达结束符, 分数保持不变
        score_eos = torch.zeros_like(logits)
        next_score = torch.where(flag == 1, score_eos, logits)
        sum_log = sum_log + next_score

        # 若某个句子到达结束符, 只需要添加结束标签
        sample_index = sample_index.reshape(-1)
        add_eos = torch.empty_like(sample_index).fill_(eos_index[0])
        sample_index = torch.where(flag == 1, add_eos, sample_index).reshape(-1, 1)
        caption_index = torch.cat([caption_index, sample_index], dim = 1)

    return caption_index


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_caption_mask_attn_mask(generator_fun, length, device):
    attn_mask = generator_fun(length * 2).to(device)
    is_mask_token = torch.tensor([0, float('-inf')] * length).to(device).unsqueeze(0).repeat(2 * length, 1)
    attn_mask = is_mask_token + attn_mask
    # attn_mask = torch.where(torch.diag_embed(torch.diag(attn_mask)) == float('-inf'), torch.zeros_like(attn_mask), attn_mask)
    return attn_mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(512, config.hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class WeightModel(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super(WeightModel, self).__init__()
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, y):
        weight = torch.sigmoid(self.linear1(torch.cat([x, y], dim = -1)))
        return weight


class ObjectModel(nn.Module):
    def __init__(self, config) -> None:
        super(ObjectModel, self).__init__()

        self.k = config.centre_nums
        self.head_nums = config.head_nums

        self.centre = nn.Parameter(torch.zeros(self.k, config.hidden_dim).data, requires_grad = False)

        encoder = nn.TransformerEncoderLayer(d_model = config.hidden_dim, nhead = config.head_nums, 
                                                dropout = config.dropout, activation = F.relu, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder, config.encoder_layer_nums)

    def forward(self, img_embedding, updata_centre):
        bs = img_embedding.size(0)
        img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))

        with torch.no_grad():
            label = self.kmeans(img_embedding, updata_centre)

        label = label.reshape(bs, -1)
        img_embedding = img_embedding.reshape(bs, -1, img_embedding.size(-1))

        attn_mask = (label.unsqueeze(1).repeat(1, img_embedding.size(1), 1) == label.unsqueeze(2).repeat(1, 1, img_embedding.size(1))).to(torch.float32)
        attn_mask = torch.where(attn_mask == 1, torch.zeros_like(attn_mask), torch.empty_like(attn_mask).fill_(float('-inf')))
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, attn_mask.size(-2), attn_mask.size(-1))
        img_embedding = self.encoder(img_embedding, mask = attn_mask)

        return img_embedding

    def kmeans(self, x, updata_centre):
        n = x.size(0)
        all_dist = x.unsqueeze(1).repeat(1, self.k, 1) - self.centre.unsqueeze(0).repeat(n, 1, 1)
        all_dist = torch.sqrt(torch.pow(torch.sum(all_dist, dim = -1), 2))
        label = all_dist.argmin(-1)

        if updata_centre:
            centre = self.centre
            for i in range(10):
                #开始聚类
                all_dist = x.unsqueeze(1).repeat(1, self.k, 1) - centre.unsqueeze(0).repeat(n, 1, 1)
                all_dist = torch.sqrt(torch.pow(torch.sum(all_dist, dim = -1), 2))
                new_label = all_dist.argmin(-1)
                # 更新聚类中心
                one_hot_label = F.one_hot(new_label, self.k).to(torch.float32).permute(1, 0)
                new_centre = torch.matmul(one_hot_label, x)
                centre = new_centre + centre
                centre = centre / (torch.sum(one_hot_label, dim = -1) + 1).unsqueeze(1).repeat(1, self.centre.size(-1))
            # self.centre.data = centre
            dist.all_reduce(centre)
            self.centre.data = centre / dist.get_world_size()
        return label


class FuseFormer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FuseFormer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.get_weight1 = WeightModel(d_model)
        self.get_weight2 = WeightModel(d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(FuseFormer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
                memory_1 = None, memory_1_mask = None, memory_1_key_padding_mask = None, attn_mask = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

            if memory_1 is None:
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            else:
                x1 = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                x2 = self._mha_block_1(self.norm2(x), memory_1, memory_1_mask, memory_1_key_padding_mask)
                x1 = self.get_weight1(x, x1) * x1
                x2 = self.get_weight2(x, x2) * x2
                x = x + x1 + x2

            if attn_mask is not None:
                x = x.reshape(x.size(0), -1, 2, x.size(-1))
                x_0 = x[:, :, 0, :]
                x_1 = x[:, :, 1, :]
                tgt_key_padding_mask = tgt_key_padding_mask.reshape(tgt_key_padding_mask.size(0), -1, 2)

                x_0 = x_0 + self._sa_block_1(self.norm4(x_0), attn_mask, tgt_key_padding_mask[:, :, 0])
                x_1 = x_1 + self._sa_block_1(self.norm4(x_1), attn_mask, tgt_key_padding_mask[:, :, 1])

                x = torch.cat([x_0.unsqueeze(2), x_1.unsqueeze(2)], dim = 2)
                x = x.reshape(x.size(0), -1, x.size(-1))

            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

            if memory_1 is None:
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            else:
                x1 = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
                x2 = self._mha_block_1(x, memory_1, memory_1_mask, memory_1_key_padding_mask)
                x1 = self.get_weight1(x, x1) * x1
                x2 = self.get_weight2(x, x2) * x2
                x = self.norm2(x + x1 + x2)

            if attn_mask is not None:
                x = x.reshape(x.size(0), -1, 2, x.size(-1))
                x_0 = x[:, :, 0, :]
                x_1 = x[:, :, 1, :]
                tgt_key_padding_mask = tgt_key_padding_mask.reshape(tgt_key_padding_mask.size(0), -1, 2)

                x_0 = self.norm4(x_0 + self._sa_block_1(x_0, attn_mask, tgt_key_padding_mask[:, :, 0]))
                x_1 = self.norm4(x_1 + self._sa_block_1(x_1, attn_mask, tgt_key_padding_mask[:, :, 1]))

                x = torch.cat([x_0.unsqueeze(2), x_1.unsqueeze(2)], dim = 2)
                x = x.reshape(x.size(0), -1, x.size(-1))

            x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _sa_block_1(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn1(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout4(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # multihead attention block
    def _mha_block_1(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn1(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout5(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# class FuseFormer(nn.Module):
#     __constants__ = ['batch_first', 'norm_first']
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(FuseFormer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout4 = nn.Dropout(dropout)

#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(FuseFormer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, attn_mask = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
#             x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)

#             if attn_mask is not None:
#                 x = x.reshape(x.size(0), -1, 2, x.size(-1))
#                 x_0 = x[:, :, 0, :]
#                 x_1 = x[:, :, 1, :]
#                 tgt_key_padding_mask = tgt_key_padding_mask.reshape(tgt_key_padding_mask.size(0), -1, 2)

#                 x_0 = x_0 + self._sa_block_1(self.norm4(x_0), attn_mask, tgt_key_padding_mask[:, :, 0])
#                 x_1 = x_1 + self._sa_block_1(self.norm4(x_1), attn_mask, tgt_key_padding_mask[:, :, 1])

#                 x = torch.cat([x_0.unsqueeze(2), x_1.unsqueeze(2)], dim = 2)
#                 x = x.reshape(x.size(0), -1, x.size(-1))

#             x = x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
#             x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))

#             if attn_mask is not None:
#                 x = x.reshape(x.size(0), -1, 2, x.size(-1))
#                 x_0 = x[:, :, 0, :]
#                 x_1 = x[:, :, 1, :]
#                 tgt_key_padding_mask = tgt_key_padding_mask.reshape(tgt_key_padding_mask.size(0), -1, 2)

#                 x_0 = self.norm4(x_0 + self._sa_block_1(x_0, attn_mask, tgt_key_padding_mask[:, :, 0]))
#                 x_1 = self.norm4(x_1 + self._sa_block_1(x_1, attn_mask, tgt_key_padding_mask[:, :, 1]))

#                 x = torch.cat([x_0.unsqueeze(2), x_1.unsqueeze(2)], dim = 2)
#                 x = x.reshape(x.size(0), -1, x.size(-1))

#             x = self.norm3(x + self._ff_block(x))
#         return x

#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)

#     def _sa_block_1(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn1(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout4(x)

#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout2(x)

#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)


# class FuseFormer(nn.Module):
#     __constants__ = ['batch_first', 'norm_first']
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(FuseFormer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)
#         self.get_weight1 = WeightModel(d_model)
#         self.get_weight2 = WeightModel(d_model)

#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(FuseFormer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
#                 memory_1 = None, memory_1_mask = None, memory_1_key_padding_mask = None, attn_mask = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

#             if memory_1 is None:
#                 x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#             else:
#                 x1 = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#                 x2 = self._mha_block_1(self.norm2(x), memory_1, memory_1_mask, memory_1_key_padding_mask)
#                 x1 = self.get_weight1(x, x1) * x1
#                 x2 = self.get_weight2(x, x2) * x2
#                 x = x + x1 + x2

#             x = x + self._sa_block_1(self.norm4(x), attn_mask, tgt_key_padding_mask)
#             x = x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

#             if memory_1 is None:
#                 x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             else:
#                 x1 = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
#                 x2 = self._mha_block_1(x, memory_1, memory_1_mask, memory_1_key_padding_mask)
#                 x1 = self.get_weight1(x, x1) * x1
#                 x2 = self.get_weight2(x, x2) * x2
#                 x = self.norm2(x + x1 + x2)

#             x = self.norm4(x + self._sa_block_1(x, attn_mask, tgt_key_padding_mask))
#             x = self.norm3(x + self._ff_block(x))
#         return x

#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)

#     def _sa_block_1(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn1(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout4(x)

#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout2(x)

#     # multihead attention block
#     def _mha_block_1(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn1(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout5(x)

#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)


# class FuseFormer(nn.Module):
#     __constants__ = ['batch_first', 'norm_first']
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(FuseFormer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.dropout5 = nn.Dropout(dropout)
#         self.get_weight1 = WeightModel(d_model)
#         self.get_weight2 = WeightModel(d_model)

#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(FuseFormer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
#                 memory_1 = None, memory_1_mask = None, memory_1_key_padding_mask = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

#             if memory_1 is None:
#                 x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#             else:
#                 x1 = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#                 x2 = self._mha_block_1(self.norm2(x), memory_1, memory_1_mask, memory_1_key_padding_mask)
#                 x1 = self.get_weight1(x, x1) * x1
#                 x2 = self.get_weight2(x, x2) * x2
#                 x = x + x1 + x2

#             x = x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

#             if memory_1 is None:
#                 x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             else:
#                 x1 = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
#                 x2 = self._mha_block_1(x, memory_1, memory_1_mask, memory_1_key_padding_mask)
#                 x1 = self.get_weight1(x, x1) * x1
#                 x2 = self.get_weight2(x, x2) * x2
#                 x = self.norm2(x + x1 + x2)

#             x = self.norm3(x + self._ff_block(x))
#         return x

#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)

#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout2(x)

#     # multihead attention block
#     def _mha_block_1(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn1(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout5(x)

#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)