import torch, copy
import torch.nn as nn

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