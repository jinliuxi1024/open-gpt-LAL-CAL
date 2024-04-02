
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gpt_tokenizer
class NewGELU(nn.Module):
      def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super(CausalSelfAttention, self).__init__()
        self.device = torch.device('mps')
        self.embed_size = config.embed_size
        self.head_size = config.head_size
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_attn = nn.Linear(self.embed_size, self.embed_size * 3)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.c_proj = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, x):
    # x: [batch_size, block_size]
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.embed_size, dim=-1)
        q = q.view(B, T, self.head_size, C // self.head_size).transpose(1, 2)
        k = k.view(B, T, self.head_size, C // self.head_size).transpose(1, 2)
        v = v.view(B, T, self.head_size, C // self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1))*(1.0*math.sqrt(k.size(-1)))
    # att: [B, head_size, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
    # y: [B, T, C]
        return y

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.embed_size, 4 * config.embed_size),
            c_proj=nn.Linear(4 * config.embed_size, config.embed_size),
            act=NewGELU(),
            dropout=nn.Dropout(0.1),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPTmodel(nn.Module):
    def __init__(self, config):
        super(GPTmodel, self).__init__()
        self.device = torch.device('mps')
        self.embed_size = config.embed_size
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.head_size = config.head_size
        self.layer_size = config.layer_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_size),
            wpe=nn.Embedding(config.block_size, config.embed_size),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block(config) for _ in range(config.layer_size)]),
            ln_f=nn.LayerNorm(config.embed_size),
        ))
        self.pro_out = nn.Linear(config.embed_size, config.vocab_size, bias=False)

    def forward(self, idx, target=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.pro_out(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_len=100):
     # idx: [1, block_size]
     # 先输出一下
      token = idx.view(-1).tolist()
      print(gpt_tokenizer.Tokenizer().decode(token), end='')
      for _ in range(max_new_len):
          idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
          output, _ = self.forward(idx_cond.to(self.device))
          output = output[:, -1, :]
          prob = F.softmax(output, dim=-1)
          idx_new = torch.multinomial(prob, num_samples=1).to(self.device)
          # 输出最新生成的字符，但是不要换行
          token = idx_new.view(-1).tolist()
          print(gpt_tokenizer.Tokenizer().decode(token), end='')
          idx = torch.cat([idx, idx_new], dim=1).to(self.device)

      return idx











