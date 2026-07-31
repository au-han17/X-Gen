from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel
from scipy.optimize import linear_sum_assignment
from transformers.models.bert.modeling_bert import (BertEncoder,BertPredictionHeadTransform,BertLMPredictionHead,BertEmbeddings)
from transformers.models.bert.configuration_bert import BertConfig

from .tokenizers import modeCLS,modePAD,modeBOS


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) #(4,8,1,96)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #(4,8,1,1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    #将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value
    p_attn = F.softmax(scores, dim=-1) #(4,8,1,1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
# torch.size[1,1,1] tensor([[True]])

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask,*,cap):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask,cap=cap)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        # return (4,98,768)
    def decode(self, hidden_states, src_mask, tgt, tgt_mask,*,cap):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory,cap=cap)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)
# hidden_states [4,98,768] src_mask[4,1,98] tgt[4,1] [4,4,4,4] [[[True]]]

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)
# x (4,1,768) src_mask (4,1,98) tgt_msk [[[True]]] memory (4,1,2304)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states #(4,98,768)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory,flag=0)# x (4,1,768)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory,flag=1)
        return self.sublayer[2](x, self.feed_forward, memory,flag=2)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.re_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory,*,flag):
        #if flag == 0 or flag == 1:
        return x + self.dropout(sublayer(self.norm(x, memory)))
        #else:
            #return x + self.dropout(sublayer(self.re_norm(x)))

class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory) #(4,1,768)
        delta_beta = self.mlp_beta(memory)#(4,1,768)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) #(1,1,1,1) [[[[True]]]]
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
#(4,8,1,96)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
#self.attn (4,8,1,1) attention score
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)#(4,1,768)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
# x (4,1) 4  lut(x) (4,1,768)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
#self.pe[:, :x.size(1)] (1,1,768)

class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
     
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        # k = (batch_size,num_slots+1,d_model)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory
    # def add_mode(self,inputs,*,cap):
    #     if len(inputs.shape) == 3 and len(cap.shape) == 2:
    #         cap = cap.unsqueeze(1)  
    #     elif len(inputs.shape) == 3 and len(cap.shape) == 1:
    #         cap = cap.unsqueeze(0).unsqueeze(1) 
    #     elif len(inputs.shape) == 2 and len(cap.shape) == 2:
    #         raise ValueError("Tensor shapes are not compatible for broadcasting.")
    #     elif len(inputs.shape) == 2 and len(cap.shape) == 1:
    #         cap = cap.unsqueeze(0)  
    #     else:
    #         raise ValueError("Unsupported tensor shapes.")

    #     result = inputs + cap
    #     return result
    def forward(self, inputs, memory,*,cap):
        outputs = []
        inputs = inputs+cap
        #inputs (4,1,768)
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)
#memory (4,2304) outputs (4,1,2304)
        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        caption_model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm)
    
        for p in caption_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return caption_model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            #seq = seq[:, :]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None,*,cap):

        start_token = seq.new_full([seq.shape[0], 1], modeBOS)
        seq = torch.cat([start_token, seq], dim=1)
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask,cap=cap)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask,*,cap):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device),cap=cap)
        return out[:, -1], [ys.unsqueeze(0)]

class ModeEncoder(nn.Module):
    def __init__(self,cfg,num_layers):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
        )
        self.embedding_layer = BertEmbeddings(bert_cfg)
        self.encoder = BertEncoder(bert_cfg)

        self.mode_transform = BertPredictionHeadTransform(bert_cfg)
        self.mode_linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, token_ids):
        mode_token = token_ids.new_full([token_ids.size(0), 1], modeCLS)
        #The purpose of this operation is to create a tensor containing the CLS token with the same batch size as the input token_ids
        token_ids = torch.cat([mode_token, token_ids], dim=1)
        #add cls token infront of every tensor
        embeds = self.embedding_layer(token_ids)

        attn_mask = (token_ids != modePAD).float()
        attn_mask = attn_mask[:, None, None, :]
        attn_mask = (1.0 - attn_mask) * -10000.0

        hidden_state = self.encoder(embeds, attn_mask).last_hidden_state
        mode_embed = self.mode_transform(hidden_state[:, 0, :])
        #extracts the first token's hidden state from the input sequence for all sequences in the batch, and then applies a transformation to these hidden states using the mode_transform module
        mode_embed = self.mode_linear(mode_embed)
        return mode_embed

class CodeBook(nn.Module):
    def __init__(self, cfg):
        super(CodeBook, self).__init__()
        self.embedding = nn.Embedding(cfg.num_modes, cfg.hidden_size)
        self.commitment_cost = cfg.loss.commitment_cost

    def forward(self, mode_emb, splits):
        
        distances = (torch.sum(mode_emb**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(mode_emb, self.embedding.weight.t())).sqrt()
     
        distances = distances.split(splits)
        indices = [
            linear_sum_assignment(d.detach().cpu().numpy())[1]
            for d in distances
        ]
        frequency = indices.copy()
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(mode_emb.device)
        #print("indices: ", indices.unique())
        quantized = self.embedding(indices)
        #The resulting indices list contains the optimal column indices for each smaller tensor, representing the optimal assignment between the elements of the two sets based on their pairwise distances.
		# Loss
        q_latent_loss = F.mse_loss(mode_emb.detach(), quantized) + \
                        F.mse_loss(mode_emb.mean(dim=0).detach(), 
                                   self.embedding.weight.mean(dim=0))
        e_latent_loss = F.mse_loss(mode_emb, quantized.detach()) + \
                        F.mse_loss(mode_emb.mean(dim=0), 
                                   self.embedding.weight.mean(dim=0).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = mode_emb + (quantized - mode_emb).detach()

        return loss, quantized[:, None, :], frequency
    
class MaskDecoder(nn.Module):
    def __init__(self, cfg, num_layers,vocab_size ):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            num_hidden_layers=num_layers,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.embedding_layer = ModeEmbeddings(bert_cfg,vocab_size)
        self.encoder = BertEncoder(bert_cfg)
        self.classifier = BertLMPredictionHead(bert_cfg)
        self.feat_embed = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 768),
            nn.Dropout(0.1)
        )
        self.layernorm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)

        position_ids = torch.arange(bert_cfg.max_position_embeddings)
        attn_mask = position_ids[None, None, :].repeat(
            1, bert_cfg.max_position_embeddings, 1
        ) <= position_ids[None, :, None]
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, attn_feats, token_ids, mode_embed,img_ids):
        batch_size = token_ids.size(0)
        region_feats = self.feat_embed(attn_feats)
        img_feat = self.layernorm(region_feats)
        img_feat = self.dropout(img_feat)

        
        mode_img_feats = img_feat[img_ids]


        start_token = token_ids.new_full([batch_size, 1], modeBOS)
        token_ids = torch.cat([start_token, token_ids], dim=1)
        embeds = self.embedding_layer(token_ids, mode_embed)

        seq_len = token_ids.size(1)
        attn_mask = self.attn_mask[:, :seq_len, :seq_len]
        attn_mask = attn_mask.repeat(batch_size, 1, 1)
        attn_mask = attn_mask[:, None, :, :].float()
        attn_mask = (1.0 - attn_mask) * -10000.0

        hidden_state = self.encoder(embeds, 
                                    attention_mask=attn_mask, 
                                    encoder_hidden_states=mode_img_feats)

        preds = self.classifier(hidden_state.last_hidden_state)
        return preds
    
class ModeEmbeddings(nn.Module):
    def __init__(self, bert_cfg,vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, 
                                            bert_cfg.hidden_size, 
                                            padding_idx=modePAD)
        self.position_embeddings = nn.Embedding(512, bert_cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_cfg.hidden_size,
                                       eps=bert_cfg.layer_norm_eps)
        self.dropout = nn.Dropout(bert_cfg.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", 
            torch.arange(bert_cfg.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids, mode_embeds):
        position_ids = self.position_ids[:, :input_ids.size(1)]
        position_embeds = self.position_embeddings(position_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + position_embeds + mode_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MaskDecodeLoss(nn.Module):
    def __init__(self):
        super(MaskDecodeLoss, self).__init__()

    def forward(self, pred, target):

        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1),1)

        loss = torch.sum(-true_dist * pred, dim=-1)

        return torch.mean(loss)
    
