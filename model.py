
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys, math
from typing import List, Tuple, Dict, Set, Union
from utils import *

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class CharCNNEmbedder(nn.Module):
    def __init__(self, char_size, char_vec_size, kernel_width, num_kernels, 
        num_highway_layers, one_hot=False):
        super(CharCNNEmbedder, self).__init__()
        if one_hot:
            self.char_emb = nn.Linear(char_size, char_vec_size)
        else:
            self.char_emb = nn.Embedding(char_size, char_vec_size)
        self.conv = nn.Conv1d(char_vec_size, num_kernels, kernel_width)
        self.highways = Highway(num_kernels, num_layers=num_highway_layers)
        self.modules = nn.Sequential(self.char_emb, self.conv, self.highways)
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        bsz = x.size(1)
        x = self.char_emb(x)
        x = self.conv(x.view(-1, x.size(2),x.size(3)).transpose(1,2))
        x = self.tanh(x)
        x = self.highways(x.max(dim=2)[0])
        return x.view(-1, bsz, x.size(1))


class Highway(nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(
            self,
            input_dim: int,
            num_layers: int = 1
    ):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(num_layers)])
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)

            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(
            self,
            x: torch.Tensor
    ):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            # ones = torch.ones_like(gate)
            # x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
            x = gate * x + (1 - gate) * proj_x
        return x

class CharLSTM(nn.Module):

    def __init__(self,  args, char_size, target_size, device, input_feed=True, pad_id=0):
        super(CharLSTM, self).__init__()

        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout
        self.input_feed = input_feed
        self.device = device
        self.pad_id = pad_id
        self.one_hot = args.one_hot

        # initialize neural network layers...

        self.src_embed = CharCNNEmbedder(char_size, args.char_vec_size, args.kernel_width, args.num_kernels, 
            args.num_highway_layers, args.one_hot)
        self.tgt_embed = nn.Embedding(target_size, self.embed_size, padding_idx=pad_id)

        self.encoder_lstm = nn.LSTM(args.num_kernels, self.hidden_size, bidirectional=True)
        decoder_lstm_input = self.embed_size + self.hidden_size if self.input_feed else self.embed_size
        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, self.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's state space
        self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(self.hidden_size, target_size, bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, src_sents_var, tgt_sents_var, src_sents_len):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

        # src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        att_vecs = self.decode(src_encodings, decoder_init_vec, tgt_sents_var[:-1])

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)


        # (tgt_sent_len, batch_size)
        tgt_words_mask = (tgt_sents_var != self.pad_id).float()

        # (tgt_sent_len - 1, batch_size)
        tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # tgt_vocab_size = tgt_words_log_prob.size(2)
        # score = F.nll_loss(tgt_words_log_prob.view(-1,tgt_vocab_size), tgt_sents_var[1:].view(-1),
        #                     reduction='none')[(tgt_sents_var[1:]!=self.pad_id).view(-1)]

        # (batch_size) nll loss
        loss = -tgt_gold_words_log_prob.sum(dim=0).sum()
        
        return loss

    # def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
    #     src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
    #     for e_id, src_len in enumerate(src_sents_len):
    #         src_sent_masks[e_id, src_len:] = 1

    #     return src_sent_masks.to(self.device)

    def encode(self, src_sents_var, src_sent_lens):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        # (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        # packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)
        src_word_embeds = src_word_embeds[:src_sent_lens]

        # src_encodings: (src_sent_len, batch_size, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(src_word_embeds)
        # src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_encodings, (dec_init_state, dec_init_cell)

    def decode(self, src_encodings: torch.Tensor, 
               decoder_init_vec: Tuple[torch.Tensor, torch.Tensor], tgt_sents_var: torch.Tensor) -> torch.Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x: torch.Tensor,
             h_tm1: Tuple[torch.Tensor, torch.Tensor],
             src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

        if mask is not None:
            att_weight.data.masked_fill_(mask.byte(), -float('inf'))

        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight

    def beam_search(self, src_sents_var, src_sent_len, vocab, beam_size: int=5, max_decoding_time_step: int=70, maxwordlength: int=35) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_encodings, dec_init_vec = self.encode(src_sents_var, src_sent_len)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = vocab.id2tgt[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def sample(self, src_sents: List[List[str]], sample_size=5, max_decoding_time_step=100) -> List[Hypothesis]:
        """
        Given a batched list of source sentences, randomly sample hypotheses from the model distribution p(y|x)

        Args:
            src_sents: a list of batched source sentences
            sample_size: sample size for each source sentence in the batch
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var = vocab.src.to_input_tensor(src_sents, self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(sent) for sent in src_sents])
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec

        batch_size = len(src_sents)
        total_sample_size = sample_size * len(src_sents)

        # (total_sample_size, max_src_len, src_encoding_size)
        src_encodings = src_encodings.repeat(sample_size, 1, 1)
        src_encodings_att_linear = src_encodings_att_linear.repeat(sample_size, 1, 1)

        src_sent_masks = self.get_attention_mask(src_encodings, [len(sent) for _ in range(sample_size) for sent in src_sents])

        h_tm1 = (h_tm1[0].repeat(sample_size, 1), h_tm1[1].repeat(sample_size, 1))

        att_tm1 = torch.zeros(total_sample_size, self.hidden_size, device=self.device)

        eos_id = vocab.tgt['</s>']
        sample_ends = torch.zeros(total_sample_size, dtype=torch.uint8, device=self.device)
        sample_scores = torch.zeros(total_sample_size, device=self.device)

        samples = [torch.tensor([vocab.tgt['<s>']] * total_sample_size, dtype=torch.long, device=self.device)]

        t = 0
        while t < max_decoding_time_step:
            t += 1

            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            # probabilities over target words
            p_t = F.softmax(self.readout(att_t), dim=-1)
            log_p_t = torch.log(p_t)

            # (total_sample_size)
            y_t = torch.multinomial(p_t, num_samples=1)
            log_p_y_t = torch.gather(log_p_t, 1, y_t).squeeze(1)
            y_t = y_t.squeeze(1)

            samples.append(y_t)

            sample_ends |= torch.eq(y_t, eos_id).byte()
            sample_scores = sample_scores + log_p_y_t * (1. - sample_ends.float())

            if torch.all(sample_ends):
                break

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        _completed_samples = [[[] for _1 in range(sample_size)] for _2 in range(batch_size)]
        for t, y_t in enumerate(samples):
            for i, sampled_word_id in enumerate(y_t):
                sampled_word_id = sampled_word_id.cpu().item()
                src_sent_id = i % batch_size
                sample_id = i // batch_size

                if t == 0 or _completed_samples[src_sent_id][sample_id][-1] != eos_id:
                    _completed_samples[src_sent_id][sample_id].append(sampled_word_id)

        completed_samples = [[None for _1 in range(sample_size)] for _2 in range(batch_size)]
        for src_sent_id in range(batch_size):
            for sample_id in range(sample_size):
                offset = sample_id * batch_size + src_sent_id
                hyp = Hypothesis(value=vocab.tgt.indices2words(_completed_samples[src_sent_id][sample_id])[:-1],
                                 score=sample_scores[offset].item())
                completed_samples[src_sent_id][sample_id] = hyp

        return completed_samples