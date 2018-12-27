import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
import argparse, os, sys, time, math

# from util import read_corpus, data_iter, batch_slice
from utils import *
from model import CharLSTM

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('-src_file', type=str, default='data/src-test.txt')
parser.add_argument('-targ_file', type=str, default='data/targ-test.txt')
parser.add_argument('-sample_size', type=int,  default=None)
parser.add_argument('-model', type=str, default='models/demo.bin')
parser.add_argument('-output_file', type=str, default='data/pred.txt')
parser.add_argument('-src_dict', type=str, default='data/demo.src.dict')
parser.add_argument('-targ_dict', type=str, default='data/demo.targ.dict')
parser.add_argument('-char_dict', type=str, default='data/demo.char.dict')
parser.add_argument('--maxwordlength', help="For the character models, words are "
                                        "(if longer than maxwordlength) or zero-padded "
                                        "(if shorter) to maxwordlength", type=int, default=35)
# model
parser.add_argument('-inputs', type=str, default='char', choices=['char', 'word', 'one_hot_char'])
parser.add_argument('-char_vec_size', type=int, default=25)
parser.add_argument('-kernel_width', type=int, default=6)
parser.add_argument('-num_kernels', type=int, default=1000)
parser.add_argument('-num_highway_layers', type=int, default=2)
parser.add_argument('-embed_size', type=int, default=512)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-dropout', type=float, default=0)

# beam search option
parser.add_argument('-beam', type=int, default=5)
parser.add_argument('-max_sent_l', type=int, default=250)
## bookkeeping
parser.add_argument('-seed', type=int, default=3435)
parser.add_argument('-verbose', action='store_true', default=False)
parser.add_argument('-save_nbest', action='store_true', default=False)

args = parser.parse_args()
print(args)

# Fix seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare data
if args.inputs in ['char', 'one_hot_char']:
    src_char2id = read_dict(args.char_dict)
elif args.inputs == 'word':
    src_word2id = read_dict(args.char_dict)
else:
    raise ValueError('input type not supported')

targ_word2id = read_dict(args.targ_dict)
targ_id2word = read_dict(args.targ_dict, reversed=True)

src_sents = read_doc(args.src_file)
if args.targ_file:
    targ_sents = read_doc(args.targ_file)
class TargetVocab():
    tgt = targ_word2id
    id2tgt = targ_id2word
vocab = TargetVocab()

assert len(src_sents) == len(targ_sents)
print('test data size: ', len(src_sents))

# build and model

target_size = len(targ_word2id)
if args.inputs in ['char', 'one_hot_char']:
    src_char_size = len(src_char2id)
    model = CharLSTM(args, src_char_size=src_char_size, target_size=target_size)
elif args.inputs == 'word':
    src_word_size = len(src_word2id)
    model = CharLSTM(args, src_word_size=src_word_size, target_size=target_size)

print('load model from [%s]' % args.model, file=sys.stderr)
params = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(params)
model.to(device)
model.eval()

# translate
hypotheses = []
if args.sample_size is None:
    args.sample_size = len(src_sents)
for i in range(len(src_sents)):
    # prepare input
    src_sent = src_sents[i]
    tgt_sent = targ_sents[i]
    if args.inputs in ['char', 'one_hot_char']:
        src_sents_var = char2tensor(src_sent, src_char2id, device, max_word_l=args.maxwordlength, 
                                    one_hot = (args.inputs=='one_hot_char'))
    elif args.inputs == 'word':
        src_sents_var = word2tensor(src_sent, src_word2id, device)
    tgt_sents_var = word2tensor(tgt_sent, vocab.tgt, device)
    # beam search
    hyps = model.beam_search(src_sents_var, len(src_sent), vocab, args.beam, 
                max_decoding_time_step=args.max_sent_l, maxwordlength=args.maxwordlength)
    hypotheses.append(hyps)
    # record
    if args.verbose:
        print('*' * 50)
        print('Source: ', ' '.join(src_sent))
        if args.targ_file:
            print('Target: ', ' '.join(tgt_sent))
        print('Top Hypothesis: ', ' '.join(hyps[0].value))
    if i % 100 == 0:
        print(i, 'samples translated')
    if i+1 == args.sample_size:
        break
print('%d samples translated' % args.sample_size)

if args.targ_file:
    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(targ_sents[:args.sample_size], top_hypotheses)
    print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

with open(args.output_file, 'w') as f:
    for hyps in hypotheses:
        top_hyp = hyps[0]
        hyp_sent = ' '.join(top_hyp.value)
        f.write(hyp_sent + '\n')