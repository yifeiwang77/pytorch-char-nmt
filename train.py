
import os
import sys
import time
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import CharLSTM
from utils import data_iter
from tensorboardX import SummaryWriter

# Arguments parser
parser = argparse.ArgumentParser(description="Deep Char-NMT")
parser.add_argument('-data_file', type=str, default='data/demo-train.hdf5')
parser.add_argument('-val_data_file', type=str, default='data/demo-val.hdf5')
parser.add_argument('-savefile', default='models/model', type=str, help='save trained model to')
parser.add_argument('-train_from', type=str, default=None)

# model
parser.add_argument('-inputs', type=str, default='char', choices=['char', 'word', 'one_hot_char'])
parser.add_argument('-char_vec_size', type=int, default=25)
parser.add_argument('-kernel_width', type=int, default=6)
parser.add_argument('-num_kernels', type=int, default=1000, help='input word embedding dim')
parser.add_argument('-num_highway_layers', type=int, default=2)
parser.add_argument('-embed_size', type=int, default=512)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-dropout', type=float, default=0)

## optimization
parser.add_argument('-epochs', type=int, default=13)
parser.add_argument('-optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizers')
parser.add_argument('-lr', type=float, default=1)
parser.add_argument('-max_grad_norm', type=float, default=5)
parser.add_argument('-lr_decay', type=float, default=.5)
parser.add_argument('-start_decay_at', type=int, default=9)
parser.add_argument('-patience', type=int, default=5, help='training patience')
parser.add_argument('-mgpu', action='store_true', default=False, help='mutiple gpus')

## bookkeeping
parser.add_argument('-seed', type=int, default=3435)
parser.add_argument('-comment', type=str, default='')
parser.add_argument('-valid_niter', default=None, type=int, help='every n iterations to perform validation')
# parser.add_argument('-valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'], help='metric used for validation')
parser.add_argument('-print_every', default=30, type=int, help='every n iterations to log training statistics')
# parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
parser.add_argument('-save_model_after', default=0, type=int, help='save the model only after n validation iterations')
# parser.add_argument('-save_to_file', default="decode/"+decode_file, type=str, help='if provided, save decoding results to file')
# parser.add_argument('-save_nbest', default=False, action='store_true', help='save nbest decoding results')

args = parser.parse_args()
print(args)

# Fix seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load_data
train_data = h5py.File(args.data_file, 'r')
valid_data = h5py.File(args.val_data_file, 'r')

train_batch_l = train_data['batch_l'][()]
valid_batch_l = valid_data['batch_l'][()]
train_size = train_batch_l.sum().item()
valid_size = valid_batch_l.sum().item()
print('[corpus info] train samples:%d, valid samples:%d, train batch num:%d, valid batch num:%d' 
        % (train_batch_l.sum().item(), valid_batch_l.sum().item(), len(train_batch_l), len(valid_batch_l)))
if args.valid_niter is None:
    args.valid_niter = len(train_batch_l)

#build model
target_size = train_data['target_size'][()].item()
if args.inputs in ['char', 'one_hot_char']:
    char_size = train_data['char_size'][()].item()
    model = CharLSTM(args, src_char_size=char_size, target_size=target_size)
elif args.inputs == 'word':
    src_size = train_data['source_size'][()].item()
    model = CharLSTM(args, src_word_size=src_size, target_size=target_size)

if args.mgpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError('Not support')

# init loggers
train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
hist_valid_scores = []
train_time = begin_time = time.time()
print('begin Maximum Likelihood training')

def evaluate_loss(model, data, device, args):
    cum_loss = 0.
    cum_tgt_words = 0.
    cum_examples = 0.
    with torch.no_grad():
        for src_sents_var, tgt_sents_var, src_words, pred_tgt_word_num in data_iter(data, device, inputs=args.inputs):
            batch_size = src_sents_var.size(1)

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            word_loss = model(src_sents_var, tgt_sents_var, src_words)

            cum_loss += word_loss.item()
            cum_tgt_words += pred_tgt_word_num
            cum_examples += batch_size
    loss = cum_loss / cum_examples
    ppl = np.exp(cum_loss/cum_tgt_words)
    return loss, ppl

for epoch in range(args.epochs):
    for src_sents_var, tgt_sents_var, src_words, pred_tgt_word_num in data_iter(train_data, device, inputs=args.inputs):
        train_iter += 1

        batch_size = src_sents_var.size(1)

        optimizer.zero_grad()

        word_loss = model(src_sents_var, tgt_sents_var, src_words)
        loss = word_loss / batch_size

        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        report_loss += word_loss.item()
        report_examples += batch_size
        report_tgt_words += pred_tgt_word_num

        cum_examples += batch_size 
        cum_batches += batch_size
        cum_loss += word_loss.item()
        cum_tgt_words += pred_tgt_word_num

        if train_iter % args.print_every == 0:
            print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                        report_loss / report_examples,
                                                                                        np.exp(report_loss / report_tgt_words),
                                                                                        cum_examples,
                                                                                        report_tgt_words / (time.time() - train_time),
                                                                                        time.time() - begin_time), file=sys.stderr)

            train_time = time.time()
            report_loss = report_tgt_words = report_examples = 0.

        # perform validation
        if train_iter % args.valid_niter == 0:
            print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                        cum_loss / cum_batches,
                                                                                        np.exp(cum_loss / cum_tgt_words),
                                                                                        cum_examples), file=sys.stderr)

            cum_loss = cum_batches = cum_tgt_words = 0.
            valid_num += 1

            print('begin validation ...', file=sys.stderr)
            model.eval()

            # compute dev. ppl and bleu

            dev_loss, dev_ppl = evaluate_loss(model, valid_data, device, args)
            valid_metric = -dev_loss    # metric: the larger the better
            print('validation: iter %d, dev. loss %f, dev. ppl %f, time elapsed %.2f sec' % 
                    (train_iter, dev_loss, dev_ppl, time.time() - begin_time),
                    file=sys.stderr)

            model.train()

            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
            hist_valid_scores.append(valid_metric)

            if valid_num > args.save_model_after:
                model_file = args.savefile + '.iter%d.bin' % train_iter
                print('save model to [%s]' % model_file, file=sys.stderr)
                if not os.path.exists(os.path.dirname(model_file)):
                    os.makedirs(os.path.dirname(model_file))
                if args.mgpu:
                    torch.save(model.module.state_dict(), model_file)
                else:
                    torch.save(model.state_dict(), model_file)

            if (not is_better_than_last) and args.lr_decay:
                lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                print('decay learning rate to %f' % lr, file=sys.stderr)
                optimizer.param_groups[0]['lr'] = lr

            if is_better:
                patience = 0
                best_model_iter = train_iter

                if valid_num > args.save_model_after:
                    print('save currently the best model ..', file=sys.stderr)
                    model_file_abs_path = os.path.abspath(model_file)
                    symlin_file_abs_path = os.path.abspath(args.savefile + '.bin')
                    os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                    # if valid_metric > args.mle_baseline:
                        # print(args.valid_metric, 'is already greater than', args.mle_baseline)
                        # exit(0)
            else:
                patience += 1 
                print('hit patience %d' % patience, file=sys.stderr)
                if patience == args.patience:
                    print('early stop!', file=sys.stderr)
                    print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                    exit(0)



# def evaluate_bleu(model, data, device, args, batch_num=float('inf')):
#     cum_loss = 0.
#     cum_tgt_words = 0.
#     cum_examples = 0.
#     with torch.no_grad():
#         for src_sents_var, tgt_sents_var, src_words, pred_tgt_word_num in data_iter(data, device, args.one_hot):
#             batch_size = src_sents_var.size(1)

#             # (tgt_sent_len, batch_size, tgt_vocab_size)
#             word_loss = model(src_sents_var, tgt_sents_var, src_words)

#             cum_loss += word_loss.item()
#             cum_tgt_words += pred_tgt_word_num
#             cum_examples += batch_size
#     loss = cum_loss / cum_examples
#     ppl = np.exp(cum_loss/cum_tgt_words)
#     return loss, ppl