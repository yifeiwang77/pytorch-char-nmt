
import torch
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Set, Union
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

def to_one_hot(tensor, char_size, device):
    zeros = torch.zeros(list(tensor.shape)+[char_size], device=device, dtype=torch.float)
    return zeros.scatter_(3, tensor.unsqueeze(3), 1)


def data_iter(data, device, one_hot=False):
    batch_l = data['batch_l'][()]
    batch_w = data['batch_w'][()]
    # source = data['source'][()]
    source_char = data['source_char'][()]
    target = data['target'][()]
    target_l_all = data['target_l_all'][()]
    char_size = data['char_size'][()].item()
    idx = 0
    for bi in range(len(batch_l)):
        l = batch_l[bi]
        w= batch_w[bi]
        tw = target_l_all[idx:idx+l].sum().item()
        src_batch = source_char[idx:idx+l]
        tgt_batch = target[idx:idx+l]
        tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device).transpose(1,0)
        src_tensor = torch.tensor(src_batch, dtype=torch.long, device=device)
        if one_hot:
            src_tensor = to_one_hot(src_tensor, char_size, device)
        yield src_tensor.transpose(1,0), tgt_tensor, w, tw
        idx += l

def read_dict(dict_file, reversed=False):
    d = defaultdict()
    f = open(dict_file, 'r', encoding='utf-8')
    for line in f.readlines():
        k,v = line.strip().split()
        if reversed:
            d[int(v)] = k
        else:
            d[k] = int(v)
    return d

def read_doc(file):
    f = open(file, 'r', encoding='utf-8')
    sents = []
    for line in f.readlines():
        sents.append(line.strip().split())
    return sents


def padword(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))

def tok2id(dict, tok):
    return dict[tok] if tok in dict else dict['<unk>']


def char2tensor(sent, char_dict, device, max_word_l=50, one_hot=False):
    bos = char_dict['{']
    eos = char_dict['}']
    pad = char_dict['<blank>']
    char = [[bos] + [tok2id(char_dict, c) for c in list(word)] + [eos]
                                                for word in sent]
    ids = [[padword(word, max_word_l, pad) for word in char]]
    char_input = torch.tensor(ids, dtype=torch.long, device=device)
    if one_hot:
        char_input = to_one_hot(char_input, len(char_dict), device)
    return char_input.transpose(0,1)

def word2tensor(sent, dict, device):
    bos = dict['<s>']
    eos = dict['</s>']
    pad = dict['<blank>']
    words = [[bos] + [tok2id(dict, word)  for word in sent] + [eos]]
    words_input = torch.tensor(words, dtype=torch.long, device=device)
    return words_input.transpose(0,1)

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score
