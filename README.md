
## A Character-level Neural Machine Translation Model in PyTorch
---
A PyTorch Implementation of the paper Character-based Neural Machine Translation by Costa-jussa and Fonollosa.

*Specifically, we implement the char model of Yoon Kim's https://github.com/harvardnlp/seq2seq-attn with PyTorch. We try to keep similar usages, hyperparameters and preprocessing, not they are the same.*

TODO:
1. different convs (But Kim's code also uses 1000 conv kernels with the same kernel width)
2. evaluation of BLEU score

### Usage
#### Preprocessing
```
python preprocess.py --srcfile data/src-train.txt --targetfile data/targ-train.txt --srcvalfile data/src-val.txt --targetvalfile data/targ-val.txt --outputfile data/demo --chars 1
```

### Training

```
python train.py -data_file data/demo-train.hdf5 -val_data_file data/demo-val.hdf5 -savefile demo-model
```


### Evaluating

```
python evaluate.py -model demo-model.bin -src_file data/src-val.txt -output_file pred.txt
-char_dict data/demo.char.dict -targ_dict data/demo.targ.dict
```
(Optional)
```
perl multi-bleu.perl pred.txt < data/targ-test.txt
```

Features:
1. Character encoder and normal word decoder, input feeding
2. multi-gpu training (but no significant speedup)

### Acknowledgement

https://github.com/pcyin/pytorch_basic_nmt
