#!/bin/sh
data_file="data/iwslt16de-en-train.hdf5"
val_data_file="data/iwslt16de-en-val.hdf5"
savefile="models/iwslt16de-en"

python train.py \
    -data_file ${data_file} \
    -val_data_file ${val_data_file} \
    -savefile ${savefile}