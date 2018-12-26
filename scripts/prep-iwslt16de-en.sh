#!/bin/sh

srcfile="data/de-en/train.de-en.de.tok"
targetfile="data/de-en/train.de-en.en.tok"
srcvalfile="data/de-en/dev.de-en.de.tok"
targetvalfile="data/de-en/dev.de-en.en.tok"
outputfile="data/iwslt16de-en"

python preprocess.py \
    --srcfile ${srcfile} \
    --targetfile ${targetfile} \
    --srcvalfile ${srcvalfile}  \
    --targetvalfile ${targetvalfile} \
    --outputfile ${outputfile} \
    --chars 1