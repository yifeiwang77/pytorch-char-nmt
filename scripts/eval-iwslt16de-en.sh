 
python evaluate.py \
    -src_file data/de-en/test.de-en.de.tok \
    -targ_file data/de-en/test.de-en.en.tok \
    -model models/iwslt16de-en.bin \
    -outputfile iwslt14predict.txt \
    -char_dict data/iwslt16de-en.char.dict \
    -targ_dict data/iwslt16de-en.targ.dict