BPE_DIR=$1
BIN_DIR=$2
python fairseq_cli/preprocess.py \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $BPE_DIR"/train.bpe" \
  --validpref $BPE_DIR"/val.bpe" \
  --destdir $BIN_DIR \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
