#!/bin/sh
TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=16
BART_PATH=/home/lr/kwonjingun/D2/guided_summarization/bart/saved_model/bart.large/model.pt
DATA_BIN=/home/lr/kwonjingun/D2/guided_summarization/dataset/bin_processed
SAVE_DIR=/home/lr/kwonjingun/data_server/len_cnndm

CUDA_VISIBLE_DEVICES=0 python train.py $DATA_BIN \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task guided_translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch guided_bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --fp16 \
    --find-unused-parameters;
