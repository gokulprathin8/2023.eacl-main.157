#!/bin/sh
TOTAL_NUM_UPDATES=30000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=16
BART_PATH=/home/lr/kwonjingun/D2/guided_summarization/bart/saved_model/bart.large/model.pt
#DATA_BIN=dataset/cnndm/processed_bin
DATA_BIN=/home/lr/kwonjingun/D2/topic-aware/dataset/baseline_processed_bin

LOC=1
ABS=0

RATIO=0.5

TYPE=regul_model_${LOC}_abs_${ABS}_ratio_${RATIO}_gpu2_real
SAVE_DIR=/home/lr/kwonjingun/data_server/only_doc/cnndm/$TYPE

CUDA_VISIBLE_DEVICES=0,1 python train.py $DATA_BIN \
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
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --find-unused-parameters \
    --loc-position $LOC \
    --add-absolute $ABS \
    --patience 3 \
    --mse-ratio $RATIO \
    --log-interval 500
