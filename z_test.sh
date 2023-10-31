SRC=/home/lr/kwonjingun/D2/topic-aware/dataset/preprocess/test.source
MODEL_NAME=checkpoint_best.pt
DATA_BIN=/home/lr/kwonjingun/D2/topic-aware/dataset/baseline_processed_bin

LOC=1
ABS=0
RATIO=0.5
#sub=180
#TYPE=kwonnotopic_baseline_processed_bin_notsecond_sumloss_lrpe_loc_${LOC}_abs_${ABS}_ratio_${RATIO}_2
TYPE=regul_model_${LOC}_abs_${ABS}_ratio_${RATIO}_gpu2_real
MODEL_DIR=/home/lr/kwonjingun/data_server/only_doc/cnndm/$TYPE
RESULT_PATH=outputs/$TYPE.result.plus3

CUDA_VISIBLE_DEVICES=3 python z_test.py $SRC $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN
