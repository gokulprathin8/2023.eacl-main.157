SRC=/home/lr/kwonjingun/D2/topic-aware/dataset/preprocess/test.source
MODEL_NAME=checkpoint_best.pt
#DATA_BIN=/home/lr/kwonjingun/D2/topic-aware/dataset/cut_topic_processed_bin
#DATA_BIN=/home/lr/kwonjingun/D2/topic-aware/dataset/not_cut_topic_processed_bin
DATA_BIN=/home/lr/kwonjingun/D2/topic-aware/dataset/full_topic_processed_bin

LOC=-1
ABS=-1
RATIO=0
#TYPE=16topic_cut_later_notsecond_sumloss_lrpe_loc_${LOC}_abs_${ABS}_ratio_${RATIO}
#TYPE=16topic_notcut_later_notsecond_sumloss_lrpe_loc_${LOC}_abs_${ABS}_ratio_${RATIO}
TYPE=16topic_full_later_notsecond_sumloss_lrpe_loc_${LOC}_abs_${ABS}_ratio_${RATIO}
MODEL_DIR=/home/lr/kwonjingun/data_server/topic-aware/$TYPE
RESULT_PATH=outputs/$TYPE.result.intminusdummy

CUDA_VISIBLE_DEVICES=1 python z_base_test.py $SRC $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN
