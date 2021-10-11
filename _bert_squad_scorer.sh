#!/bin/bash
cat <<__EOT__
TARGET=$1
LR=$2
EPOCH=$3
data_dir=$4
prefix=$5
test_case_str=$6
aug_prefix=$7
# sample_size=$6
__EOT__

TARGET=$1
LR=$2
EPOCH=$3
data_dir=$4
prefix=$5
test_case_str=$6
aug_prefix=$7

batch_size=16
eval_batch_size=16
max_seq_length=384
doc_stride=128

# model_dir=./models/shinra_${TARGET}_${test_case_str}_train_batch${batch_size}_epoch10.0_lr${LR}_seq${max_seq_length}${aug_prefix}
model_dir=./output/${TARGET}_${test_case_str}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
scorer_dir=../shinra_jp_scorer
work_dir=./data/work/${test_case_str}


echo "EPOCH="${EPOCH}

# python regulation_bio.py --predicate_json ${model_dir}/epoch-${EPOCH}/shinra_${TARGET}_test_results.json --category ${TARGET} --html_dir ${data_dir}/html/${TARGET} --prefix ${prefix} --dist_file ${data_dir}/annotation/${TARGET}_dist.json
# cd ..
python ${scorer_dir} --target ${work_dir}/${TARGET}-test-id.txt --html ${data_dir}/html/${TARGET} --score ${model_dir}/epoch-${EPOCH}/scorer_score_${TARGET}${prefix} ${data_dir}/annotation/${TARGET}_dist.json  ${model_dir}/epoch-${EPOCH}/shinra_${TARGET}_test_results.reg${prefix}.json
