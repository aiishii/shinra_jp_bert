#!/bin/bash
batch_size=16
eval_batch_size=16
max_seq_length=384
doc_stride=128
mode=train
test_case_str=shinra_jp_bert_html
data_dir=./data
work_dir=${data_dir}/work/${mode}-${test_case_str}

html_data_dir=./data
LR=2e-05
prefix=simple


scorer_dir=../shinra_jp_scorer

GROUP=JP5

# test_case_str=${test_case_str}_${GROUP}
group_dir=${html_data_dir}/JP-5

best_model_path=./models/NICT_BERT-base_JapaneseWikipedia_32K_BPE

categories=(Compound Airport City Person Company)
best_epochs=(8 7 8 9 6)
# categories=(Airport)
# best_epochs=(9)
i=0
for target in ${categories[@]}; do
    echo ${target} ${best_epochs[i]}
    target=${categories[i]}
    BEST_EPOCH=${best_epochs[i]}
    # python bert_squad.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str}_${GROUP} --best_model_dir /epoch-${BEST_EPOCH} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
    model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
    python regulation_bio.py --predicate_json ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --prefix ${prefix} --dist_file ${group_dir}/annotation/${target}_dist.json
    python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${model_dir}/epoch-${BEST_EPOCH}/scorer_score_${target}${prefix} ${group_dir}/annotation/${target}_dist.json  ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.reg${prefix}.json
    let i++
done
exit 0

GROUP=Location
group_dir=${html_data_dir}/${GROUP}
categories=(Location_Other GPE_Other Province Country Continental_Region Domestic_Region Geological_Region_Other Spa Mountain Island River Lake Sea Bay)
categories=(Location_Other)
best_epochs=(9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
i=0
for target in ${categories[@]}; do
    echo ${target} ${best_epochs[i]}
    target=${categories[i]}
    BEST_EPOCH=${best_epochs[i]}
    python bert_squad.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --best_model_dir /epoch-${BEST_EPOCH} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
    model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
    python regulation_bio.py --predicate_json ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --prefix ${prefix} --dist_file ${group_dir}/annotation/${target}_dist.json
    python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${model_dir}/epoch-${BEST_EPOCH}/scorer_score_${target}${prefix} ${group_dir}/annotation/${target}_dist.json  ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.reg${prefix}.json
    let i++
done

GROUP=Organization
group_dir=${html_data_dir}/${GROUP}
categories=(Family Military Organization_Other Show_Organization Sports_Team Company_Group Government Political_Organization_Other Sports_Federation Ethnic_Group_Other International_Organization Nonprofit_Organization Political_Party Sports_League)
best_epochs=(9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
i=0
for target in ${categories[@]}; do
    echo ${target} ${best_epochs[i]}
    target=${categories[i]}
    BEST_EPOCH=${best_epochs[i]}
    python bert_squad.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --best_model_dir /epoch-${BEST_EPOCH} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
    model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
    python regulation_bio.py --predicate_json ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --prefix ${prefix} --dist_file ${group_dir}/annotation/${target}_dist.json
    python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${model_dir}/epoch-${BEST_EPOCH}/scorer_score_${target}${prefix} ${group_dir}/annotation/${target}_dist.json  ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.reg${prefix}.json
    let i++
done

GROUP=Event
group_dir=${html_data_dir}/${GROUP}
categories=(Event_Other Occasion_Other Election Religious_Festival Competition Conference Exhibition Incident_Other War Traffic_Accident Earthquake Flood_Damage)
best_epochs=(9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
i=0
for target in ${categories[@]}; do
    echo ${target} ${best_epochs[i]}
    target=${categories[i]}
    BEST_EPOCH=${best_epochs[i]}
    python bert_squad.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --best_model_dir /epoch-${BEST_EPOCH} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
    model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
    python regulation_bio.py --predicate_json ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --prefix ${prefix} --dist_file ${group_dir}/annotation/${target}_dist.json
    python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${model_dir}/epoch-${BEST_EPOCH}/scorer_score_${target}${prefix} ${group_dir}/annotation/${target}_dist.json  ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.reg${prefix}.json
    let i++
done

GROUP=Facility
group_dir=${html_data_dir}/${GROUP}
categories=(Facility_Other Dam Archaeological_Place_Other Cemetery FOE_Other Military_Base Castle Palace Public_Institution Accommodation Medical_Institution School Research_Institute Power_Plant Park Shopping_Complex Sports_Facility Museum Zoo Amusement_Park Theater Worship_Place Car_Stop Station Port Road_Facility Railway_Facility Line_Other Railroad Road Canal Tunnel Bridge)
best_epochs=(9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
i=0
for target in ${categories[@]}; do
    echo ${target} ${best_epochs[i]}
    target=${categories[i]}
    BEST_EPOCH=${best_epochs[i]}
    python bert_squad.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --best_model_dir /epoch-${BEST_EPOCH} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
    model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch10_lr${LR}_seq${max_seq_length}${aug_prefix}
    python regulation_bio.py --predicate_json ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --prefix ${prefix} --dist_file ${group_dir}/annotation/${target}_dist.json
    python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${model_dir}/epoch-${BEST_EPOCH}/scorer_score_${target}${prefix} ${group_dir}/annotation/${target}_dist.json  ${model_dir}/epoch-${BEST_EPOCH}/shinra_${target}_formal_results.reg${prefix}.json
    let i++
done
