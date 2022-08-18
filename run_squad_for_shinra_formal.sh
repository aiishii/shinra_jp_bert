#!/bin/bash
batch_size=16
eval_batch_size=16
max_seq_length=384
doc_stride=128
LR=2e-05
num_train_epochs=10

mode=train
label=shinra2020
data_dir=./data
work_dir=${data_dir}/work/${mode}-${label}

base_model_dir=./models/NICT_BERT-base_JapaneseWikipedia_32K_BPE
html_data_dir=./data
scorer_dir=./shinra_jp_scorer
output_dir=./output
submit_dir=./submit
# mkdir ${submit_dir}
#####################################
html_data_dir=../../shinra_jp_bert/data
data_dir=../../shinra_jp_bert/data
work_dir=${data_dir}/work/${mode}-${label}
base_model_dir=../../shinra_jp_bert/models/NICT_BERT-base_JapaneseWikipedia_32K_BPE
# scorer_dir=./shinra_jp_scorer
# output_dir=./output2
output_dir=./output

#Location グループ　Bayカテゴリ実行サンプル
GROUP=Location
# mkdir ${submit_dir}/${GROUP}
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}

# Bayカテゴリの予測
target=Bay
category_best_epoch=8
# 予測
python run_squad_for_shinra.py 	--do_formal \
						--category ${target} \
						--per_gpu_train_batch_size ${batch_size} \
						--per_gpu_eval_batch_size ${eval_batch_size} \
						--learning_rate ${LR} \
						--max_seq_length ${max_seq_length} \
						--doc_stride ${doc_stride} \
						--test_case_str ${test_case_str} \
						--output_dir ${output_dir} \
						--base_model_name_or_path ${base_model_dir} \
						--best_model_dir /epoch-${category_best_epoch} \
						--data_dir ${data_dir} \
						--dist_file ${group_dir}/annotation/${target}_dist.json \
						--html_dir ${group_dir}/html/${target}
# exit 0
pred_dir=${output_dir}/${target}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${category_best_epoch}
# 予測結果の整形
python regulation_bio.py 	--predicate_json ${pred_dir}/shinra_${target}_formal_results.json \
							--category ${target} \
							--html_dir ${group_dir}/html/${target} \
							--dist_file ${group_dir}/annotation/${target}_dist.json
# 予測結果の評価（正しく出力できているかの確認）
python ${scorer_dir} 	--target ${work_dir}/${target}-test-id.txt \
						--html ${group_dir}/html/${target} \
						--score ${pred_dir}/scorer_score_${target}_formal \
						${group_dir}/annotation/${target}_dist.json \
						${pred_dir}/shinra_${target}_formal_results.reg.json
# submit用フォルダにリネームしてコピー
cp ${pred_dir}/shinra_${target}_formal_results.reg.json ${submit_dir}/${GROUP}/${target}.json
exit 0
#####################################
#JP-5 全カテゴリ実行サンプル　※bashで実行する
GROUP=JP-5
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}

targets=(Compound Airport City Person Company)
category_best_epochs=(8 7 8 9 6)
i=0
for target in ${targets[@]}; do
	category_best_epoch=${category_best_epochs[i]}
	# 予測
	python run_squad_for_shinra.py --do_formal --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --output_dir ${output_dir} --base_model_name_or_path ${base_model_dir} --best_model_dir /epoch-${category_best_epoch} --data_dir ${data_dir} --dist_file ${group_dir}/annotation/${target}_dist.json --html_dir ${group_dir}/html/${target}
	pred_dir=${output_dir}/${target}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${category_best_epoch}
	# 予測結果の整形
	python regulation_bio.py --predicate_json ${pred_dir}/shinra_${target}_formal_results.json --category ${target} --html_dir ${group_dir}/html/${target} --dist_file ${group_dir}/annotation/${target}_dist.json
	# 予測結果の評価（正しく出力できているかの確認）
	python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${pred_dir}/scorer_score_${target}_formal ${group_dir}/annotation/${target}_dist.json ${pred_dir}/shinra_${target}_formal_results.reg.json
	# submit用フォルダにリネームしてコピー
	cp ${pred_dir}/shinra_${target}_formal_results.reg.json ${submit_dir}/${GROUP}/${target}.json
	let i++
done

exit 0

#####################################
#グループごとのカテゴリリスト
GROUP=JP-5
targets=(Compound Airport City Person Company)

GROUP=Location
targets=(Location_Other GPE_Other Province Country Continental_Region Domestic_Region Geological_Region_Other Spa Mountain Island River Lake Sea Bay)

GROUP=Organization
targets=(Family Military Organization_Other Show_Organization Sports_Team Company_Group Government Political_Organization_Other Sports_Federation Ethnic_Group_Other International_Organization Nonprofit_Organization Political_Party Sports_League)

GROUP=Event
targets=(Competition Conference Earthquake Election Event_Other Exhibition Flood_Damage Incident_Other Occasion_Other Religious_Festival Traffic_Accident War)

GROUP=Facility
targets=(Facility_Other Dam Archaeological_Place_Other Cemetery FOE_Other Military_Base Castle Palace Public_Institution Accommodation Medical_Institution School Research_Institute Power_Plant Park Shopping_Complex Sports_Facility Museum Zoo Amusement_Park Theater Worship_Place Car_Stop Station Port Road_Facility Railway_Facility Line_Other Railroad Road Canal Tunnel Bridge)
