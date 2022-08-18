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

html_data_dir=./data
base_model_dir=./models/NICT_BERT-base_JapaneseWikipedia_32K_BPE
scorer_dir=./shinra_jp_scorer
output_dir=./output

#####################################
#Location グループ　Bayカテゴリ実行サンプル
GROUP=Location
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}
categories="Location_Other,GPE_Other,Province,Country,Continental_Region,Domestic_Region,Geological_Region_Other,Spa,Mountain,Island,River,Lake,Sea,Bay"
# グループごとの中間学習
python run_squad_for_shinra.py 	--do_train \
						--group ${GROUP} \
						--categories ${categories} \
						--not_with_negative \
						--per_gpu_train_batch_size ${batch_size} \
						--per_gpu_eval_batch_size ${eval_batch_size} \
						--learning_rate ${LR} \
						--max_seq_length ${max_seq_length} \
						--doc_stride ${doc_stride} \
						--test_case_str ${test_case_str} \
						--data_dir ${work_dir} \
						--num_train_epochs ${num_train_epochs} \
						--base_model_name_or_path ${base_model_dir} \
						--model_name_or_path ${base_model_dir} \
						--output_dir ${output_dir} \
						--evaluate_during_training 

group_best_epoch=9
best_model_path=${output_dir}/${GROUP}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${group_best_epoch}

# カテゴリごとの中間学習
target=Bay
python run_squad_for_shinra.py 	--do_train \
						--category ${target} \
						--per_gpu_train_batch_size ${batch_size} \
						--per_gpu_eval_batch_size ${eval_batch_size} \
						--learning_rate ${LR} \
						--max_seq_length ${max_seq_length} \
						--doc_stride ${doc_stride} \
						--test_case_str ${test_case_str} \
						--data_dir ${work_dir} \
						--model_name_or_path ${best_model_path} \
						--base_model_name_or_path ${base_model_dir} \
						--num_train_epochs ${num_train_epochs} \
						--output_dir ${output_dir} \
						--evaluate_during_training

# 予測
category_best_epoch=8
python run_squad_for_shinra.py 	--do_predict \
						--category ${target} \
						--per_gpu_train_batch_size ${batch_size} \
						--per_gpu_eval_batch_size ${eval_batch_size} \
						--learning_rate ${LR} \
						--max_seq_length ${max_seq_length} \
						--doc_stride ${doc_stride} \
						--test_case_str ${test_case_str}  \
						--data_dir ${work_dir} \
						--num_train_epochs ${num_train_epochs} \
						--base_model_name_or_path ${base_model_dir} \
						--output_dir ${output_dir} \
						--best_model_dir /epoch-${category_best_epoch} 

# 予測結果の整形
pred_dir=${output_dir}/${target}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${category_best_epoch}
python regulation_bio.py 	--predicate_json ${pred_dir}/shinra_${target}_test_results.json \
							--category ${target} \
							--html_dir ${group_dir}/html/${target} \
							--dist_file ${group_dir}/annotation/${target}_dist.json
# 予測結果の評価
python ${scorer_dir} 	--target ${work_dir}/${target}-test-id.txt \
						--html ${group_dir}/html/${target} \
						--score ${pred_dir}/scorer_score_${target} \
						${group_dir}/annotation/${target}_dist.json \
						${pred_dir}/shinra_${target}_test_results.reg.json

exit 0
#####################################
#JP-5 全カテゴリ実行サンプル　※bashで実行する
GROUP=JP-5
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}
categories="Compound,Person,Company,City,Airport"
# グループごとの中間学習
python run_squad_for_shinra.py --group ${GROUP} --categories ${categories} --not_with_negative --do_train --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --data_dir ${work_dir} --num_train_epochs ${num_train_epochs} --base_model_name_or_path ${base_model_dir} --model_name_or_path ${base_model_dir} --evaluate_during_training
group_best_epoch=9
best_model_path=${output_dir}/${GROUP}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${group_best_epoch}

# カテゴリごとの中間学習
targets=(Compound Airport City Person Company)
for target in ${targets[@]}; do
	echo $target
	python run_squad_for_shinra.py --do_train --category ${target} --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str} --data_dir ${work_dir} --model_name_or_path ${best_model_path} --base_model_name_or_path ${base_model_dir} --num_train_epochs ${num_train_epochs} --evaluate_during_training
done

category_best_epochs=(8 7 8 9 6)
i=0
for target in ${targets[@]}; do
	category_best_epoch=${category_best_epochs[i]}
	# 予測
	python run_squad_for_shinra.py --category ${target} --do_predict --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${eval_batch_size} --learning_rate ${LR} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --test_case_str ${test_case_str}  --data_dir ${work_dir} --num_train_epochs ${num_train_epochs} --base_model_name_or_path ${base_model_dir} --best_model_dir /epoch-${category_best_epoch} 
	# 予測結果の整形
	pred_dir=${output_dir}/${target}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${category_best_epoch}
	python regulation_bio.py --predicate_json ${pred_dir}/shinra_${target}_test_results.json --category ${target} --html_dir ${group_dir}/html/${target} --dist_file ${group_dir}/annotation/${target}_dist.json
	# 予測結果の評価
	python ${scorer_dir} --target ${work_dir}/${target}-test-id.txt --html ${group_dir}/html/${target} --score ${pred_dir}/scorer_score_${target} ${group_dir}/annotation/${target}_dist.json  ${pred_dir}/shinra_${target}_test_results.reg.json
	let i++
done

exit 0

#####################################
#グループごとのカテゴリリスト
GROUP=JP-5
categories="Compound,Person,Company,City,Airport"
targets=(Compound Airport City Person Company)

GROUP=Location
categories="Location_Other,GPE_Other,Province,Country,Continental_Region,Domestic_Region,Geological_Region_Other,Spa,Mountain,Island,River,Lake,Sea,Bay"
targets=(Location_Other GPE_Other Province Country Continental_Region Domestic_Region Geological_Region_Other Spa Mountain Island River Lake Sea Bay)

GROUP=Organization
categories="Family,Military,Organization_Other,Show_Organization,Sports_Team,Company_Group,Government,Political_Organization_Other,Sports_Federation,Ethnic_Group_Other,International_Organization,Nonprofit_Organization,Political_Party,Sports_League"
targets=(Family Military Organization_Other Show_Organization Sports_Team Company_Group Government Political_Organization_Other Sports_Federation Ethnic_Group_Other International_Organization Nonprofit_Organization Political_Party Sports_League)

GROUP=Event
categories="Competition,Conference,Earthquake,Election,Event_Other,Exhibition,Flood_Damage,Incident_Other,Occasion_Other,Religious_Festival,Traffic_Accident,War"
targets=(Competition Conference Earthquake Election Event_Other Exhibition Flood_Damage Incident_Other Occasion_Other Religious_Festival Traffic_Accident War)

GROUP=Facility
categories="Facility_Other,Dam,Archaeological_Place_Other,Cemetery,FOE_Other,Military_Base,Castle,Palace,Public_Institution,Accommodation,Medical_Institution,School,Research_Institute,Power_Plant,Park,Shopping_Complex,Sports_Facility,Museum,Zoo,Amusement_Park,Theater,Worship_Place,Car_Stop,Station,Port,Road_Facility,Railway_Facility,Line_Other,Railroad,Road,Canal,Tunnel,Bridge"
targets=(Facility_Other Dam Archaeological_Place_Other Cemetery FOE_Other Military_Base Castle Palace Public_Institution Accommodation Medical_Institution School Research_Institute Power_Plant Park Shopping_Complex Sports_Facility Museum Zoo Amusement_Park Theater Worship_Place Car_Stop Station Port Road_Facility Railway_Facility Line_Other Railroad Road Canal Tunnel Bridge)
