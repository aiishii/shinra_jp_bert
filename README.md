# shinra_jp_bert

[森羅プロジェクト](http://shinra-project.info/)の[森羅2020-JP: 日本語構造化タスク](http://shinra-project.info/shinra2020jp/)(属性値抽出タスク)で用いた手法の実装です。本手法では，属性値抽出タスクを機械読解タスクとみなし、[BERT](https://github.com/google-research/bert)を用いて解いています。グループ（地名、組織名など）内の全カテゴリで中間的に学習した後、各カテゴリでもう一度学習することで、同じグループ内の異なるカテゴリとの共通要素を学習して精度を向上させることを目指しました。システムについての説明は[こちら](https://www.anlp.jp/proceedings/annual_meeting/2021/pdf_dir/P9-11.pdf)もご参照ください。


## 使用方法

### データのダウンロード:

データセットは， [森羅2020-JP: データダウンロード](http://shinra-project.info/shinra2020jp/data_download/)からダウンロードし、以下のように配置してください。

事前学習済みのBERTモデルは[NICT BERT 日本語 Pre-trained モデル](https://alaginrc.nict.go.jp/nict-bert/index.html)の[NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip](https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip)を用います。

[shinra jp scorer](https://github.com/k141303/shinra_jp_scorer)を一部利用するため、使用するファイルを本リポジトリに含んでいます。

```
shinra_jp_bert
|____ data                # 森羅2020-JPからダウンロードしたファイルを解凍して配置
|       |____ JP-5                  
|       |   |____ annotation
|       |   |       |____ Company_dist.json
|       |   |         ...
|       |   |____ html
|       |           |____ Company
|       |           |       |____xxxxx.html
|       |             
|       |____ Location              
|       |       ...
|       |             
|       |____ work        # convert_shinra_to_squad.shが前処理後のファイルを出力するディレクトリ
|____ models              # NICT_BERT-base_JapaneseWikipedia_32K_BPEを解凍して配置 
|____ shinra_jp_scorer    # shinra jp scorerのソースコードの一部

```

### 1. 前処理：
地名(Location)グループを対象とする場合
```
bash convert_shinra_to_squad.sh Location
```
指定されたグループの各カテゴリについてSQuAD形式に変換したファイルを生成します。
デフォルトの設定では、```./data/work/train-shinra_jp_bert_html/```ディレクトリに以下のファイル群を出力します。
```
squad_[カテゴリ名]-[mode].json # modeはtrain、dev、testでデフォルトで85%、5%、10%に分割
[カテゴリ名]-test-id.txt
```

### 2.学習：
地名(Location)グループごとに中間学習し、Bayカテゴリで再学習する場合
#run_squad_for_shinra.sh内にサンプルがあります

#### 2.1 パラメータの設定
```
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
work_dir=${data_dir}/work/${mode}-${test_case_str}
html_data_dir=./data
base_model_dir=../../shinra_jp_bert/models/NICT_BERT-base_JapaneseWikipedia_32K_BPE
```
#### 2.2 グループごとの中間学習

学習は```num_train_epochs```に設定したエポック数分実施されます。
```
GROUP=Location
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}
categories="Location_Other,GPE_Other,Province,Country,Continental_Region,Domestic_Region,Geological_Region_Other,Spa,Mountain,Island,River,Lake,Sea,Bay"

python run_squad_for_shinra.py   --do_train \
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
```
#### 2.3 カテゴリごとの再学習
ログを参照して精度のよかったエポックを設定し、カテゴリごとの再学習を実施します。
```
#グループごとの中間学習のログファイル（上記のパラメータの場合）
./output/Location_shinra_jp_bert_html_Location_train_batch16_epoch10_lr2e-05_seq384/train.log
```
```
group_best_epoch=9
best_model_path=./output/${GROUP}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${group_best_epoch}

target=Bay
python run_squad_for_shinra.py   --do_train \
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
```

### 3. 予測と評価
ログを参照してカテゴリごとの再学習で精度のよかったエポックを設定し、カテゴリごとモデルのパスを設定し、予測と評価を実行します。
```
#カテゴリごとの再学習のログファイル（上記のパラメータの場合）
./output/Bay_shinra2020_Location_train_batch32_epoch10_lr2e-05_seq384/train.log
```
#### 3.1 予測
予測を実行します。
```
category_best_epoch=9
python run_squad_for_shinra.py   --do_predict \
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
```
予測結果は```/output/Bay_shinra2020_Location_train_batch32_epoch10_lr2e-05_seq384/epoch-9/shinra_Bay-test-results.json```(上記パラメータの場合)に出力されます。

#### 3.2 予測結果の整形
出力された予測結果を```regulation_bio.py```でルールで整形します。ルールは以下の通りです。
* 解答文字列前後のHTMLタグの削除
* 空文字および漢字以外で1文字の解答を除外
* HTMLタグと記号のみの解答を除外

```
model_dir=./output/${target}_${test_case_str}_${GROUP}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}
python regulation_bio.py   --predicate_json ${pred_dir}/shinra_${target}_test_results.json \
              --category ${target} \
              --html_dir ${group_dir}/html/${target} \
              --dist_file ${group_dir}/annotation/${target}_dist.json
```
#### 3.3 評価
[shinra jp scorer](https://github.com/k141303/shinra_jp_scorer)で評価結果を出力します。

```
scorer_dir=./shinra_jp_scorer
python ${scorer_dir}   --target ${work_dir}/${target}-test-id.txt \
            --html ${group_dir}/html/${target} \
            --score ${pred_dir}/scorer_score_${target} \
            ${group_dir}/annotation/${target}_dist.json \
            ${pred_dir}/shinra_${target}_test_results.reg.json

```
実行結果（batch_size=16）

System result score
(html)
|属性名|精度|再現率|F値|
|-|-|-|-|
|公園|1.000|1.000|1.000|
|別名|0.704|0.826|0.760|
|動物|0.840|0.824|0.832|
|名前の謂れ|0.000|0.000|0.000|
|国|0.500|0.818|0.621|
|国内位置|0.800|0.667|0.727|
|属する海域|0.562|0.711|0.628|
|属する湾|0.643|0.692|0.667|
|島|0.831|0.841|0.836|
|幅|1.000|0.833|0.909|
|平均水深|0.833|1.000|0.909|
|座標・経度|0.571|0.762|0.653|
|座標・緯度|0.571|0.762|0.653|
|形成時期|0.000|0.000|0.000|
|所在地|0.387|0.612|0.474|
|最大水深|0.750|0.857|0.800|
|構成する湾|0.800|0.667|0.727|
|橋・トンネル|0.857|1.000|0.923|
|水深|0.333|1.000|0.500|
|氷結時期|0.000|0.000|0.000|
|河川|0.902|0.919|0.910|
|海岸線の長さ|1.000|1.000|1.000|
|港|0.818|0.857|0.837|
|湾口幅|0.000|0.000|0.000|
|観光地|0.614|0.659|0.635|
|読み|1.000|0.923|0.960|
|長さ|0.714|0.714|0.714|
|面積|0.571|0.800|0.667|
|macro_ave|0.629|0.705|0.655|
|micro_ave|0.668|0.781|0.720|

### 4. submit用ファイル作成
地名(Location)グループのBayカテゴリの学習モデルでsubmit用ファイルを作成する場合

#run_squad_for_shinra_formal.sh内にサンプルがあります

```
submit_dir=./submit #submit用フォルダ
GROUP=Location
test_case_str=${label}_${GROUP}
group_dir=${html_data_dir}/${GROUP}

# カテゴリごとの予測
target=Bay
category_best_epoch=8
python run_squad_for_shinra.py --do_formal \
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

pred_dir=${output_dir}/${target}_${test_case_str}_train_batch${batch_size}_epoch${num_train_epochs}_lr${LR}_seq${max_seq_length}/epoch-${category_best_epoch}

# 予測結果の整形
python regulation_bio.py   --predicate_json ${pred_dir}/shinra_${target}_formal_results.json \
              --category ${target} \
              --html_dir ${group_dir}/html/${target} \
              --dist_file ${group_dir}/annotation/${target}_dist.json

# 評価（正しく出力できているかの確認）
python ${scorer_dir}   --target ${work_dir}/${target}-test-id.txt \
            --html ${group_dir}/html/${target} \
            --score ${pred_dir}/scorer_score_${target}_formal \
            ${group_dir}/annotation/${target}_dist.json \
            ${pred_dir}/shinra_${target}_formal_results.reg.json

# submit用フォルダにリネームしてコピー
cp ${pred_dir}/shinra_${target}_formal_results.reg.json ${submit_dir}/${GROUP}/${target}.json
```

## インストール

shinra_jp_bertは、Linux/OSX と Python 3.5以上、[PyTorch](http://pytorch.org/) 1.9、[transformers](https://github.com/huggingface/transformers) 2.11で動作します。その他の必要なソフトウェアはrequirements.txtに記載されています。

以下のコマンドを実行し、リポジトリをクローン後インストールしてください。

```bash
git clone https://github.com/aiishii/shinra_jp_bert.git
cd shinra_jp_bert; pip install -r requirements.txt;
```
加えて、[NICT BERT 日本語 Pre-trained モデル](https://alaginrc.nict.go.jp/nict-bert/index.html)を利用するため、以下をインストールする必要があります。
- MeCab (0.996) および Juman辞書 (7.0-20130310)
    - Juman辞書は、--with-charset=utf-8 オプションをつけてインストールする必要があります
- Juman++ (v2.0.0-rc2)

<!-- ## License
shinra_jp_bert はのxxxライセンスを継承しています。
 -->



