# eraser_movie_dataset

## Usage
### Format the Dataset
Extract the lines of the documents that are annotated as rationale from the docs and annotation files from the **Eraser Movie Dataset**

Examples: <br/>
Format all datasets (train, dev, and test)
```bash
python script/dataset_formatter.py \
  --data_root './movies/' \
  --format_all_dataset \
  --save_sentences_in_txt \
  --save_type 'Datasets'
```

Format only the train and dev datasets
```bash
python script/dataset_formatter.py \ 
  --data_root './movies/' \ 
  --format_dataset 'train,dev' \ 
  --save_sentences_in_txt \
  --save_type 'Datasets'
```
### Finetuning the formatted dataset extracted from the Eraser Movie Dataset by previous step
Examples: <br/>
```bash
python run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name sst2 \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir glue-cased-models-movie
```