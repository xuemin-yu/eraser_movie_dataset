# eraser_movie_dataset

## Usage
### Format the Dataset
Extract the lines of the documents that are annotated as rationale from the docs and annotation files from the [Eraser Movie Dataset]

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
