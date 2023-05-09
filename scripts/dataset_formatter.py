""" Extract the lines of the documents that are annotated as rationale and save them in a json file.

Example:
    $ python dataset_formatter.py \
        --data_root './movies/' \
        --format_all_dataset \
        --save_sentences_in_txt

    or

    $ python dataset_formatter.py \
        --data_root './movies/' \
        --format_dataset 'train,dev' \
        --save_sentences_in_txt
"""

import json
import os
from eraserbenchmark.rationale_benchmark.utils import load_documents, load_datasets, _annotation_to_dict
import argparse


def load_dataset_annotation(data_root):
    """
    Load the dataset annotations

    Parameters:
    ----------
    data_root: str
        The path to the dataset folder

    Returns:
    -------
    train: list
        The list of the training annotations

    dev: list
        The list of the development annotations

    test: list
        The list of the testing annotations

    """
    train, dev, test = load_datasets(data_root)
    return train, dev, test


def load_dataset_documents(data_root):
    """
    Load the dataset documents

    Parameters:
    ----------
    data_root: str
        The path to the dataset folder

    Returns:
    -------
    documents: dict
        The dictionary of the documents
    """
    documents = load_documents(data_root)
    return documents


def get_annotation_sentences_ids_single_file(annotation, document):
    """
    Extract the lines of a document that are annotated as rationale

    Parameters:
    ----------
    annotation: Annotation
        The annotation of the document

    document: list
        The list of tokens of each line in the document

    Returns:
    -------
    sentences: list
        The list of the sentences that are annotated as rationale

    evidences: list
        The list of the evidences for the extracted sentences/lines
    """
    sentences = []
    for ev in annotation.all_evidences():
        start_sentence_idx = ev.start_sentence
        end_sentence_idx = ev.end_sentence
        sentence = ""
        for i in range(start_sentence_idx, end_sentence_idx):
            for token in document[i]:
                sentence += token + " "
        sentences.append(sentence)
    return sentences, annotation.all_evidences()


def get_annotation_sentences_ids(annotations, documents):
    """
    Extract the lines of all documents that are annotated as rationale

    Parameters:
    ----------
    annotations: list
        The list of the annotations of the documents

    documents: dict
        The dictionary of the documents

    Returns:
    -------
    all_sentences: list
        The list of the sentences that are annotated as rationale

    all_evidences: list
        The list of the evidences for the extracted sentences/lines
    """
    all_sentences = []
    all_evidences = []

    for annotation in annotations:
        sentences, evidences = get_annotation_sentences_ids_single_file(annotation, documents[annotation.annotation_id])
        all_sentences.append(sentences)
        all_evidences.append(evidences)
    return all_sentences, all_evidences


def save_dataset_annotations_json(annotations, documents, output_file):
    """
    Save the annotations(evidence part) and related sentence as a dataset in a json file

    Parameters:
    ----------
    annotations: list
        The list of the annotations of the documents

    documents: dict
        The dictionary of the documents

    output_file: str
        The path to the output file

    Returns:
    -------
    dataset: list
        The list of the dataset
    """
    all_sentences, all_evidences = get_annotation_sentences_ids(annotations, documents)
    dataset = []

    idx = 0
    for sentences, evidences in zip(all_sentences, all_evidences):
        for sentence, evidence in zip(sentences, evidences):
            if evidence.docid[:3] == 'neg':
                label = 0
            else:
                label = 1

            dataset.append({
                "id": idx,
                "sentence": sentence,
                "label": label,
                "evidence": _annotation_to_dict(evidence)
            })
            idx += 1

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    return dataset


def save_sentences_in_txt(annotations, documents, output_file):
    """
    Save the sentences of a dataset in a txt file

    Parameters:
    ----------
    annotations: list
        The list of the annotations of the documents

    documents: dict
        The dictionary of the documents

    output_file: str
        The path to the output file

   """
    all_sentences, all_evidences = get_annotation_sentences_ids(annotations, documents)

    with open(output_file, 'w') as f:
        for sentences in all_sentences:
            for sentence in sentences:
                f.write(sentence + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./movies/', help='path to the dataset folder')
    parser.add_argument('--format_all_dataset', action='store_true', help='format all dataset (train, dev, test)')
    parser.add_argument('--format_dataset', type=str, default='train', help='format some types of dataset (train, '
                                                                            'dev, test) separate by comma')
    parser.add_argument('--save_sentences_in_txt', action='store_true', help='save sentences in files')
    parser.add_argument('--save_Dir', type=str, default='./datasets/', help='path to save the format dataset')

    args = parser.parse_args()

    train, dev, test = load_dataset_annotation(args.data_root)
    documents = load_dataset_documents(args.data_root)

    if not os.path.exists(args.save_Dir):
        os.makedirs(args.save_Dir)

    if args.format_all_dataset:
        save_dataset_annotations_json(train, documents, args.save_Dir + 'movie_train.json')
        save_dataset_annotations_json(dev, documents, args.save_Dir + 'movie_dev.json')
        save_dataset_annotations_json(test, documents, args.save_Dir + 'movie_test.json')

        if args.save_sentences_in_txt:
            save_sentences_in_txt(train, documents, args.save_Dir + 'movie_train.txt')
            save_sentences_in_txt(dev, documents, args.save_Dir + 'movie_dev.txt')
            save_sentences_in_txt(test, documents, args.save_Dir + 'movie_test.txt')

    else:
        types = args.format_dataset.split(',')
        for t in types:
            if t == 'train':
                save_dataset_annotations_json(train, documents, args.save_Dir + 'movie_train.json')

                if args.save_sentences_in_txt:
                    save_sentences_in_txt(train, documents, args.save_Dir + 'movie_train.txt')
            elif t == 'dev':
                save_dataset_annotations_json(dev, documents, args.save_Dir + 'movie_dev.json')

                if args.save_sentences_in_txt:
                    save_sentences_in_txt(dev, documents, args.save_Dir + 'movie_dev.txt')
            elif t == 'test':
                save_dataset_annotations_json(test, documents, args.save_Dir + 'movie_test.json')

                if args.save_sentences_in_txt:
                    save_sentences_in_txt(test, documents, args.save_Dir + 'movie_test.txt')
            else:
                raise ValueError('Type of dataset not found!')


    # document = documents['negR_000.txt']
    # get_annotation_sentences_ids_single_file(train[0], document)

    # print(documents['negR_000.txt'][13])


if __name__ == '__main__':
    main()