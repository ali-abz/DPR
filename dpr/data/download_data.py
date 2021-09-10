#!/usr/bin/env python3
import pathlib

local = True
drive_dataset_path = pathlib.Path('/content/drive/MyDrive/IR_Datasets')


def download(resource_key: str, local=False):
    # porseman 1 (mozafari)
    if local:
        porseman_path = pathlib.Path('/home/ali/Downloads/Colab Notebooks/IRDataset/')
    else:
        porseman_path = drive_dataset_path / 'Mozafari'
    if resource_key == 'data.porseman-train':
        return [porseman_path / 'IR-train.json']
    if resource_key == 'data.porseman-dev':
        return [porseman_path / 'IR-dev.json']
    if resource_key == 'data.porseman-test':
        return [porseman_path / 'IR-test.csv']
    if resource_key == 'data.porseman_wiki':
        return [porseman_path / 'IR-wiki.tsv']

    # porseman 2
    if local:
        porseman_path2 = pathlib.Path('/home/ali/Desktop/sem4/projects/DPR-proper-small-dataset/outputs/')
    else:
        porseman_path2 = drive_dataset_path / 'Porseman2'
    if resource_key in ['data.porseman2_train_binary', 'data.porseman2_train_graded']:
        return [porseman_path2 / 'train_questions_pos.jsonl']
    if resource_key in ['data.porseman2_dev_binary', 'data.porseman2_dev_graded']:
        return [porseman_path2 / 'dev_questions_pos.jsonl']
    if resource_key == 'data.porseman2-test':
        return [porseman_path2 / 'test.jsonl']
    if resource_key == 'data.porseman2_wiki':
        return [porseman_path2 / 'collection.csv']

    # trivia
    if local:
        pass
    else:
        trivia_path = drive_dataset_path / 'trivia'
    if resource_key == 'data.trivia-train-1':
        return [trivia_path / 'trivia-train-1.jl']
    if resource_key == 'data.trivia-train-40':
        return [trivia_path / 'trivia-train-40.jl']
    if resource_key == 'data.trivia-train-79':
        return [trivia_path / 'trivia-train-79.jl']
    if resource_key == 'data.trivia-dev-1':
        return [trivia_path / 'trivia-dev-1.jl']

    # trec dl
    if local:
        trec_dl_path = pathlib.Path('/home/ali/Desktop/sem4/projects/TREC-DL-graded/dpr_output/')
    else:
        trec_dl_path = drive_dataset_path / 'trec_dl_dataset'
    if resource_key == 'data.msmarco_super_small':
        return [trec_dl_path / 'super_small_collection.tsv']
    if resource_key == 'data.trec_train_graded':
        return [trec_dl_path / 'train.jsonl']
    if resource_key == 'data.trec_dev_graded':
        return [trec_dl_path / 'dev.jsonl']
    if resource_key == 'data.trec_train_binary':
        return [trec_dl_path / 'train.jsonl']
    if resource_key == 'data.trec_dev_binary':
        return [trec_dl_path / 'dev.jsonl']
    if resource_key == 'data.trec_test':
        return [trec_dl_path / 'test.jsonl']

    # binary trec dl
    if local:
        binary_trec_dl_path = pathlib.Path('/home/ali/Desktop/sem4/projects/TREC-DL-binary/dpr_output/')
    else:
        binary_trec_dl_path = drive_dataset_path / 'trec_dl_dataset_binary'
    if resource_key == 'data.binary_msmarco_super_small':
        return [binary_trec_dl_path / 'super_small_collection.tsv']
    if resource_key == 'data.binary_trec_train':
        return [binary_trec_dl_path / 'train.jsonl']
    if resource_key == 'data.binary_trec_dev':
        return [binary_trec_dl_path / 'dev.jsonl']
    if resource_key == 'data.binary_trec_test':
        return [binary_trec_dl_path / 'test.jsonl']
