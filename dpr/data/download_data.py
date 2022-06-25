#!/usr/bin/env python3
import pathlib

local = False
drive_dataset_path = pathlib.Path('/content/drive/MyDrive/IR_Datasets')


def download(resource_key: str):
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
        # porseman_path2 = pathlib.Path('/home/ali/Desktop/sem4/projects/DPR-proper-small-dataset/outputs/')
        porseman_path2 = pathlib.Path('C:\\Users\\ali\\Deskop\\ds\\porseman2')
    else:
        porseman_path2 = drive_dataset_path / 'Porseman2'
    if resource_key in ['data.porseman2_train_binary', 'data.porseman2_train_graded']:
        return [porseman_path2 / 'train_questions_pos.jsonl']
    if resource_key in ['data.porseman2_dev_binary', 'data.porseman2_dev_graded']:
        return [porseman_path2 / 'dev_questions_pos.jsonl']
    if resource_key == 'data.porseman2_test':
        return [porseman_path2 / 'test.jsonl']
    if resource_key == 'data.porseman2_wiki':
        return [porseman_path2 / 'collection.csv']
    if resource_key == 'data.porseman2_ss_graded':
        return [porseman_path2 / 'ss_questions_pos.jsonl']
    if resource_key == 'data.porseman2_ss_binary':
        return [porseman_path2 / 'ss_questions_pos.jsonl']
    if resource_key == 'data.porseman2_dumped':
        return [porseman_path2 / 'dump.jsonl']

    # porseman 2 tree generated
    porseman_2_tree_path = drive_dataset_path / 'Porseman2_Tree'
    if resource_key == 'data.porseman2_train_graded_tree':
        return [porseman_2_tree_path / 'train.jsonl']
    if resource_key == 'data.porseman2_dev_graded_tree':
        return [porseman_2_tree_path / 'dev.jsonl']
    if resource_key == 'data.porseman2_train_graded_tree_flat':
        return [porseman_2_tree_path / 'train_flat_reduced.jsonl']
    if resource_key == 'data.porseman2_dev_graded_tree_flat':
        return [porseman_2_tree_path / 'dev_flat_reduced.jsonl']

    # porseman 3
    if local:
        # porseman_path2 = pathlib.Path('/home/ali/Desktop/sem4/projects/DPR-proper-small-dataset/outputs/')
        porseman_path3 = pathlib.Path('C:\\Users\\ali\\Deskop\\ds\\porseman2')
    else:
        porseman_path3 = drive_dataset_path / 'Porseman3'
    if resource_key in ['data.porseman3_train_binary', 'data.porseman3_train_graded']:
        return [porseman_path3 / 'train_questions_pos.jsonl']
    if resource_key in ['data.porseman3_dev_binary', 'data.porseman3_dev_graded']:
        return [porseman_path3 / 'dev_questions_pos.jsonl']
    if resource_key == 'data.porseman3_test':
        return [porseman_path3 / 'test.jsonl']
    if resource_key == 'data.porseman3_wiki':
        return [porseman_path3 / 'collection.csv']

    # porseman 26k collections
    if resource_key == 'data.porseman_26k_summarized':
        return [drive_dataset_path / '26k-summarized-collection.tsv']
    if resource_key == 'data.porseman_26k':
        return [drive_dataset_path / '26k-collection.tsv']

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
        # trec_dl_path = pathlib.Path('/home/ali/Desktop/sem4/projects/TREC-DL-graded/dpr_output/')
        trec_dl_path = pathlib.Path('C:\\Users\\ali\\Desktop\\ds\\trec-dl')
    else:
        trec_dl_path = drive_dataset_path / 'TREC-DL-graded-DPR-format'
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

    # trec dl translated
    if local:
        trec_dl_translated_path = pathlib.Path('/home/ali/Desktop/sem4/projects/TREC-DL-graded-DPR-format-TRANSLATED/TREC-DL-graded-DPR-format-TRANSLATED/')
        # trec_dl_path = pathlib.Path('C:\\Users\\ali\\Desktop\\ds\\trec-dl')
    else:
        trec_dl_translated_path = drive_dataset_path / 'TREC-DL-graded-DPR-format-TRANSLATED'
    if resource_key == 'data.msmarco_super_small_translated':
        return [trec_dl_translated_path / 'super_small_collection.tsv']
    if resource_key == 'data.trec_train_graded_translated':
        return [trec_dl_translated_path / 'train.jsonl']
    if resource_key == 'data.trec_dev_graded_translated':
        return [trec_dl_translated_path / 'dev.jsonl']
    if resource_key == 'data.trec_train_binary_translated':
        return [trec_dl_translated_path / 'train.jsonl']
    if resource_key == 'data.trec_dev_binary_translated':
        return [trec_dl_translated_path / 'dev.jsonl']
    if resource_key == 'data.trec_test_translated':
        return [trec_dl_translated_path / 'test.jsonl']

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

    # AA general
    if local:
        aa_path = pathlib.Path('/home/ali/Desktop/sem4/projects/datasets-for-AA/outputs/')
    else:
        aa_path = drive_dataset_path / 'AA'

    # c50
    if resource_key == 'data.c50_train':
        return [aa_path / 'C50-train.json']
    if resource_key == 'data.c50_dev':
        return [aa_path / 'C50-dev.json']
    if resource_key == 'data.c50_test':
        return [aa_path / 'C50-test.jsonl']
    if resource_key == 'data.c50_wiki':
        return [aa_path / 'C50-wiki.tsv']

    # gutenberg
    if resource_key == 'data.gutenberg_train':
        return [aa_path / 'Gutenberg-train.json']
    if resource_key == 'data.gutenberg_dev':
        return [aa_path / 'Gutenberg-dev.json']
    if resource_key == 'data.gutenberg_test':
        return [aa_path / 'Gutenberg-test.jsonl']
    if resource_key == 'data.gutenberg_wiki':
        return [aa_path / 'Gutenberg-wiki.tsv']

    # imdb
    if resource_key == 'data.imdb_train':
        return [aa_path / 'IMDB-train.json']
    if resource_key == 'data.imdb_dev':
        return [aa_path / 'IMDB-dev.json']
    if resource_key == 'data.imdb_test':
        return [aa_path / 'IMDB-test.jsonl']
    if resource_key == 'data.imdb_wiki':
        return [aa_path / 'IMDB-wiki.tsv']

    # news
    if resource_key == 'data.news_train':
        return [aa_path / 'News-train.json']
    if resource_key == 'data.news_dev':
        return [aa_path / 'News-dev.json']
    if resource_key == 'data.news_test':
        return [aa_path / 'News-test.jsonl']
    if resource_key == 'data.news_wiki':
        return [aa_path / 'News-wiki.tsv']

    # twitter
    if resource_key == 'data.twitter_train':
        return [aa_path / 'Twitter-train.json']
    if resource_key == 'data.twitter_dev':
        return [aa_path / 'Twitter-dev.json']
    if resource_key == 'data.twitter_test':
        return [aa_path / 'Twitter-test.jsonl']
    if resource_key == 'data.twitter_wiki':
        return [aa_path / 'Twitter-wiki.tsv']

    raise ValueError(f'Download module: resource key: {resource_key}, local: {local}')
