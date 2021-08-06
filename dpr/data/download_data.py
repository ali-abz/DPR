#!/usr/bin/env python3
import pathlib

def download(resource_key: str, out_dir: str = None):
    # porseman 1 (mozafari)
    # porseman_path = pathlib.Path('/content/drive/MyDrive/IRDateset')
    porseman_path = pathlib.Path('/home/ali/Downloads/Colab Notebooks/IRDataset/')
    if resource_key == 'data.porseman-train': return [porseman_path / 'IR-train.json']
    if resource_key == 'data.porseman-dev':  return [porseman_path / 'IR-dev.json']
    if resource_key == 'data.porseman-test':  return [porseman_path / 'IR-test.csv']
    if resource_key == 'data.porseman_wiki':  return [porseman_path / 'IR-wiki.tsv']

    # porseman 2
    # porseman_path2 = pathlib.Path('/content/drive/MyDrive/IRDateset2')
    porseman_path2 = pathlib.Path('/home/ali/Desktop/sem4/projects/DPR-proper-small-dataset/outputs/')
    if resource_key == 'data.porseman-train2': return [porseman_path2 / 'train_questions.jsonl']
    if resource_key == 'data.porseman-dev2': return [porseman_path2 / 'dev_questions.jsonl']
    if resource_key == 'data.porseman-test2': return [porseman_path2 / 'test.jsonl']
    if resource_key == 'data.porseman-train2_pos': return [porseman_path2 / 'train_questions_pos.jsonl']
    if resource_key == 'data.porseman-dev2_pos': return [porseman_path2 / 'dev_questions_pos.jsonl']
    if resource_key == 'data.porseman_wiki2': return [porseman_path2 / 'collection.csv']

    # trivia
    # trivia_path = pathlib.Path('/content/drive/MyDrive/trivia/')
    trivia_path = pathlib.Path('/content/drive/MyDrive/trivia/')
    if resource_key == 'data.trivia-train-1': return [trivia_path / 'trivia-train-1.jl']
    if resource_key == 'data.trivia-train-40': return [trivia_path / 'trivia-train-40.jl']
    if resource_key == 'data.trivia-train-79': return [trivia_path / 'trivia-train-79.jl']
    if resource_key == 'data.trivia-dev-1': return [trivia_path / 'trivia-dev-1.jl']
