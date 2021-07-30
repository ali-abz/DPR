#!/usr/bin/env python3
import os

def download(resource_key: str, out_dir: str = None):
    # porseman resources
    porseman_datasets_path = '/content/drive/MyDrive/IRDataset/'
    if resource_key == 'data.retriever.porseman-train':
        path = os.path.join(porseman_datasets_path, 'IR-train.json')
        return [path]
    if resource_key == 'data.retriever.porseman-dev':
        path = os.path.join(porseman_datasets_path, 'IR-dev.json')
        return [path]
    if resource_key == 'data.retriever.porseman-test':
        path = os.path.join(porseman_datasets_path, 'IR-test.csv')
        return [path]
    if resource_key == 'data.porseman_wiki':
        path = os.path.join(porseman_datasets_path, 'IR-wiki.tsv')
        return [path]

    # porseman 2 (5 hn per q)
    porseman_datasets_path2 = '/content/drive/MyDrive/IRDateset2'
    if resource_key == 'data.retriever.porseman-train2':
        path = os.path.join(porseman_datasets_path2, 'train_questions.jsonl')
        return [path]
    if resource_key == 'data.retriever.porseman-dev2':
        path = os.path.join(porseman_datasets_path2, 'dev_questions.jsonl')
        return [path]

    # trivia
    trivia_datasets_path = '/content/drive/MyDrive/trivia/'
    if resource_key == 'data.retriever.trivia-train-1':
        path = os.path.join(trivia_datasets_path, 'trivia-train-1.jl')
        return path
    if resource_key == 'data.retriever.trivia-train-40':
        path = os.path.join(trivia_datasets_path, 'trivia-train-40.jl')
        return path
    if resource_key == 'data.retriever.trivia-train-79':
        path = os.path.join(trivia_datasets_path, 'trivia-train-79.jl')
        return path
    if resource_key == 'data.retriever.trivia-dev-1':
        path = os.path.join(trivia_datasets_path, 'trivia-dev-1.jl')