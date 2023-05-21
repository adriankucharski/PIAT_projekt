from collections import defaultdict
import itertools
from multiprocessing import Pool
import os
import pickle
from typing import Dict, List
from dataset import load_tokenized_sentences
from tqdm import tqdm

def get_words_indexes(dataset: List[str]) -> Dict[str, List[int]]:
    data = defaultdict(list)
    for index, word in enumerate(tqdm(dataset)):
        data[word].append(index)
    return data

if __name__ == "__main__":
    dataset = load_tokenized_sentences("datasets/pickled/books_3500_flat.pickle")
    result = get_words_indexes(dataset)
    with open('datasets/pickled/books_3500_occurs.pickle', 'wb') as f:
        pickle.dump(result, f)
    