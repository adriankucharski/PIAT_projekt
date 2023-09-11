import glob
import os
import pickle
from typing import List, Literal, Tuple

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numba

def read_texts(corpus_folder, encoding="utf8", errors=None):
    texts: List[str] = []
    for path in glob.glob(os.path.join(corpus_folder, "*.txt")):
        with open(path, "r", encoding=encoding) as f:
            texts.append(f.read())
    return texts


def save_tokenized_sentences(tokenized_sentences, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(tokenized_sentences, file)


def load_tokenized_sentences(file_path) -> List[str]:
    with open(file_path, "rb") as file:
        tokenized_sentences = pickle.load(file)
    return tokenized_sentences


def convert_encoding(
    path: str, from_format: str = "windows-1250", to_format: str = "utf8"
):
    with open(path, "r", encoding=from_format) as f:
        text = f.read()
    with open(path, "w", encoding=to_format) as f:
        f.write(text)


def get_sequence_of_tokens(
    dataset: List[str],
    tokenizer: Tokenizer,
    sentence_min_len: int,
    sentence_max_len: int,
    skip: int,
) -> List[List[int]]:
    word2int_sequences = []
    for seq_text in tokenizer.texts_to_sequences(dataset):
        for sindex in range(0, len(seq_text) - sentence_max_len, skip):
            rlen = np.random.randint(sentence_min_len, sentence_max_len)
            n_gram_sequence = seq_text[sindex : sindex + rlen]
            word2int_sequences.append(n_gram_sequence)
    return word2int_sequences


def calculate_dataset_size(
    dataset: List[str], tokenizer: Tokenizer, sentence_max_len: int, skip: int
) -> int:
    size = 0
    for seq_text in tokenizer.texts_to_sequences(dataset):
        size += len(range(0, len(seq_text) - sentence_max_len, skip))
    return size

def generate_padded_sequences(
    word2int_sequences: List[List[int]], sentence_max_len: int, for_transformer=False, padding = 'pre'
) -> Tuple[np.ndarray, np.ndarray]:
    if padding == 'pre':
        word2int_sequences_padded = np.array(
            pad_sequences(word2int_sequences, maxlen=sentence_max_len + 1, padding="pre")
        )
        if for_transformer:
            predictors, label = word2int_sequences_padded[:, :-1], word2int_sequences_padded[:, 1:]
        else:
            predictors, label = word2int_sequences_padded[:, :-1], word2int_sequences_padded[:, -1]
    else:
        if for_transformer:
            x, y = [], []
            for seq in word2int_sequences:
                x.append(seq[:-1])
                y.append(seq[1:])
            predictors = np.array(
                pad_sequences(x, maxlen=sentence_max_len, padding="post")
            )
            label = np.array(
                pad_sequences(y, maxlen=sentence_max_len, padding="post")
            )
        else:
            word2int_sequences_padded = np.array(
                pad_sequences(word2int_sequences, maxlen=sentence_max_len, padding="post")
            )
            indexes = np.asarray([len(seq) - 1 for seq in word2int_sequences])
            indices = np.stack([np.arange(len(word2int_sequences)), indexes]).T
            idx = np.s_[indices[:, 0], indices[:, 1]]
            
            label = word2int_sequences_padded[idx]
            word2int_sequences_padded[idx] = 0
            predictors = word2int_sequences_padded
        
    return predictors, label


def dataset_generator(
    dataset: List[str],
    tokenizer: Tokenizer,
    sentence_min_len: int,
    sentence_max_len: int,
    skip: int,
    batch_size=256,
    shuffle=True,
    for_transformer=False,
    padding: Literal["pre", "post"] = "pre",
):
    while True:
        word2int_sequences = get_sequence_of_tokens(
            dataset, tokenizer, sentence_min_len, sentence_max_len, skip
        )
        predictors, label = generate_padded_sequences(
            word2int_sequences, sentence_max_len, for_transformer, padding
        )

        p = (
            np.random.permutation(len(predictors))
            if shuffle
            else np.arange(len(predictors))
        )
        for i in range(0, len(predictors) - batch_size + 1, batch_size):
            indexes = p[i : i + batch_size]
            yield predictors[indexes], label[indexes]

def dataset_generator2(
    dataset: List[str],
    tokenizer: Tokenizer,
    sentence_min_len: int,
    sentence_max_len: int,
    skip: int,
    batch_size=256,
    shuffle=True,
    for_transformer=False,
    padding: Literal["pre", "post"] = "pre",
):
    while True:
        word2int_sequences = get_sequence_of_tokens(
            dataset, tokenizer, sentence_min_len, sentence_max_len, skip
        )
        predictors, label = generate_padded_sequences(
            word2int_sequences, sentence_max_len, for_transformer, padding
        )

        p = (
            np.random.permutation(len(predictors))
            if shuffle
            else np.arange(len(predictors))
        )
        for i in range(0, len(predictors) - batch_size + 1, batch_size):
            indexes = p[i : i + batch_size]
            yield [predictors[indexes], label[indexes]], label[indexes]
