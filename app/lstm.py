import re
from typing import Literal
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.optimizers import Adam

import tensorflow as tf

import numpy as np

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

LEN_MIN_LIMIT = 5
LEN_MAX_LIMIT = 50


class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        seed_text: str,
        next_words: int,
        max_sequence_len: int,
        tokenizer: Tokenizer,
        padding: Literal["post", "pre"] = "pre",
        temperature=0.0,
        model: Model = None,
    ):
        self.seed_text = seed_text
        self.next_words = next_words
        self.max_sequence_len = max_sequence_len
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.padding = padding
        if model is not None:
            self.model: Model = model

    def sample(self, preds: np.ndarray):
        if self.temperature > 0:
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds) / self.temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            preds = np.random.multinomial(1, preds, 1)
        return np.argmax(preds)

    def generate_text(self, _seed_text: str = None):
        seed_text: str = self.seed_text if _seed_text is None else _seed_text
        for _ in range(self.next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_sequence_len, padding=self.padding
            )
            predicted = self.model.predict_on_batch(token_list)[0]
            predicted = self.sample(predicted)
            try:
                output_word = self.tokenizer.index_word[predicted]
            except:
                output_word = ""
            seed_text += " " + output_word

        # seed_text = re.sub(r'\s+([.,?!:;()\n])', r'\1', seed_text)
        return seed_text

    def on_epoch_begin(self, epoch, logs=None):
        seed_text = self.generate_text()
        print(
            f"Start epoch {epoch} of training; Temperature: {self.temperature:.1f} Generated text:",
            seed_text,
        )


def create_model(max_sequence_len: int, total_words: int):
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_len))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.2))
    model.add(Dense(total_words + 1, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


def create_model_v2(max_sequence_len: int, total_words: int):
    model = Sequential()
    model.add(Embedding(total_words, 75, input_length=max_sequence_len))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(total_words + 1, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(1e-3))
    return model


"""
1. dodanie do zbioru znaków .,?!;:
2. dodanie temperatury
3. usunięcie niektórych słów 
4. dodanie bidirectional
5. ograniczenie wyjścia dense 
6. przygotowanie zbioru
"""
