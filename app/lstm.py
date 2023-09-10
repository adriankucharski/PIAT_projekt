import re
from typing import List, Literal
from keras import losses
from keras.utils import pad_sequences, losses_utils
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


class WeightedSCCE(losses.Loss):
    def __init__(
        self, class_weight: List[float], from_logits=False, name="weighted_scce"
    ):
        if class_weight is None or all(v == 1.0 for v in class_weight):
            self.class_weight = None
        else:
            self.class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
        self.name = name
        self.reduction = losses_utils.ReductionV2.NONE
        self.unreduced_scce = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, name=name, reduction=self.reduction
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.unreduced_scce(y_true, y_pred, sample_weight)
        if self.class_weight is not None:
            weight_mask = tf.gather(self.class_weight, y_true)
            loss = tf.math.multiply(loss, weight_mask)
        return loss

    

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


def calculate_class_weights(tokenizer: Tokenizer, alpha: float = 1.0) -> List[float]:
    num_classes = tokenizer.num_words
    word_count = sorted(tokenizer.word_counts.values(), reverse=True)[:num_classes]
    class_weights = sum(word_count) / np.array(word_count, dtype=float) ** alpha
    val = np.concatenate([[0], class_weights])
    return val / np.min(val[np.nonzero(val)])

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


def create_model_v2(
    max_sequence_len: int, total_words: int, class_weight: List[float] = None
):
    model = Sequential()
    model.add(
        Embedding(total_words, 50, input_length=max_sequence_len, mask_zero=False)
    )
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(total_words + 1, activation="softmax"))

    loss = "sparse_categorical_crossentropy"
    if class_weight is not None:
        loss = WeightedSCCE(class_weight)
    model.compile(loss=loss, optimizer="adam")
    return model


"""
1. dodanie do zbioru znaków .,?!;:
2. dodanie temperatury
3. usunięcie niektórych słów 
4. dodanie bidirectional
5. ograniczenie wyjścia dense 
6. przygotowanie zbioru
"""
