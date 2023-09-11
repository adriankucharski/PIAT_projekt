import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.layers import (
    Layer,
    MultiHeadAttention,
    Dense,
    LayerNormalization,
    Dropout,
    Embedding,
)
from keras import Model, losses, Sequential, callbacks, activations, utils
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from typing import Literal
import numpy as np
from typing import Tuple
from keras import layers, activations

class MaskedSparseCategoricalCrossentropy(losses.Loss):
    def __init__(self, from_logits: bool = True, pad_value: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value
        self.loss = losses.SparseCategoricalCrossentropy(from_logits, reduction="none")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != self.pad_value, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


class TransformerBlock(Layer):
    def __init__(self, embedding_dim: int, num_heads: int, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.emb = Sequential(
            [
                Dense(self.embedding_dim, activation="relu"),
                Dense(self.embedding_dim),
            ]
        )
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)
        self.layernorm1 = LayerNormalization(epsilon=1e-5)
        self.layernorm2 = LayerNormalization(epsilon=1e-5)
        self.mha_mask = MultiHeadAttention(
            self.num_heads, self.embedding_dim, name="mha_mask"
        )
        self.mha = MultiHeadAttention(self.num_heads, self.embedding_dim, name="mha")

    def call(self, inputs: Tuple[tf.Tensor]):
        dec, enc = inputs
        print(dec.shape, enc.shape)
        batch_size, seq_len = tf.shape(dec)[0], tf.shape(dec)[0]
        # attention_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_mask = self._mask(tf.shape(dec)[1])
        attention_out = self.mha_mask(dec, dec, attention_mask=attention_mask)
        attention_out = self.dropout1(attention_out)
        attention_out = self.layernorm1(dec + attention_out)

        # padding_mask = self._create_padding_mask(dec)
        mha_out = self.mha(enc, attention_out)
        embedding_out = self.emb(mha_out)
        embedding_out = self.dropout2(embedding_out)
        return self.layernorm2(mha_out + embedding_out)

    def _mask(self, size: int):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def _create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def causal_attention_mask(
        self, batch_size: int, len_input: int, len_output: int, dtype: tf.DType
    ):
        range_input = tf.range(len_input)[:, tf.newaxis]
        range_output = tf.range(len_output)
        attention_mask = range_input >= range_output - len_output + len_input
        attention_mask = tf.cast(attention_mask, dtype)
        attention_mask = tf.reshape(attention_mask, [1, len_input, len_output])

        const = tf.constant([1, 1], dtype=tf.int32)
        batch = tf.expand_dims(batch_size, -1)
        tiling_tensor = tf.concat([batch, const], 0)

        return tf.tile(attention_mask, tiling_tensor)

    def get_config(self):
        config = {}
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class TokenAndPositionEmbedding(Layer):
    def __init__(
        self,
        sequence_length: int,
        vocabulary_size: int,
        embedding_dim: int,
        mask_zero: bool = True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size

        self.token_emb = Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_dim,
            mask_zero=mask_zero,
        )
        self.pos_emb = Embedding(
            input_dim=self.sequence_length, output_dim=self.embedding_dim
        )

    def call(self, x: tf.Tensor):
        positions = tf.range(start=0, limit=tf.shape(x)[-1])
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = {}
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
                "vocabulary_size": self.vocabulary_size,
            }
        )
        return config


class TextGenerator(callbacks.Callback):
    def __init__(
        self,
        seed_text: str,
        next_words: int,
        max_sequence_len: int,
        tokenizer: Tokenizer,
        top_k=10,
        print_every=1,
        model=None,
        padding: Literal["pre", "post"] = "pre",
    ):
        self.seed_text = seed_text
        self.next_words = next_words
        self.max_sequence_len = max_sequence_len
        self.tokenizer = tokenizer
        if model is not None:
            self.model: Model = model
        self.print_every = print_every
        self.k = top_k
        self.padding = padding

    def sample_from(self, logits: np.ndarray) -> np.ndarray:
        indices = logits.argpartition(-self.k)[-self.k :].astype("int32")
        logits = logits[indices]
        preds = activations.softmax(tf.expand_dims(logits, 0))
        preds = np.array(preds[0]).astype("float32")
        return np.random.choice(indices, p=preds)

    def generate_text(self) -> str:
        start_tokens = self.tokenizer.texts_to_sequences([self.seed_text])[0]
        tokens_generated = []
        while len(tokens_generated) <= self.next_words:
            x = utils.pad_sequences(
                [start_tokens], maxlen=self.max_sequence_len, padding=self.padding
            )
            z = utils.pad_sequences(
                [start_tokens[1:]], maxlen=self.max_sequence_len, padding=self.padding
            )

            y = self.model.predict_on_batch([x, z])[0]

            idx = -1
            if self.padding == "post":
                idx = min(len(start_tokens) - 1, self.max_sequence_len - 1)

            sample_token = self.sample_from(y[idx])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)

        token_to_word = []
        for tok in tokens_generated:
            try:
                word = self.tokenizer.index_word[tok]
                token_to_word.append(word)
            except:
                token_to_word.append("")
        txt = self.seed_text + " " + " ".join(token_to_word)
        return txt

    def on_epoch_begin(self, epoch: int, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        txt = self.generate_text()
        print(f"Epoch: {epoch}; Generated text:\n{txt}\n")


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.path, save_format="tf")





class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        intermediate_dim: int,
        num_heads: int,
        dropout=0,
        layer_norm_epsilon=1e-05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.supports_masking = True

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = layers.MultiHeadAttention(
            self.num_heads,
            key_dim,
            dropout=self.dropout,
        )
        self._self_attention_layer.build(
            query_shape=inputs_shape,
            value_shape=inputs_shape,
        )

        # Feedforward layers.
        self._feedforward_layernorm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_layernorm.build(inputs_shape)
        self._feedforward_intermediate_dense = layers.Dense(
            self.intermediate_dim,
            activation="relu",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = layers.Dense(
            hidden_dim,
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = layers.Dropout(
            rate=self.dropout,
        )
        self._self_attention_dropout = layers.Dropout(
            rate=self.dropout,
        )
        self.built = True

    def call(self, inputs):
        x = inputs  
        residual = x
        x = self._self_attention_layer(x,x)
        x = self._self_attention_dropout(x)
        x = x + residual

        residual = x
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        x = self._feedforward_layernorm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        embedding_dim: int,
        mask_zero=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if vocabulary_size is None:
            raise ValueError(
                "`vocabulary_size` must be an Integer, received `None`."
            )
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        if embedding_dim is None:
            raise ValueError(
                "`embedding_dim` must be an Integer, received `None`."
            )
        self.vocabulary_size = int(vocabulary_size)
        self.sequence_length = int(sequence_length)
        self.embedding_dim = int(embedding_dim)
        self.token_embedding = layers.Embedding(
            vocabulary_size,
            embedding_dim,
            mask_zero=mask_zero,
        )
        self.position_embedding = PositionEmbedding(
            sequence_length=sequence_length,
        )
        self.supports_masking = self.token_embedding.supports_masking

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.token_embedding.build(input_shape)
        self.position_embedding.build(input_shape + (self.embedding_dim,))
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
                ),
                "mask_zero": self.token_embedding.mask_zero,
            },
        )
        return config

    def call(self, inputs, start_index=0):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(
            embedded_tokens,
            start_index=start_index,
        )
        outputs = embedded_tokens + embedded_positions
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.token_embedding.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.embedding_dim,)
