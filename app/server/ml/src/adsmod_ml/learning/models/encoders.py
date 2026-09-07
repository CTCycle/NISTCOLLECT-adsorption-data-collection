from __future__ import annotations

from typing import Any

import keras
from keras import activations, layers

from adsmod_ml.common.constants import PAD_VALUE
from adsmod_ml.learning.models.transformers import AddNorm, FeedForward


# [STATE ENCODER]

###############################################################################
@keras.saving.register_keras_serializable(package="Encoders", name="StateEncoder")
class StateEncoder(keras.layers.Layer):

    # -------------------------------------------------------------------------
    def __init__(self, dropout_rate: float = 0.2, seed: int = 42, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dense_units = [16, 32, 64, 128, 128]
        self.dense_layers = [
            layers.Dense(units, kernel_initializer="he_uniform")
            for units in self.dense_units
        ]
        self.bn_layers = [layers.BatchNormalization() for _ in self.dense_units]
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.seed = seed

    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super().build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, x, training: bool | None = None) -> Any:
        layer = keras.ops.expand_dims(x, axis=-1)
        for dense, bn in zip(self.dense_layers, self.bn_layers):
            layer = dense(layer)
            layer = bn(layer, training=training)
            layer = activations.relu(layer)

        output = self.dropout(layer, training=training)

        return output

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"dropout_rate": self.dropout_rate, "seed": self.seed})
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls: type[StateEncoder], config: dict[str, Any]) -> StateEncoder:
        return cls(**config)


# [PRESSURE SERIES ENCODER]

###############################################################################
@keras.saving.register_keras_serializable(
    package="Encoders", name="PressureSerierEncoder"
)
class PressureSerierEncoder(keras.layers.Layer):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        embedding_dims: int,
        dropout_rate: float,
        num_heads: int,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.seed = seed
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dims
        )
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)
        self.ffn2 = FeedForward(self.embedding_dims, 0.3, seed)
        self.P_dense = layers.Dense(
            self.embedding_dims, kernel_initializer="he_uniform"
        )
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)

        self.supports_masking = True
        self.attention_scores = {}

    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super().build(input_shape)

    # -------------------------------------------------------------------------
    def call(
        self, pressure, context, key_mask=None, training: bool | None = None
    ) -> Any:
        query_mask = self.compute_mask(pressure)
        pressure = keras.ops.expand_dims(pressure, axis=-1)
        pressure = self.P_dense(pressure)

        attention_output, cross_attention_scores = self.attention(
            query=pressure,
            key=context,
            value=context,
            query_mask=query_mask,
            key_mask=key_mask,
            value_mask=key_mask,
            training=training,
            return_attention_scores=True,
        )
        addnorm = self.addnorm1([pressure, attention_output])
        self.attention_scores["cross_attention_scores"] = cross_attention_scores

        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])

        return output

    # -------------------------------------------------------------------------
    def get_attention_scores(self) -> dict:
        return self.attention_scores

    # -------------------------------------------------------------------------
    def compute_mask(self, inputs: Any, previous_mask=None) -> Any:
        mask = keras.ops.not_equal(inputs, PAD_VALUE)
        mask = keras.ops.cast(mask, keras.config.floatx())

        return mask

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embedding_dims": self.embedding_dims,
                "dropout_rate": self.dropout_rate,
                "num_heads": self.num_heads,
                "seed": self.seed,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[PressureSerierEncoder], config: dict[str, Any]
    ) -> PressureSerierEncoder:
        return cls(**config)


# [UPTAKE DECODER]

###############################################################################
@keras.saving.register_keras_serializable(package="Decoders", name="QDecoder")
class QDecoder(keras.layers.Layer):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        embedding_dims: int = 128,
        dropout_rate: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.state_dense = layers.Dense(
            self.embedding_dims, kernel_initializer="he_uniform"
        )
        self.dense = [
            layers.Dense(self.embedding_dims, kernel_initializer="he_uniform")
            for _ in range(4)
        ]
        self.batch_norm = [layers.BatchNormalization() for _ in range(4)]
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.Q_output = layers.Dense(1, kernel_initializer="he_uniform")
        self.supports_masking = True

    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super().build(input_shape)

    # -------------------------------------------------------------------------
    def compute_mask(self, inputs: Any, mask=None) -> Any:
        if mask is None:
            mask = keras.ops.not_equal(inputs, PAD_VALUE)
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask = keras.ops.cast(mask, keras.config.floatx())

        return mask

    # -------------------------------------------------------------------------
    def call(
        self, p_logits, pressure, state, mask=None, training: bool | None = None
    ) -> Any:
        mask = self.compute_mask(pressure) if mask is None else mask
        layer = p_logits * mask if mask is not None else p_logits

        state = self.state_dense(state)
        state = activations.relu(state)
        t_scale = keras.ops.expand_dims(keras.ops.exp(-state), axis=1)

        for dense, bn in zip(self.dense, self.batch_norm):
            layer = dense(layer)
            layer = bn(layer, training=training)
            layer = activations.relu(layer)
            layer = layer * t_scale

        output = self.Q_output(layer)
        output = activations.relu(output)

        return output

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "embedding_dims": self.embedding_dims,
                "seed": self.seed,
            }
        )
        return config

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls: type[QDecoder], config: dict[str, Any]) -> QDecoder:
        return cls(**config)
