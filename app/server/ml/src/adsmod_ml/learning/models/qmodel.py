from __future__ import annotations

from typing import Any

import keras
from keras import Model, activations, layers, optimizers

from adsmod_ml.learning.metrics import MaskedMeanSquaredError, MaskedRSquared
from adsmod_ml.learning.models.embeddings import MolecularEmbedding
from adsmod_ml.learning.models.encoders import (
    PressureSerierEncoder,
    QDecoder,
    StateEncoder,
)
from adsmod_ml.learning.models.transformers import TransformerEncoder
from adsmod_ml.learning.training.scheduler import LinearDecayLRScheduler

from torch import compile as torch_compile


# [SCADS SEQUENCE MODEL]

###############################################################################
class SCADSModel:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any], metadata: dict[str, Any]) -> None:
        smile_vocab = metadata.get("smile_vocabulary") or {}
        ads_vocab = metadata.get("adsorbent_vocabulary") or {}
        max_smile_index = max(smile_vocab.values()) if smile_vocab else 0
        self.smile_vocab_size = int(max_smile_index) + 1
        self.ads_vocab_size = metadata.get("adsorbent_vocabulary_size") or len(
            ads_vocab
        )
        self.smile_length = metadata.get("smile_sequence_size", 20)
        self.series_length = metadata.get("max_measurements", 30)

        self.seed = (
            configuration.get("train_seed")
            or configuration.get("training_seed")
            or configuration.get("seed")
            or 42
        )
        self.dropout_rate = configuration.get("dropout_rate", 0.2)
        self.embedding_dims = configuration.get("molecular_embedding_size", 64)
        self.num_heads = configuration.get("num_attention_heads", 2)
        self.num_encoders = configuration.get("num_encoders", 2)
        self.jit_compile = configuration.get("use_jit", False) or configuration.get(
            "jit_compile", False
        )
        self.jit_backend = configuration.get("jit_backend", "inductor")
        self.configuration = configuration

        self.state_encoder = StateEncoder(self.dropout_rate, seed=self.seed)
        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size,
            self.ads_vocab_size,
            self.embedding_dims,
            self.smile_length,
            mask_values=True,
        )
        self.encoders = [
            TransformerEncoder(self.embedding_dims, self.num_heads, self.seed)
            for _ in range(self.num_encoders)
        ]
        self.pressure_encoder = PressureSerierEncoder(
            self.embedding_dims, self.dropout_rate, self.num_heads, self.seed
        )
        self.q_decoder = QDecoder(self.embedding_dims, self.dropout_rate, self.seed)

        self.state_input = layers.Input(shape=(), name="state_input")
        self.chemo_input = layers.Input(shape=(), name="chemo_input")
        self.adsorbents_input = layers.Input(shape=(), name="adsorbent_input")
        self.adsorbates_input = layers.Input(
            shape=(self.smile_length,), name="adsorbate_input"
        )
        self.pressure_input = layers.Input(
            shape=(self.series_length,), name="pressure_input"
        )

    # -------------------------------------------------------------------------
    def compile_model(self, model: Model, model_summary: bool = True) -> Model:
        initial_lr = (
            self.configuration.get("initial_lr")
            or self.configuration.get("initial_LR")
            or self.configuration.get("initial_RL")
            or 0.001
        )
        lr_schedule = initial_lr
        use_scheduler = self.configuration.get(
            "use_lr_scheduler", False
        ) or self.configuration.get("use_scheduler", False)
        if use_scheduler:
            constant_steps = self.configuration.get("constant_steps", 1000)
            decay_steps = self.configuration.get("decay_steps", 1000)
            target_lr = self.configuration.get("target_lr", 0.0001)
            lr_schedule = LinearDecayLRScheduler(
                initial_lr, constant_steps, decay_steps, target_lr
            )

        opt = optimizers.AdamW(learning_rate=lr_schedule)  # type: ignore
        loss = MaskedMeanSquaredError()
        metric = [MaskedRSquared()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)  # type: ignore
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode="default")  # type: ignore

        return model

    # -------------------------------------------------------------------------
    def get_model(self, model_summary: bool = True) -> Model:
        molecular_embeddings = self.molecular_embeddings(
            self.adsorbates_input, self.adsorbents_input, self.chemo_input
        )
        smile_mask = self.molecular_embeddings.compute_mask(self.adsorbates_input)

        encoder_output = molecular_embeddings
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, mask=smile_mask, training=False)

        encoded_states = self.state_encoder(self.state_input, training=False)

        encoded_pressure = self.pressure_encoder(
            self.pressure_input, encoder_output, smile_mask, training=False
        )

        output = self.q_decoder(encoded_pressure, self.pressure_input, encoded_states)

        model = Model(
            inputs=[
                self.state_input,
                self.chemo_input,
                self.adsorbents_input,
                self.adsorbates_input,
                self.pressure_input,
            ],
            outputs=output,
            name="SCADS_model",
        )
        model = self.compile_model(model, model_summary=model_summary)

        return model

###############################################################################
class SCADSAtomicModel:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any], metadata: dict[str, Any]) -> None:
        smile_vocab = metadata.get("smile_vocabulary") or {}
        ads_vocab = metadata.get("adsorbent_vocabulary") or {}
        max_smile_index = max(smile_vocab.values()) if smile_vocab else 0
        self.smile_vocab_size = int(max_smile_index) + 1
        self.ads_vocab_size = metadata.get("adsorbent_vocabulary_size") or len(
            ads_vocab
        )
        self.smile_length = metadata.get("smile_sequence_size", 20)
        self.feature_size = 4

        self.seed = (
            configuration.get("train_seed")
            or configuration.get("training_seed")
            or configuration.get("seed")
            or 42
        )
        self.dropout_rate = configuration.get("dropout_rate", 0.2)
        self.embedding_dims = configuration.get("molecular_embedding_size", 64)
        self.jit_compile = configuration.get("use_jit", False) or configuration.get(
            "jit_compile", False
        )
        self.jit_backend = configuration.get("jit_backend", "inductor")
        self.configuration = configuration

        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size,
            self.ads_vocab_size,
            self.embedding_dims,
            self.smile_length,
            mask_values=True,
        )
        self.context_pooling = layers.GlobalAveragePooling1D(
            name="molecular_context_pooling"
        )
        self.feature_projection = layers.Dense(
            self.embedding_dims,
            kernel_initializer="he_uniform",
            name="feature_projection",
        )
        self.hidden_blocks = [
            layers.Dense(self.embedding_dims, kernel_initializer="he_uniform")
            for _ in range(2)
        ]
        self.block_norms = [layers.BatchNormalization() for _ in range(2)]
        self.dropout = layers.Dropout(rate=self.dropout_rate, seed=self.seed)
        self.output_head = layers.Dense(
            1,
            kernel_initializer="he_uniform",
            name="adsorption_output",
        )

        self.adsorbates_input = layers.Input(
            shape=(self.smile_length,), name="adsorbate_input"
        )
        self.features_input = layers.Input(
            shape=(self.feature_size,), name="features_input"
        )

    # -------------------------------------------------------------------------
    def compile_model(self, model: Model, model_summary: bool = True) -> Model:
        initial_lr = (
            self.configuration.get("initial_lr")
            or self.configuration.get("initial_LR")
            or self.configuration.get("initial_RL")
            or 0.001
        )
        lr_schedule = initial_lr
        use_scheduler = self.configuration.get(
            "use_lr_scheduler", False
        ) or self.configuration.get("use_scheduler", False)
        if use_scheduler:
            constant_steps = self.configuration.get("constant_steps", 1000)
            decay_steps = self.configuration.get("decay_steps", 1000)
            target_lr = self.configuration.get("target_lr", 0.0001)
            lr_schedule = LinearDecayLRScheduler(
                initial_lr, constant_steps, decay_steps, target_lr
            )

        opt = optimizers.AdamW(learning_rate=lr_schedule)  # type: ignore
        loss = MaskedMeanSquaredError()
        metric = [MaskedRSquared()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)  # type: ignore
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode="default")  # type: ignore

        return model

    # -------------------------------------------------------------------------
    def get_model(self, model_summary: bool = True) -> Model:
        adsorbent_index = keras.ops.cast(self.features_input[:, 2], "int32")
        chemometric_feature = self.features_input[:, 1]
        molecular_embeddings = self.molecular_embeddings(
            self.adsorbates_input, adsorbent_index, chemometric_feature
        )
        context_vector = self.context_pooling(molecular_embeddings)

        temperature_feature = keras.ops.expand_dims(self.features_input[:, 0], axis=-1)
        molecular_weight_feature = keras.ops.expand_dims(self.features_input[:, 1], axis=-1)
        pressure_feature = keras.ops.expand_dims(self.features_input[:, 3], axis=-1)
        numeric_features = layers.concatenate(
            [temperature_feature, molecular_weight_feature, pressure_feature],
            name="numeric_features",
        )
        projected_features = self.feature_projection(numeric_features)
        projected_features = activations.relu(projected_features)

        combined = layers.concatenate(
            [context_vector, projected_features],
            name="measurement_features",
        )

        layer = combined
        for dense, norm in zip(self.hidden_blocks, self.block_norms):
            layer = dense(layer)
            layer = norm(layer, training=False)
            layer = activations.relu(layer)
            layer = self.dropout(layer, training=False)

        output = self.output_head(layer)
        output = activations.relu(output)

        model = Model(
            inputs=[self.adsorbates_input, self.features_input],
            outputs=output,
            name="SCADS_ATOMIC_MODEL",
        )
        model = self.compile_model(model, model_summary=model_summary)

        return model
