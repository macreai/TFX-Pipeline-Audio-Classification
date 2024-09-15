"""
Author: Macreai
Date: 13/9/2024
This is the tuner.py module.
Usage:
- Tuner
"""
# pylint: disable=no-member
from typing import NamedTuple, Dict, Text, Any
import pickle

from keras_tuner.engine import base_tuner
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
import keras_tuner as kt
import tensorflow_transform as tft

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])


with open("modules/LABEL_KEY.pkl", "rb") as f:
    LABEL_KEY = pickle.load(f)

with open("modules/FEATURE_KEYS.pkl", "rb") as f:
    FEATURE_KEYS = pickle.load(f)


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


def transformed_name(key):
    """Renaming transformed features"""
    key = key.replace(' ', '_')
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs=None,
        batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


def model_builder(hp):
    """
    Builds the neural network model

    Args:
        hp: Hyperparameters object used for tuning, passed by Keras Tuner

    Return:
        model: A compiled Keras model
    """

    num_hidden_layers = hp.Choice("num_hidden_layers", values=[1, 2, 3])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    input_layer = [
        tf.keras.layers.Input(
            name=transformed_name(colname),
            shape=(
                1,
            ),
            dtype=tf.float32) for colname in FEATURE_KEYS]

    x = tf.keras.layers.concatenate(input_layer)

    for i in range(num_hidden_layers):
        num_nodes = hp.Int(
            'unit_' + str(i),
            min_value=8,
            max_value=256,
            step=64)
        x = tf.keras.layers.Dense(num_nodes, activation='relu')(x)
        num_dropout_rate = hp.Float(
            'dropout_rate_' + str(i),
            min_value=0.0,
            max_value=0.5,
            step=0.1)
        x = tf.keras.layers.Dropout(num_dropout_rate)(x)

    num_nodes = hp.Int(
            'unit_' + str(i),
            min_value=8,
            max_value=256,
            step=64)
    x = tf.keras.layers.Dense(num_nodes, activation='relu')(x)

    output = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(input_layer, output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    model.summary()

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Defines the tuner and training setup

    Args:
        fn_args: A `FnArgs` object containing information about
        the training and evaluation datasets, transformation graph,
        and other training settings

    Return:
        A `TunerFnResult` containing the configured tuner
        and the fit arguments for training the model
    """

    tuner = kt.GridSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=20,
        directory=fn_args.working_dir,
        project_name='heartbeat_classification'
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps
        }
    )
