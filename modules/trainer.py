"""
Author: Macreai
Date: 13/9/2024
This is the trainer.py module.
Usage:
- Train
"""
# pylint: disable=no-member
import pickle
import os

import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_transform as tft

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
    """Get post_transform feature & create batches of data

    Args:
        file_pattern: File pattern for the TFRecord files (string or list of strings).
        tf_transform_output: A `TFTransformOutput` object containing transformation specs.
        num_epochs: Number of epochs to repeat the dataset. 
        If `None`, the dataset is repeated indefinitely.
        batch_size: Number of records per batch.

    Returns:
        A `tf.data.Dataset` object with transformed features and batching applied.
    """
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

    return dataset.repeat(num_epochs)


def model_builder(hp):
    """Builds and compiles a Keras model for classification

    Args:
        hp: A dictionary of hyperparameters, where:
            - "num_hidden_layers_1": Number of hidden layers.
            - "unit_<i>": Number of units in the i-th hidden layer.
            - "dropout_rate_<i>": Dropout rate for the i-th hidden layer.
            - "learning_rate": Learning rate for the optimizer.

    Returns:
        A compiled Keras model.
    """

    input_layer = [
        tf.keras.layers.Input(
            name=transformed_name(colname),
            shape=(
                1,
            ),
            dtype=tf.float32) for colname in FEATURE_KEYS]

    x = tf.keras.layers.concatenate(input_layer)

    for i in range(hp["num_hidden_layers"]):
        x = tf.keras.layers.Dense(hp['unit_' + str(i)], activation='relu')(x)
        x = tf.keras.layers.Dropout(hp['dropout_rate_' + str(i)])(x)

    x = tf.keras.layers.Dense(hp['unit_' + str(i)], activation='relu')(x)

    output = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(input_layer, output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        metrics=['accuracy']
    )

    model.summary()

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function for serving transformed examples

    Args:
        model: The trained Keras model.
        tf_transform_output: A `TFTransformOutput` object containing transformation specs.

    Returns:
        A function that takes serialized TFExamples, 
        applies transformations, and returns model predictions.
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
    """Returns a function for transforming features for evaluation

    Args:
        model: The trained Keras model.
        tf_transform_output: A `TFTransformOutput` object containing
        transformation specs.

    Returns:
        A function that takes serialized TFExamples and returns transformed features.
    """

    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn


def run_fn(fn_args: FnArgs) -> None:
    """Train and save the model

    Args:
        fn_args: An `FnArgs` object containing:
            - train_files: List of file paths to the training data.
            - eval_files: List of file paths to the evaluation data.
            - transform_graph_path: Path to the transformation graph.
            - hyperparameters: Dictionary of hyperparameters for the model.
            - serving_model_dir: Directory to save the trained model.
            - train_steps: Number of training steps per epoch.
            - eval_steps: Number of evaluation steps per epoch.

    Returns:
        None
    """
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    model = model_builder(fn_args.hyperparameters["values"])

    model.fit(x=train_set,
              validation_data=val_set,
              callbacks=[tensorboard_callback, es, mc],
              steps_per_epoch=fn_args.train_steps,
              validation_steps=fn_args.eval_steps,
              epochs=50)

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='heartbeat_training')),

        'transform_features':
        _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)
