"""
Author: Macreai
Date: 13/9/2024
This is the transform.py module.
Usage:
- Transform
"""

import pickle

import tensorflow_transform as tft

with open("modules/LABEL_KEY.pkl", "rb") as f:
    LABEL_KEY = pickle.load(f)

with open("modules/FEATURE_KEYS.pkl", "rb") as f:
    FEATURE_KEYS = pickle.load(f)


def transformed_name(key):
    """Renaming transformed features"""
    key = key.replace(' ', '_')
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    for key in FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY])

    return outputs
