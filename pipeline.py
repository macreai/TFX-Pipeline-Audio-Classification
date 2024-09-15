"""
Author: Macreai
Date: 13/9/2024
This is the pipeline.py module.
Usage:
- Pipeline
"""
# pylint: disable=redefined-outer-name
import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "arda24-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"
# requirement_file = os.path.join(root, "requirements.txt")

# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """
    Sets up a local TFX pipeline with specified components and configuration

    Args:
        components: A list or tuple of TFX components that 
        have been initialized and are part of the pipeline.
        These components define the various stages of the pipeline, 
        such as data ingestion, transformation, training, evaluation, 
        and deployment.
        pipeline_root:  The root directory where the pipeline's artifacts, 
        such as models, data, and metadata, will be stored.
        It should be a string representing the file system path.

    Returns:
        pipeline_name: The name of the pipeline. 
        PIPELINE_NAME should be defined elsewhere in the code.
        pipeline_root: The root directory for storing pipeline artifacts.
        components: The list or tuple of TFX components to be included in the pipeline.
        enable_cache: Whether to enable caching for the pipeline. 
        This can help speed up subsequent pipeline runs by reusing intermediate results.
        metadata_connection_config: Configuration for connecting to the
        metadata store. Uses SQLite for local execution, 
        with metadata_path specifying the path to the SQLite database.
        beam_pipeline_args: Arguments for configuring the Apache Beam runner.
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        serving_model_dir=serving_model_dir,
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
