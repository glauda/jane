"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from jane.pipelines import training as training


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    training_pipeline = training.create_pipeline()

    return {
        "__default__": pipeline([]),
        "training": training_pipeline
    }


