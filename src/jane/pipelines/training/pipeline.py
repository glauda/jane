"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from jane.pipelines.training.nodes import (
    medium_extraction_node, 
    text_preprocessing_node, 
    model_training_node, 
    model_prediction_node
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        medium_extraction_node,
        text_preprocessing_node,
        model_training_node,
        model_prediction_node
    ])
