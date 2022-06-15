"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.1
"""

from kedro.pipeline import node

from jane.datalab.ETL.scraping import create_extracted_dataset
from jane.datalab.ETL.text_preprocessing import apply_preprocessing
from jane.datalab.models.tf_idf import train_tf_idf, prediction_tf_idf

medium_extraction_node = node(
    create_extracted_dataset,
    name="article_extraction",
    inputs=["params:start_date", "params:end_date", "params:nb_iteration", "params:base_url", "params:seed"],
    outputs="medium_extract"
)

text_preprocessing_node = node(
    apply_preprocessing,
    name="article_preprocessing",
    inputs=["medium_extract", "params:threshold_len", "params:test_size"],
    outputs=["training", "test"]
)

model_training_node = node(
    train_tf_idf,
    name="fit_model",
    inputs="training",
    outputs="model_tf_idf"
)

model_prediction_node = node(
    prediction_tf_idf,
    name="model_inference",
    inputs=["model_tf_idf", "test", "params:n_topics"],
    outputs="test_predictions"
)

