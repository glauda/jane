import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List


def train_tf_idf(df_train: pd.DataFrame) -> TfidfVectorizer:
    """ Train the TF IDF model

    Args:
        df: cleaned dataframe

    Returns:
        trained TF-IDF model
    """

    vectorizer = TfidfVectorizer()
    TF_IDF = vectorizer.fit(df_train.text)

    return TF_IDF


def prediction_tf_idf(tf_idf: TfidfVectorizer, df_test: pd.DataFrame, n_topics: int) -> Dict[int, List[str]]:
    """Predict the topics of an article dataframe

    Args:
        tf_idf: model used to predict the most important words (ie topics in that context)
        df_test: article dataframe to make predictions on
        n_topics: The number of token to keep as topics

    Returns:
        The id and the topics of the articles
    """


    # =========== Just for test with old format =============== 
    # df_test["id"] = df_test.index
    # ========================================================= 
    
    # Apply TF IDF to test data
    df_enc = pd.DataFrame(tf_idf.transform(df_test.text).toarray())
    
    # Create column id / label mapping
    ax_labels = tf_idf.get_feature_names_out()
    n_labels = ax_labels.shape[0]
    d_labels = {i: ax_labels[i] for i in range(n_labels)}

    # Init topic selection loop
    d_topics = {}
    df_test = df_test.reset_index(drop=True)

    for index, row in df_enc.iterrows():
        # Get article ID
        article_id = df_test.loc[index, "id"]
        
        # Get most important topics in article
        l_topics = row.sort_values(ascending=False)[:n_topics].index.tolist()
        l_topics = [d_labels[elt] for elt in l_topics]

        # update result
        d_topics[article_id] = l_topics

    return d_topics
