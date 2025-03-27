from preprocessing.preprocess import *
from preprocessing.embeddings import *
from preprocessing.data_loader import *
from models.trainer import *
from utils.data_model import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return DataBundle(X, df)


def perform_modelling(data: DataBundle, df: pd.DataFrame, name, model_name):
    model_predict(data, df, name, model_name)


if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype(
        'U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name, 'RandomForest')
        perform_modelling(data, group_df, name, 'AdaBoost')
        perform_modelling(data, group_df, name, 'SGD')
