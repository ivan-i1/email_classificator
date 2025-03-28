from preprocessing.preprocess import *
from preprocessing.embeddings import *
from preprocessing.data_loader import *
from preprocessing.data_splitter import *
from models.trainer import *
from utils.data_model import *
from preprocessing.label_generator import generate_combined_labels
from preprocessing.data_splitter import split_data
from models.model_selector import *
from architecture.chained_classifier import ChainedClassifier
from sklearn.metrics import classification_report
from evaluation.dependency_score import dependency_aware_accuracy
import random
import argparse
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    df = get_input_data()
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return DataBundle(X, df)


def perform_modelling(data: DataBundle, df: pd.DataFrame, name, model_name):
    model_predict(data, df, name, model_name)


def run_chained(group_df, X, base_model):
    model = ChainedClassifier(base_model)
    model.fit(X, group_df)
    preds = model.predict_all(X)

    print("\n📊 Classification Report (Chained)")
    for level in ["type_2", "type_2_3", "type_2_3_4"]:
        print(f"\nLevel: {level}")
        print(classification_report(
            group_df[level], preds[level], zero_division=0))

    print("\n🔁 Dependency-Aware Accuracy:")
    for level in ["type_2", "type_3", "type_4"]:
        score = dependency_aware_accuracy(group_df, preds, level)
        print(f"{level}: {score:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run Models and Generate Results")
    parser.add_argument(
        "--mode", choices=["chained", "hierarchical"], required=True, help="Architecture type")
    parser.add_argument(
        "--model", choices=["rf", "ada", "sgd"], required=True, help="Classifier to use")
    args = parser.parse_args()

    df = load_data()
    df = preprocess_data(df)

    df = generate_combined_labels(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype(
        'U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"\n Group: {name} | Samples: {len(group_df)}")

        X, group_df = get_embeddings(group_df)

        label_col = "type_2_3_4" if args.mode == "chained" else "type_2"
        X_train, X_test, y_train, y_test, y, classes, embeddings = split_data(
            X, group_df, label_col)

        if X_train is None:
            print(
                f"Skipping group {name} due to insufficient class samples.")
            continue

        data_bundle = DataBundle(X_train, X_test, y_train, y_test,
                                 y=label_col, classes=classes, embeddings=X)
        model = get_model(args.model, embeddings, label_col).mdl
        run_chained(group_df, X, model)

        '''
        if args.mode == "chained":
            run_chained(group_df, X, model)
        else:
            run_hierarchical(group_df, X, model)
        '''
