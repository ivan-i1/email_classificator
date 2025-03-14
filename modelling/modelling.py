from model.randomforest import RandomForest
from model.adaboost import AdaBoost

def model_predict(data, df, name, model_name):
    results = []
    print("Model: ", model_name)

    # Dictionary to map model names to their respective classes
    model_switch = {
        "RandomForest": RandomForest,
        "AdaBoost": AdaBoost
    }

    # Get the model class from the dictionary
    model_class = model_switch.get(model_name)

    if model_class:
        model = model_class(model_name, data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)
    else:
        print(f"Model {model_name} not recognized.")

def model_evaluate(model, data):
    model.print_results(data)