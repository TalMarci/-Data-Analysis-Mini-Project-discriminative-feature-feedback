import pandas as pd


def get_nursery_data():
    nursery_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data")
    nursery_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health",
                     "reception_status"]
    nursery_data.columns = nursery_names
    X, y = nursery_data.iloc[:, :-1], nursery_data.iloc[:, [-1]]
    # prepare X
    for feature in X.columns:
        dummy = pd.get_dummies(X[feature])
        X = pd.concat((X, dummy), axis=1)
        X = X.drop([feature], axis=1)

    # prepare Y
    y = y.replace(["not_recom", "priority", "spec_prior", "very_recom", "recommend"], [1, 2, 3, 4, 5])

    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def get_zoo_data():
    zoo_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data")
    zoo_names = ["animal name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed",
                 "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
    zoo_data.columns = zoo_names
    X, y = zoo_data.iloc[:, 1:-1], zoo_data.iloc[:, [-1]]
    dummy = pd.get_dummies(X["legs"])
    X = pd.concat((X, dummy), axis=1)
    X = X.drop(["legs"], axis=1)
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def get_mushrooms_data():
    mushrooms_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")
    mushrooms_names = ["label", "cap-Shape", "cap_surface", "cap_color", "bruises?", "odor", "gill-attachment",
                       "gill-spacing",
                       "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                       "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
                       "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population",
                       "habitat"]
    mushrooms_data.columns = mushrooms_names
    X, y = mushrooms_data.iloc[:, 1:], mushrooms_data.iloc[:, [0]]
    # prepare X
    for feature in X.columns:
        dummy = pd.get_dummies(X[feature])
        X = pd.concat((X, dummy), axis=1)
        X = X.drop([feature], axis=1)

    # prepare Y
    y = y.replace(["e", "p"], [1, 2])

    X = X.to_numpy()
    y = y.to_numpy()
    return X, y

