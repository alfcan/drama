import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from metrics import Metrics
from utils import get_label
from argparse import ArgumentParser

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df = df.replace('?', np.nan)
    df = df.dropna()
    df['age'] = df['age'].astype("category")
    df = pd.get_dummies(df)
    return df


def analysis(
    test: pd.DataFrame,
    predicted_label,
    true_label,
    positive_value,
):
    symptoms = pd.DataFrame()
    binary_variables = [
        c
        for c in test.drop(columns=[true_label, predicted_label]).columns
        if test[c].nunique() == 2
    ]
    metrics = Metrics(test, predicted_label, true_label, positive_value)


    sp = []
    eo = []
    aod = []
    variable = []

    for i in binary_variables:
        sp.append(metrics.statistical_parity({i: 0}))
        eo.append(metrics.equal_accuracy({i: 0}))
        aod.append(metrics.average_odds({i: 0}))
        variable.append(i.split("_")[0])

    symptoms["variable"] = variable
   
    symptoms["statistical_parity"] = sp
    symptoms["equal_opportunity"] = eo
    symptoms["average_odds"] = aod
    symptoms["variable"] = variable
    return symptoms


if __name__ == "__main__":

    models = ["rf", "xgb"]

    for data_name in os.listdir("dataset"):
        for model_name in models:
            print(f"Processing {data_name} with model {model_name}")
            label, pos_val = get_label(data_name)
            data = pd.read_csv(f"dataset/{data_name}")
            data = preprocess_data(data)
            n_folds = 10

            kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            results = pd.DataFrame()

            for i, (train_index, test_index) in enumerate(kfolds.split(data)):

                train = data.iloc[train_index]
                test = data.iloc[test_index]
                if model_name == "logreg":
                    model = LogisticRegression()
                if model_name == "rf":
                    model = RandomForestClassifier()
                if model_name == "mlp":
                    model = MLPClassifier()
                if model_name == "xgb":
                    model = XGBClassifier()
                model = model.fit(
                    train.drop(columns=label, axis=1).values, train[label].values.ravel()
                )

                test["prediction"] = model.predict(test.drop(columns=label))

                symptoms = analysis(test, "prediction", label, pos_val)
                symptoms["fold"] = i
                results = pd.concat([results, symptoms])
            name = os.path.basename(data_name)
            name = name.split(".")[0]
            os.makedirs("results", exist_ok=True)
            results.to_csv(
                os.path.join(
                    "results", f"{name}_{model_name}.csv"
                )
            )
