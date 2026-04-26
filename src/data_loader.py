import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(dataset_path="data/Dataset"):

    train = pd.read_csv(f"{dataset_path}/train.csv")
    test = pd.read_csv(f"{dataset_path}/test.csv")

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return X_train, X_test, y_train, y_test, encoder