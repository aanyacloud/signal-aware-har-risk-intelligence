from sklearn.ensemble import RandomForestClassifier
import numpy as np

def select_features(X_train, X_test, y_train, top_k=200):

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-top_k:]

    X_train_red = X_train.iloc[:, indices]
    X_test_red = X_test.iloc[:, indices]

    print("Reduced shape:", X_train_red.shape)

    return X_train_red, X_test_red