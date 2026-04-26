from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train, X_test, y_test):

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results[name] = acc
        print(f"{name}: {acc:.4f}")

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print("\nBest Model:", best_model_name)

    return results, best_model