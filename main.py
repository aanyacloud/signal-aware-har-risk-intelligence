from src.data_loader import load_data
from src.models import train_models
from src.feature_selection import select_features
from src.preprocessing import scale_data
from src.decision_system import build_decision_dataset, train_decision_model, simulate_system
from src.evaluation import (
    plot_model_comparison,
    plot_confusion_matrix,
    print_classification_report,
    plot_decision_distribution
)

def main():

    print("Loading data...")
    X_train, X_test, y_train, y_test, encoder = load_data()

    print("\nFeature selection...")
    X_train_red, X_test_red = select_features(X_train, X_test, y_train)

    print("\nScaling...")
    X_train_scaled, X_test_scaled = scale_data(X_train_red, X_test_red)

    print("\nTraining models...")
    results, best_model = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # 📊 Evaluation
    plot_model_comparison(results)

    y_pred = best_model.predict(X_test_scaled)
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

    print("\nBuilding decision dataset...")
    X_decision = build_decision_dataset(X_test_scaled, best_model)

    print("\nTraining decision model...")
    decision_model = train_decision_model(X_decision)

    print("\nRunning simulation...")
    decisions = simulate_system(X_test_scaled, best_model, decision_model, encoder)

    plot_decision_distribution(decisions)


if __name__ == "__main__":
    main()