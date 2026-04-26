import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def extract_features(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    energy = np.sum(signal**2)
    change = np.mean(np.abs(np.diff(signal)))

    fft = np.abs(np.fft.fft(signal))
    freq_power = np.mean(fft[:50])

    return [mean_val, std_val, energy, change, freq_power]


def build_decision_dataset(X_test, model):
    X_decision = []

    for i in range(len(X_test)):
        signal = X_test[i]
        features = extract_features(signal)
        activity = model.predict(signal.reshape(1, -1))[0]
        X_decision.append(features + [activity])

    return np.array(X_decision)


def train_decision_model(X_decision):
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_decision = kmeans.fit_predict(X_decision)

    model = RandomForestClassifier()
    model.fit(X_decision, y_decision)

    return model


def simulate_system(X_test, model, decision_model, encoder):

    decision_map = {
        0: "Safe",
        1: "Moderate Risk",
        2: "Critical Alert"
    }

    print("\n==== LIVE SYSTEM ====\n")

    indices = np.random.choice(len(X_test), 30, replace=False)

    decisions = []

    for i in indices:
        sample = X_test[i]

        features = extract_features(sample)
        activity = model.predict(sample.reshape(1, -1))[0]

        activity_name = encoder.inverse_transform([activity])[0]

        final_input = np.array(features + [activity]).reshape(1, -1)
        decision = decision_model.predict(final_input)[0]

        decisions.append(decision)

        print(f"{activity_name} → {decision_map[decision]}")

    return decisions