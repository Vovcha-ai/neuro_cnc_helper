import pandas as pd
import joblib
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from symptoms_manager import get_all_symptoms_keys, get_symptom_name


def train_model():
    print("Загрузка данных...")
    df = pd.read_csv("data/faults_dataset.csv")

    X = df.drop("fault_type", axis=1)
    y = df["fault_type"]

    print(f"Образцов: {len(df)}, Признаков: {len(X.columns)}, Классов: {len(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Обучение нейронной сети...")
    model = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        max_iter=3000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    model.fit(X_train_scaled, y_train)

    test_acc = model.score(X_test_scaled, y_test)
    print(f"Точность на тесте: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    joblib.dump(model, "model/simple_nn.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    # Сохраняем список симптомов для интерфейса
    symptoms_info = []
    for key in get_all_symptoms_keys():
        symptoms_info.append({"key": key, "name": get_symptom_name(key)})

    with open("model/symptoms_info.json", "w", encoding="utf-8") as f:
        json.dump(symptoms_info, f, indent=4, ensure_ascii=False)