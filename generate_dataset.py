import pandas as pd
import json
import random
from symptoms_manager import get_all_symptoms_keys

def generate_dataset(json_path="data/knowledge_base.json", samples_per_class=600):
    with open(json_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    symptoms = get_all_symptoms_keys()

    print(f"Классов: {len(kb)}, Симптомов: {len(symptoms)}")

    num_classes = len(kb)
    total_samples = num_classes * samples_per_class
    print(f"Всего образцов: {total_samples}")

    data = []

    for class_id, fault_info in kb.items():
        base_symptoms = fault_info.get("symptoms", {})

        for _ in range(samples_per_class):
            sample = {}
            for symptom in symptoms:
                base_value = base_symptoms.get(symptom, 0)
                if base_value == 1:
                    sample[symptom] = 1 if random.random() < 0.9 else 0
                else:
                    sample[symptom] = 1 if random.random() < 0.05 else 0
            sample["fault_type"] = int(class_id)
            data.append(sample)

    random.shuffle(data)
    df = pd.DataFrame(data)
    return df


def save_dataset(df, path="data/faults_dataset.csv"):
    df.to_csv(path, index=False)
    print(f"Датасет сохранен в {path}, размер: {df.shape}")


if __name__ == "__main__":
    df = generate_dataset(samples_per_class=600)
    save_dataset(df)

# import pandas as pd
# import json
# import random
#
#
# def generate_dataset(json_path="data/knowledge_base.json", samples_per_class=600):
#     with open(json_path, "r", encoding="utf-8") as f:
#         kb = json.load(f)
#
#     symptoms = [
#         'temp_high', 'vibration', 'strange_noise', 'smoke', 'error_code',
#         'power_loss', 'oil_leak', 'sparks', 'burning_smell', 'current_spike',
#         'voltage_drop', 'speed_unstable', 'temp_uneven', 'lubrication_issue',
#         'rust', 'loose_parts', 'belt_squeak', 'bearing_grind', 'electric_arc',
#         'cooling_failure'
#     ]
#
#     num_classes = len(kb)
#     total_samples = num_classes * samples_per_class
#
#     print(f"Классов: {num_classes}, образцов на класс: {samples_per_class}, всего: {total_samples}")
#
#     data = []
#
#     for class_id, fault_info in kb.items():
#         base_symptoms = fault_info.get("symptoms", {})
#
#         for _ in range(samples_per_class):
#             sample = {}
#             for symptom in symptoms:
#                 base_value = base_symptoms.get(symptom, 0)
#                 if base_value == 1:
#                     sample[symptom] = 1 if random.random() < 0.9 else 0
#                 else:
#                     sample[symptom] = 1 if random.random() < 0.05 else 0
#             sample["fault_type"] = int(class_id)
#             data.append(sample)
#
#     random.shuffle(data)
#     df = pd.DataFrame(data)
#     return df
#
#
# def save_dataset(df, path="data/faults_dataset.csv"):
#     df.to_csv(path, index=False)
#     print(f"Датасет сохранен в {path}")
#
#
# if __name__ == "__main__":
#     # Можно указать любое количество образцов на класс
#     SAMPLES_PER_CLASS = 600  # Измени это число, чтобы увеличить/уменьшить датасет
#     df = generate_dataset(samples_per_class=SAMPLES_PER_CLASS)
#     save_dataset(df)