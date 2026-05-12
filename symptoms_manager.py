import json
import os

SYMPTOMS_FILE = "data/symptoms.json"


def init_symptoms():
    """Создает файл с симптомами, если его нет"""
    if not os.path.exists(SYMPTOMS_FILE):
        default_symptoms = {
            "thermal": {
                "name": "Термальные признаки",
                "symptoms": [
                    {"key": "temp_high", "name": "Перегрев"},
                    {"key": "temp_uneven", "name": "Неравномерный нагрев"},
                    {"key": "burning_smell", "name": "Запах гари"},
                    {"key": "cooling_failure", "name": "Не работает охлаждение"}
                ]
            },
            "mechanical": {
                "name": "Механические признаки",
                "symptoms": [
                    {"key": "vibration", "name": "Вибрация"},
                    {"key": "strange_noise", "name": "Странный шум"},
                    {"key": "loose_parts", "name": "Ослаблены крепления"},
                    {"key": "belt_squeak", "name": "Свистит ремень"},
                    {"key": "bearing_grind", "name": "Скрежет подшипника"},
                    {"key": "lubrication_issue", "name": "Проблема со смазкой"}
                ]
            },
            "electrical": {
                "name": "Электрические признаки",
                "symptoms": [
                    {"key": "error_code", "name": "Код ошибки"},
                    {"key": "sparks", "name": "Искрит"},
                    {"key": "current_spike", "name": "Скачки тока"},
                    {"key": "voltage_drop", "name": "Падает напряжение"},
                    {"key": "electric_arc", "name": "Электрическая дуга"}
                ]
            },
            "visual": {
                "name": "Визуальные и прочие признаки",
                "symptoms": [
                    {"key": "smoke", "name": "Дым"},
                    {"key": "oil_leak", "name": "Течь масла"},
                    {"key": "rust", "name": "Коррозия"},
                    {"key": "power_loss", "name": "Падение мощности"},
                    {"key": "speed_unstable", "name": "Нестабильная скорость"}
                ]
            }
        }
        save_symptoms(default_symptoms)
    return load_symptoms()


def load_symptoms():
    with open(SYMPTOMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_symptoms(symptoms_data):
    with open(SYMPTOMS_FILE, "w", encoding="utf-8") as f:
        json.dump(symptoms_data, f, indent=4, ensure_ascii=False)


def get_all_symptoms_keys():
    """Возвращает список всех ключей симптомов"""
    symptoms = load_symptoms()
    all_keys = []
    for category in symptoms.values():
        for symptom in category["symptoms"]:
            all_keys.append(symptom["key"])
    return all_keys


def get_all_symptoms_with_names():
    """Возвращает список всех симптомов с ключами и названиями"""
    symptoms = load_symptoms()
    result = []
    for category in symptoms.values():
        for symptom in category["symptoms"]:
            result.append({"key": symptom["key"], "name": symptom["name"], "category": category["name"]})
    return result


def get_symptom_name(key):
    """Возвращает русское название симптома по ключу"""
    symptoms = load_symptoms()
    for category in symptoms.values():
        for symptom in category["symptoms"]:
            if symptom["key"] == key:
                return symptom["name"]
    return key


def add_symptom_to_all_faults(symptom_key, symptom_name, category_key):
    """Добавляет новый симптом во все существующие поломки со значением 0"""
    with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)

    for fault_id in kb:
        kb[fault_id]["symptoms"][symptom_key] = 0

    with open("data/knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)

    return True


def remove_symptom_from_all_faults(symptom_key):
    """Удаляет симптом из всех поломок"""
    with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)

    for fault_id in kb:
        if symptom_key in kb[fault_id]["symptoms"]:
            del kb[fault_id]["symptoms"][symptom_key]

    with open("data/knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)

    return True


def add_symptom(category_key, symptom_key, symptom_name):
    """Добавляет новый симптом в категорию"""
    symptoms = load_symptoms()

    # Проверяем, существует ли уже такой ключ
    for category in symptoms.values():
        for s in category["symptoms"]:
            if s["key"] == symptom_key:
                return False, "Симптом с таким ключом уже существует"

    # Добавляем симптом в категорию
    if category_key in symptoms:
        symptoms[category_key]["symptoms"].append({"key": symptom_key, "name": symptom_name})
        save_symptoms(symptoms)
        return True, "Симптом добавлен"
    return False, "Категория не найдена"


def remove_symptom(category_key, symptom_key):
    """Удаляет симптом из категории"""
    symptoms = load_symptoms()

    if category_key in symptoms:
        for i, s in enumerate(symptoms[category_key]["symptoms"]):
            if s["key"] == symptom_key:
                symptoms[category_key]["symptoms"].pop(i)
                save_symptoms(symptoms)
                return True, "Симптом удален"
    return False, "Симптом не найден"


def get_symptoms_by_category():
    """Возвращает симптомы сгруппированные по категориям для интерфейса"""
    symptoms = load_symptoms()
    result = {}
    for cat_key, cat_data in symptoms.items():
        result[cat_key] = {
            "name": cat_data["name"],
            "symptoms": {s["key"]: s["name"] for s in cat_data["symptoms"]}
        }
    return result