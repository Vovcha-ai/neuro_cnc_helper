import streamlit as st
import joblib
import json
from kb_helper import KnowledgeBase
from database import save_prediction, get_statistics
from symptoms_manager import (
    get_all_symptoms_keys, get_symptom_name, get_symptoms_by_category,
    add_symptom, remove_symptom, add_symptom_to_all_faults,
    remove_symptom_from_all_faults, init_symptoms, load_symptoms
)

st.set_page_config(layout="wide")

# Инициализируем файл с симптомами
init_symptoms()


@st.cache_resource
def load_model():
    model = joblib.load("model/simple_nn.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler


@st.cache_resource
def load_kb():
    return KnowledgeBase()


def save_kb(kb_data):
    with open("data/knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb_data, f, indent=4, ensure_ascii=False)


def load_kb_raw():
    with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
        return json.load(f)


def generate_new_dataset(samples_per_class=600):
    from generate_dataset import generate_dataset, save_dataset
    df = generate_dataset(samples_per_class=samples_per_class)
    save_dataset(df)
    return True


def retrain_model():
    from train_nn import train_model
    train_model()
    return True


def update_model(samples_per_class=600):
    progress_bar = st.progress(0)
    status_text = st.empty()

    with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    num_classes = len(kb_data)
    total_samples = num_classes * samples_per_class

    status_text.text(f"Генерация датасета ({num_classes} классов, {total_samples} образцов)...")
    progress_bar.progress(30)

    try:
        generate_new_dataset(samples_per_class)
    except Exception as e:
        st.error(f"Ошибка генерации датасета: {e}")
        return False

    status_text.text("Переобучение нейронной сети...")
    progress_bar.progress(70)

    try:
        retrain_model()
    except Exception as e:
        st.error(f"Ошибка обучения модели: {e}")
        return False

    status_text.text("Обновление завершено!")
    progress_bar.progress(100)

    return True


model, scaler = load_model()
kb = load_kb()

with st.sidebar:
    st.header("Статистика")
    total, top_faults = get_statistics()
    st.metric("Всего диагнозов", total)
    if top_faults:
        st.write("**Частые поломки:**")
        for fault, count in top_faults[:5]:
            st.write(f"- {fault}: {count}")

st.title("Нейро-помощник техника")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Диагностика по симптомам", "Поиск информации об ошибке", "Управление базой знаний", "Управление симптомами"])

# Вкладка 1: Диагностика
with tab1:
    st.markdown("Укажите симптомы для определения поломки")

    # Динамическое создание чекбоксов на основе симптомов
    symptoms_by_category = get_symptoms_by_category()
    symptoms_values = {}

    for cat_key, cat_data in symptoms_by_category.items():
        with st.expander(cat_data["name"], expanded=True):
            cols = st.columns(3)
            for i, (symptom_key, symptom_name) in enumerate(cat_data["symptoms"].items()):
                with cols[i % 3]:
                    symptoms_values[symptom_key] = st.checkbox(symptom_name, key=f"diag_{symptom_key}")

    symptoms_list = [symptoms_values.get(s, False) for s in get_all_symptoms_keys()]
    input_data = [[1 if s else 0 for s in symptoms_list]]

    if st.button("Определить поломку", key="diagnose"):
        input_scaled = scaler.transform(input_data)
        pred_class = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        confidence = max(probs) * 100

        fault_info = kb.get_info(pred_class)

        symptoms_dict = {s: symptoms_values.get(s, False) for s in get_all_symptoms_keys()}
        save_prediction(int(pred_class), fault_info['name'], confidence, symptoms_dict)

        st.subheader("Результат")
        st.write(f"**Поломка:** {fault_info['name']}")
        st.write(f"**Рекомендация:** {fault_info['recommendation']}")
        st.write(f"**Уверенность:** {confidence:.1f}%")

# Вкладка 2: Поиск в интернете
with tab2:
    st.subheader("Поиск информации об ошибке в интернете")

    col1, col2 = st.columns([3, 1])
    with col1:
        error_text = st.text_input("Введите модель станка и код ошибки:",
                                   placeholder="Пример: HAAS VF-2 ошибка 123")
    with col2:
        search_engine = st.selectbox("Поисковая система", ["Google", "Yandex", "YouTube"])

    if st.button("Найти решение", key="search"):
        if error_text:
            query = error_text.replace(" ", "+")
            if search_engine == "Google":
                url = f"https://www.google.com/search?q={query}+станок+ошибка+ремонт"
            elif search_engine == "Yandex":
                url = f"https://yandex.ru/search/?text={query}+станок+ошибка+ремонт"
            else:
                url = f"https://www.youtube.com/results?search_query={query}+станок+ошибка"

            st.success(f"Поиск по запросу: {error_text}")
            st.markdown(f"[🔗 Открыть результаты поиска в {search_engine}]({url})", unsafe_allow_html=True)
        else:
            st.warning("Введите описание ошибки")

# Вкладка 3: Управление базой знаний
with tab3:
    st.subheader("Управление базой знаний")

    mode = st.radio("Выберите действие:", ["Просмотр поломок", "Добавить новую поломку", "Удалить поломку"])
    kb_data = load_kb_raw()
    symptoms_keys = get_all_symptoms_keys()

    if mode == "Просмотр поломок":
        st.write("Список всех поломок в базе знаний:")
        for fault_id, fault_info in kb_data.items():
            with st.expander(f"{fault_id}: {fault_info['name']}"):
                st.write(f"**Рекомендация:** {fault_info['recommendation']}")
                present_symptoms = []
                for symptom_key in symptoms_keys:
                    if fault_info['symptoms'].get(symptom_key, 0) == 1:
                        present_symptoms.append(get_symptom_name(symptom_key))
                if present_symptoms:
                    st.write("**Признаки поломки:**")
                    for s in present_symptoms:
                        st.write(f"{s}")
                else:
                    st.write("**Признаки поломки:** Нет симптомов")

    elif mode == "Добавить новую поломку":
        new_id = str(len(kb_data))
        st.markdown(f"**Добавление новой поломки (следующий ID: {new_id})**")

        new_name = st.text_input("Название поломки:")
        new_recommendation = st.text_area("Рекомендация по ремонту:")

        st.markdown("### Выберите симптомы для новой поломки:")

        symptom_cols = st.columns(4)
        new_symptoms = {}

        for i, symptom_key in enumerate(symptoms_keys):
            with symptom_cols[i % 4]:
                rus_name = get_symptom_name(symptom_key)
                new_symptoms[symptom_key] = 1 if st.checkbox(rus_name, key=f"new_{symptom_key}") else 0

        if st.button("Сохранить новую поломку и обновить модель"):
            if new_name and new_recommendation:
                kb_data[new_id] = {
                    "name": new_name,
                    "recommendation": new_recommendation,
                    "symptoms": new_symptoms
                }
                save_kb(kb_data)
                st.success(f"Поломка '{new_name}' добавлена под ID {new_id}")

                if update_model():
                    st.success("Модель успешно переобучена!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Ошибка при обновлении модели")
            else:
                st.warning("Заполните название и рекомендацию")

    elif mode == "Удалить поломку":
        fault_ids = list(kb_data.keys())
        selected_id = st.selectbox("Выберите ID поломки для удаления:", fault_ids)

        if selected_id:
            st.warning(f"Вы точно хотите удалить поломку: {kb_data[selected_id]['name']}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Да, удалить и обновить модель"):
                    del kb_data[selected_id]
                    new_kb = {}
                    for i, (old_id, value) in enumerate(kb_data.items()):
                        new_kb[str(i)] = value
                    save_kb(new_kb)
                    st.success("Поломка удалена. ID переиндексированы.")

                    if update_model():
                        st.success("Модель успешно переобучена!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("Ошибка при обновлении модели")
            with col2:
                if st.button("Отмена"):
                    st.info("Удаление отменено")

# Вкладка 4: Управление симптомами
with tab4:
    st.subheader("Управление симптомами")

    st.info("""
    **Важно:** При добавлении или удалении симптомов модель будет автоматически переобучена.
    Новые симптомы добавляются во все существующие поломки со значением 0.
    """)

    mode_symptom = st.radio("Выберите действие:",
                            ["Просмотр симптомов", "Добавить симптом", "Удалить симптом"])

    symptoms_data = load_symptoms()

    if mode_symptom == "Просмотр симптомов":
        for category_key, category_data in symptoms_data.items():
            st.write(f"### {category_data['name']}")
            for symptom in category_data["symptoms"]:
                st.write(f"- **{symptom['key']}** - {symptom['name']}")

    elif mode_symptom == "Добавить симптом":
        categories = list(symptoms_data.keys())
        category_names = {k: symptoms_data[k]["name"] for k in categories}
        selected_category = st.selectbox("Выберите категорию:", categories, format_func=lambda x: category_names[x])

        new_symptom_name = st.text_input("Название симптома (на русском):", placeholder="Пример: Высокое давление")
        new_symptom_key = st.text_input("Ключ симптома (на английском, без пробелов):",
                                        placeholder="Пример: high_pressure",
                                        help="Используется в коде, должен быть уникальным")

        st.warning("Новый симптом будет добавлен во ВСЕ существующие поломки со значением 0")

        if st.button("Добавить симптом и переобучить модель"):
            if new_symptom_name and new_symptom_key:
                success, message = add_symptom(selected_category, new_symptom_key, new_symptom_name)
                if success:
                    add_symptom_to_all_faults(new_symptom_key, new_symptom_name, selected_category)
                    st.success(
                        f"Симптом '{new_symptom_name}' добавлен в категорию '{category_names[selected_category]}'")
                    st.info("Обновляем модель...")

                    if update_model():
                        st.success("Модель успешно переобучена с новым симптомом!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("Ошибка при обновлении модели")
                else:
                    st.error(message)
            else:
                st.warning("Заполните название и ключ симптома")


    elif mode_symptom == "Удалить симптом":
        st.warning("Удаление симптома удалит его из ВСЕХ поломок. Модель нужно будет переобучить.")
        symptoms_data = load_symptoms()

        for category_key, category_data in symptoms_data.items():
            if category_data["symptoms"]:
                st.write(f"### {category_data['name']}")

                for symptom in category_data["symptoms"]:
                    st.write(f"- **{symptom['key']}** - {symptom['name']}")
                    if st.button(f"Удалить", key=f"del_{symptom['key']}"):
                        success, message = remove_symptom(category_key, symptom['key'])
                        if success:
                            remove_symptom_from_all_faults(symptom['key'])
                            st.success(f"Симптом '{symptom['name']}' удален")
                            st.info("Обновляем модель...")
                            if update_model():
                                st.success("Модель успешно переобучена!")
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error("Ошибка при обновлении модели")
                        else:
                            st.error(message)
                    st.write("")