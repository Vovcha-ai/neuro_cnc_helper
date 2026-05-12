import streamlit as st
import importlib


def update_model():
    """Запускает генерацию датасета и переобучение модели"""

    with st.spinner("Генерация нового датасета..."):
        # Импортируем и запускаем функцию generate_dataset
        from generate_dataset import generate_dataset, save_dataset
        df = generate_dataset(samples_per_class=600)
        save_dataset(df)

    with st.spinner("Переобучение нейронной сети..."):
        # Импортируем и запускаем функцию train_model
        from train_nn import train_model
        train_model()

    return True