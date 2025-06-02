import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('ridge_model.pkl', 'rb') as f:
            models['Ridge Regression'] = pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки Ridge Regression: {str(e)}")

    try:
        # Для Gradient Boosting попробуем альтернативный подход
        from sklearn.ensemble import GradientBoostingRegressor
        models['Gradient Boosting'] = GradientBoostingRegressor()
        # Здесь можно загрузить только параметры или переобучить модель
    except Exception as e:
        st.error(f"Ошибка загрузки Gradient Boosting: {str(e)}")

    try:
        models['CatBoost'] = CatBoostRegressor().load_model('catboost_model.cbm')
    except Exception as e:
        st.error(f"Ошибка загрузки CatBoost: {str(e)}")

    try:
        with open('rf_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки Random Forest: {str(e)}")

    try:
        with open('stacking_model.pkl', 'rb') as f:
            models['Stacking'] = pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки Stacking: {str(e)}")

    try:
        models['Neural Network'] = keras.models.load_model('nn_model.h5')
    except Exception as e:
        st.error(f"Ошибка загрузки Neural Network: {str(e)}")

    return models


models = load_models()


# Загрузка данных для визуализаций
@st.cache_data
def load_data():
    return pd.read_csv('./data/processed_moldova_cars.csv')


data = load_data()

# Настройка страниц
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:",
                        ["О разработчике", "О данных", "Визуализации", "Предсказание"])

if page == "О разработчике":
    st.title("Информация о разработчике")
    col1, col2 = st.columns(2)

    with col1:
        st.header("ФИО: Данилов Егор Дмитриевич")
        st.subheader("Номер учебной группы: ФИТ-232")
        st.write("Тема РГР: Прогнозирование цен на автомобили в Молдове")

    with col2:
        image = Image.open('developer_photo.jpg')
        st.image(image, caption='Фото разработчика', width=200)

elif page == "О данных":
    st.title("Информация о наборе данных")

    st.header("Описание предметной области")
    st.write("""
    Данный датасет содержит информацию о подержанных автомобилях, продающихся в Молдове.
    Он включает различные характеристики автомобилей, такие как марка, модель, год выпуска,
    пробег, объем двигателя и другие параметры, которые влияют на конечную цену автомобиля.
    """)

    st.header("Описание признаков")
    st.write("""
    - Make: Марка автомобиля (например, BMW, Audi, Volkswagen)
    - Model: Конкретная модель автомобиля
    - Year: Год выпуска автомобиля
    - Style: Тип кузова (седан, универсал, внедорожник и т.д.)
    - Distance: Пробег автомобиля в километрах
    - Engine_capacity(cm3): Объем двигателя в кубических сантиметрах
    - Fuel_type: Тип топлива (бензин, дизель, гибрид и т.д.)
    - Price(euro): Цена автомобиля в евро (целевая переменная)
    - IsTransmissionAuto: Флаг автоматической коробки передач (1 - автоматическая, 0 - механическая)
    """)

    st.header("Предобработка данных")
    st.write("""
    - Обработка пропущенных значений
    - Удаление выбросов
    - Кодирование категориальных переменных
    - Нормализация числовых признаков
    """)

elif page == "Визуализации":
    st.title("Визуализации зависимостей в данных")

    st.header("1. Распределение цен")
    fig1, ax1 = plt.subplots()
    sns.histplot(data['Price(euro)'], bins=30, kde=True, ax=ax1)
    ax1.set_xlabel("Цена (евро)")
    ax1.set_ylabel("Количество")
    st.pyplot(fig1)

    st.header("2. Зависимость цены от года выпуска")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Year', y='Price(euro)', data=data, ax=ax2)
    ax2.set_xlabel("Год выпуска")
    ax2.set_ylabel("Цена (евро)")
    st.pyplot(fig2)

    st.header("3. Зависимость цены от пробега")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='Distance', y='Price(euro)', data=data, ax=ax3)
    ax3.set_xlabel("Пробег (км)")
    ax3.set_ylabel("Цена (евро)")
    st.pyplot(fig3)

    st.header("4. Средняя цена по маркам (топ-10)")
    top_makes = data['Make'].value_counts().nlargest(10).index
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Make', y='Price(euro)',
                data=data[data['Make'].isin(top_makes)],
                estimator=np.mean, ax=ax4)
    ax4.set_xlabel("Марка")
    ax4.set_ylabel("Средняя цена (евро)")
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

elif page == "Предсказание":
    st.title("Предсказание цены автомобиля")

    model_choice = st.selectbox("Выберите модель для предсказания:",
                                list(models.keys()))

    st.header("Введите параметры автомобиля")

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Марка", sorted(data['Make'].unique()))
        model = st.text_input("Модель", "A4")
        year = st.slider("Год выпуска", 1990, 2023, 2015)
        style = st.selectbox("Тип кузова", sorted(data['Style'].unique()))

    with col2:
        distance = st.number_input("Пробег (км)", min_value=0, max_value=500000, value=100000)
        engine_capacity = st.number_input("Объем двигателя (см3)", min_value=800, max_value=6000, value=2000)
        fuel_type = st.selectbox("Тип топлива", sorted(data['Fuel_type'].unique()))
        is_auto = st.radio("Коробка передач", ["Автоматическая", "Механическая"])

    is_transmission_auto = 1 if is_auto == "Автоматическая" else 0

    input_data = pd.DataFrame([[make, model, year, style, distance, engine_capacity, fuel_type, is_transmission_auto]],
                              columns=['Make', 'Model', 'Year', 'Style', 'Distance', 'Engine_capacity(cm3)',
                                       'Fuel_type', 'IsTransmissionAuto'])

    if st.button("Предсказать цену"):
        model = models[model_choice]

        if model_choice == 'Neural Network':
            # Для нейронной сети нужна специальная обработка
            preprocessor = models['Ridge Regression'].named_steps['preprocessor']
            processed_data = preprocessor.transform(input_data)
            if hasattr(processed_data, 'toarray'):
                processed_data = processed_data.toarray()
            prediction = model.predict(processed_data)[0][0]
        elif model_choice == 'CatBoost':
            prediction = model.predict(input_data)[0]
        else:
            prediction = model.predict(input_data)[0]

        st.success(f"Предсказанная цена автомобиля: {prediction:.2f} евро")

        # Покажем реальные цены похожих автомобилей для сравнения
        st.subheader("Реальные цены похожих автомобилей")
        similar_cars = data[
            (data['Make'] == make) &
            (data['Model'].str.contains(model, case=False)) &
            (data['Year'] >= year - 2) &
            (data['Year'] <= year + 2)
            ]

        if not similar_cars.empty:
            st.dataframe(similar_cars[['Make', 'Model', 'Year', 'Distance', 'Price(euro)']].head(10))
        else:
            st.warning("В данных нет похожих автомобилей для сравнения")