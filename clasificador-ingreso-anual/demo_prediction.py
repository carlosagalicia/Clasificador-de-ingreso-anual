import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import tensorflow as tf

preprocessor = joblib.load("modelos/preprocessor.pkl")
model = tf.keras.models.load_model("modelos/modelo_mejorado.keras")

THRESHOLD = 0.37

workclass_options = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"
]

education_options = [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
    "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
]

marital_status_options = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]

occupation_options = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces", "Unknown"
]

relationship_options = [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
]

race_options = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
]

gender_options = [
    "Male", "Female"
]

country_options = [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
    "Germany", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
    "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
    "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
    "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
    "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
    "Hong", "Holand-Netherlands", "Unknown"
]

root = tk.Tk()
root.geometry("580x400")

main_frame = ttk.Frame(root, padding=5)
main_frame.pack(fill="both")

def var_input(row, text, default_value):
    ttk.Label(main_frame, text=text).grid(row=row, column=0, sticky="w")
    entry = ttk.Entry(main_frame)
    entry.grid(row=row, column=1, sticky="ew")
    entry.insert(0, default_value)
    return entry

def var_dropDown(row, text, options, default_value):
    ttk.Label(main_frame, text=text).grid(row=row, column=0, sticky="w")
    combo = ttk.Combobox(main_frame, values=options, state="readonly")
    combo.grid(row=row, column=1, sticky="ew")
    combo.set(default_value)
    return combo

age = var_input(0, "Age", "39")
workclass = var_dropDown(1, "Workclass", workclass_options, "Private")
education = var_dropDown(2, "Education", education_options, "Bachelors")
marital = var_dropDown(3, "Marital Status", marital_status_options, "Never-married")
occupation = var_dropDown(4, "Occupation", occupation_options, "Adm-clerical")
relationship = var_dropDown(5, "Relationship", relationship_options, "Not-in-family")
race = var_dropDown(6, "Race", race_options, "White")
gender = var_dropDown(7, "Gender", gender_options, "Male")
capital_gain = var_input(8, "Capital Gain", "2174")
capital_loss = var_input(9, "Capital Loss", "0")
hours = var_input(10, "Hours per Week", "40")
country = var_dropDown(11, "Native Country", country_options, "United-States")

result_label = ttk.Label(
    main_frame,
    text="Predicción",
    font=("Arial", 11, "bold"),
    justify="left"
)
result_label.grid(row=13, column=0, columnspan=2, sticky="w", pady=5)

def predict_income():
    try:
        case = pd.DataFrame([{
            "age": int(age.get()),
            "workclass": workclass.get(),
            "education": education.get(),
            "marital-status": marital.get(),
            "occupation": occupation.get(),
            "relationship": relationship.get(),
            "race": race.get(),
            "gender": gender.get(),
            "capital-gain": int(capital_gain.get()),
            "capital-loss": int(capital_loss.get()),
            "hours-per-week": int(hours.get()),
            "native-country": country.get()
        }])

        X = preprocessor.transform(case)
        prob = float(model.predict(X, verbose=0).flatten()[0])
        pred = ">50K" if prob >= THRESHOLD else "<=50K"

        texto = f"Probabilidad de >50K: {prob * 100:.2f}%\nPredicción: {pred}"
        result_label.config(text=texto)

    except Exception as e:
        messagebox.showerror("Error", f"{e}")

predict_button = ttk.Button(main_frame, text="Predecir", command=predict_income)
predict_button.grid(row=12, column=0, columnspan=2, pady=20)

main_frame.columnconfigure(1, weight=1)

root.mainloop()