import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# Lade das Modell und die Daten
def load_model_and_data(model_file, reference_file):
    model = xgb.Booster()
    model.load_model(model_file)

    reference_data = pd.read_csv(reference_file)

    return model, reference_data


# Eingabedaten verarbeiten und auf das Modell vorbereiten
def prepare_input_data(input_data, reference_data, target_column):
    features = reference_data.drop(columns=[target_column])

    # Sicherstellen, dass alle Eingabedaten im gleichen Format sind (kategorische Variablen)
    input_row = {col: input_data.get(col, np.nan) for col in features.columns}

    # Optional: Bei leeren Eingabewerten mit Mittelwert füllen (anstatt NaN)
    input_row = {k: (v if pd.notna(v) else features[k].mean()) for k, v in input_row.items()}

    input_df = pd.DataFrame([input_row])

    # Optional: Kategorische Features als kategorische Variablen behandeln
    input_df = pd.get_dummies(input_df)

    # Sicherstellen, dass alle Features mit dem trainierten Modell übereinstimmen
    missing_cols = set(features.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Füge fehlende Spalten hinzu (mit 0 initialisiert)

    # Sicherstellen, dass keine unnötigen Spalten vorhanden sind
    input_df = input_df[features.columns]

    # Skalierung der Eingabedaten wie im Trainingsprozess
    scaler = StandardScaler()
    input_df_scaled = scaler.fit_transform(input_df)

    return input_df_scaled


# Vorhersagefunktion
def predict(input_data, model_file, reference_file, target_column):
    model, reference_data = load_model_and_data(model_file, reference_file)
    prepared_data = prepare_input_data(input_data, reference_data, target_column)

    # Umwandeln in DMatrix für XGBoost
    dmatrix = xgb.DMatrix(prepared_data)

    # Vorhersage
    prediction = model.predict(dmatrix)

    # Rücktransformation der log-transformierten Vorhersage
    predicted_revenue = np.expm1(prediction[0])  # Rücktransformation von log1p

    # Umrechnung der Vorhersage in Millionen
    predicted_revenue_in_millions = predicted_revenue / 1_000_000

    # Berechne ±5% Spanne
    lower_bound = predicted_revenue_in_millions * 0.95
    upper_bound = predicted_revenue_in_millions * 1.05

    return predicted_revenue_in_millions, lower_bound, upper_bound


# Eingabeaufforderungen an den Benutzer
def get_user_input():
    print(
        "Bitte geben Sie die Werte für den Film ein. Felder, die nicht erforderlich sind, können leer gelassen werden.")

    print(
        "Verfügbare Genres zur Auswahl: Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western")
    genre = input("Geben Sie das Genre des Films ein (z.B. Action, Drama, Komödie - erforderlich): ").strip()

    # Sicherstellen, dass Genre als Pflichtfeld abgefragt wird
    while genre not in ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery",
                        "Romance", "Sci-Fi", "Thriller", "War", "Western"]:
        print("Ungültiges Genre! Wählen Sie bitte aus der Liste.")
        genre = input("Geben Sie das Genre des Films ein (z.B. Action, Drama, Komödie - erforderlich): ").strip()

    budget = input("Geben Sie das Budget des Films ein (erforderlich): ").strip()
    while not budget.isdigit():
        print("Bitte geben Sie eine gültige Zahl für das Budget ein.")
        budget = input("Geben Sie das Budget des Films ein (erforderlich): ").strip()

    running_time = input("Geben Sie die Laufzeit des Films in Minuten ein (erforderlich): ").strip()
    while not running_time.isdigit():
        print("Bitte geben Sie eine gültige Zahl für die Laufzeit ein.")
        running_time = input("Geben Sie die Laufzeit des Films in Minuten ein (erforderlich): ").strip()

    director = input("Geben Sie den Namen des Regisseurs ein (oder lassen Sie das Feld leer): ").strip()
    main_actor = input("Geben Sie den Namen des Hauptdarstellers ein (oder lassen Sie das Feld leer): ").strip()
    release_year = input(
        "Geben Sie das Erscheinungsjahr des Films ein (z.B. 2012 - oder lassen Sie das Feld leer): ").strip()

    # Eingabedaten zusammenstellen
    input_data = {
        "Budget": float(budget),
        "Running time": int(running_time),
        genre: 1,  # One-Hot-Encoding für Genre
        "Director_" + director: 1 if director else 0,
        "Main Actor_" + main_actor: 1 if main_actor else 0,
        "Release year_" + release_year: 1 if release_year else 0
    }

    return input_data


# Hauptprogramm
if __name__ == "__main__":
    model_file = "xgboost_model_with_cv.json"
    reference_file = "MLWS_BOP_Dataset_encoded.csv"
    target_column = "Box Office"  # Zielspalte im Datensatz

    user_input = get_user_input()
    prediction_result, lower_bound, upper_bound = predict(user_input, model_file, reference_file, target_column)

    print(f"Die geschätzten Einnahmen an den Kinokassen betragen: {prediction_result:.2f} Millionen")
    print(f"±5% Spanne: {lower_bound:.2f} Millionen bis {upper_bound:.2f} Millionen")
