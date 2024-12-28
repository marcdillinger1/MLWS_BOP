import xgboost as xgb
import numpy as np


def load_model_and_predict(input_data, model_file, valid_genres):
    """
    Lädt das Modell und führt eine Vorhersage basierend auf den Eingabedaten durch.

    Args:
        input_data (dict): Eingabedaten mit Runtime, Budget, Director, Lead Actor und Genre.
        model_file (str): Pfad zur gespeicherten Modelldatei.
        valid_genres (list): Liste gültiger Genres.

    Returns:
        tuple: Vorhergesagte Umsätze, untere und obere Schranken des Vorhersagebereichs.
    """
    # Lade das Modell
    model = xgb.Booster()
    model.load_model(model_file)

    # Debugging: Originale Eingabedaten ausgeben
    print("[DEBUG] Originale Eingabedaten:", input_data)

    # Verarbeitung der Eingabedaten
    runtime = float(input_data['Runtime'])
    budget = float(input_data['Budget'])
    director = hash(input_data['Director'].lower()) % 10000
    lead_actor = hash(input_data['Lead Actor'].lower()) % 10000
    genre = input_data['Genre'].lower()

    # Prüfen, ob das Genre gültig ist
    if genre not in valid_genres:
        raise ValueError(f"Ungültiges Genre '{genre}'. Gültige Genres sind: {valid_genres}")

    genre_index = valid_genres.index(genre)

    # Finale Feature-Matrix
    features = np.array([[runtime, budget, director, lead_actor, genre_index]])
    print("[DEBUG] Finale Features:", features)

    # XGBoost DMatrix erstellen
    dmatrix = xgb.DMatrix(features)

    # Vorhersage berechnen
    log_pred = model.predict(dmatrix)
    pred = np.expm1(log_pred)  # Rücktransformation
    lower_bound = np.expm1(log_pred - 0.1)  # Beispiel: Unsicherheitsbereich
    upper_bound = np.expm1(log_pred + 0.1)

    # Debugging: Ausgaben
    print("[DEBUG] Vorhersage (log-transformiert):", log_pred)
    print("[DEBUG] Rücktransformierte Vorhersage:", pred)
    print("[DEBUG] Vorhersagebereich:", lower_bound, "bis", upper_bound)

    return pred[0], lower_bound[0], upper_bound[0]


def main():
    """
    Hauptfunktion für die Vorhersage.
    Nimmt Benutzereingaben entgegen, verarbeitet sie und gibt die Vorhersage aus.
    """
    # Gültige Genres definieren (ersetzen durch tatsächliche Genres im Datensatz)
    valid_genres = ["action", "comedy", "drama", "thriller", "horror", "romance"]

    # Modell-Dateipfad (anpassen)
    model_file = "xgboost_model_with_cv.json"

    # Eingaben des Benutzers
    print("Bitte geben Sie die folgenden Informationen ein:")
    genre = input("Genre: ").strip().lower()
    runtime = input("Laufzeit (in Minuten): ").strip()
    budget = input("Budget (in USD): ").strip()
    director = input("Regisseur: ").strip()
    lead_actor = input("Hauptschauspieler: ").strip()

    # Validierung und Erstellung der Eingabedaten
    input_data = {
        "Runtime": runtime,
        "Budget": budget,
        "Director": director,
        "Lead Actor": lead_actor,
        "Genre": genre
    }

    try:
        # Vorhersage ausführen
        predicted_value, lower_bound, upper_bound = load_model_and_predict(
            input_data, model_file, valid_genres
        )

        # Ausgabe der Ergebnisse
        print(f"\nVorhergesagter Box-Office Umsatz: ${predicted_value:.2f}")
        print(f"Vorhersagebereich: ${lower_bound:.2f} bis ${upper_bound:.2f}")

    except ValueError as e:
        print(f"Fehler: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    main()
