import pandas as pd

def apply_one_hot_encoding(file_path, encoded_file_path):
    # CSV-Datei einlesen mit dem korrekten Trennzeichen
    df = pd.read_csv(file_path, sep=';')  # Trennzeichen ';' verwenden, falls Semikolon das Trennzeichen ist

    # Bereinigen der Spaltennamen (Entfernen von führenden/nachfolgenden Leerzeichen)
    df.columns = df.columns.str.strip()

    # Überprüfe die Spaltennamen nach Bereinigung
    print("Bereinigte Spalten im DataFrame:", df.columns)

    # 'Movie Title' entfernen, wenn es existiert
    if 'Movie Title' in df.columns:
        df = df.drop(columns=['Movie Title'])
    else:
        print("'Movie Title' wurde nicht gefunden und wird daher nicht entfernt.")

    # Kategoriale Spalten definieren, auf die One-Hot-Encoding angewendet werden soll
    columns_to_encode = ['Director', 'Main Actor', 'Genre', 'Release year']  # 'Release Year' ist in deinem DataFrame als 'Release year'

    # One-Hot-Encoding anwenden
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    # Ergebnis in eine neue CSV-Datei speichern
    df_encoded.to_csv(encoded_file_path, index=False)

    print(f"Die Daten wurden erfolgreich mit One-Hot-Encoding verarbeitet und in {encoded_file_path} gespeichert.")

# Wenn dieses Skript direkt ausgeführt wird, wird die Funktion angewendet.
if __name__ == "__main__":
    input_file = 'MLWS_BOP_Dataset.csv'
    output_file = 'MLWS_BOP_Dataset_encoded.csv'
    apply_one_hot_encoding(input_file, output_file)
