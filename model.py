import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_xgboost_model_with_cv(data_file, target_column, model_output_file, metrics_output_file):
    # Datensatz laden
    df = pd.read_csv(data_file)

    # Zielvariable und Features trennen
    X = df.drop(columns=[target_column])  # Alle Spalten außer der Zielvariable
    y = df[target_column]  # Zielvariable

    # Überprüfen auf fehlende Werte (NaN) in den Daten
    if df.isnull().sum().any():
        print("Warnung: Es gibt fehlende Werte (NaN) im Datensatz.")
        # Fehlende Werte im Datensatz auffüllen (zum Beispiel mit dem Mittelwert)
        df.fillna(df.mean(), inplace=True)
        X = df.drop(columns=[target_column])
        y = df[target_column]

    # Zielvariable log-transformieren (log1p für stabilere Transformation)
    y_log = np.log1p(y)

    # Skalierung der Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-Validation manuell durchführen
    n_splits = 5
    fold_size = len(X_scaled) // n_splits
    best_mse = float("inf")
    best_params = None
    avg_mse = avg_mape = avg_mae = avg_r2 = avg_rmse = None

    # Hyperparameter für das Modell definieren (manuelles Tuning)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
    }

    # Manuelles Hyperparameter-Tuning und Cross-Validation
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for min_child_weight in param_grid['min_child_weight']:
                            # XGBoost-Modell initialisieren
                            model = xgb.XGBRegressor(
                                objective="reg:squarederror",
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                min_child_weight=min_child_weight,
                                random_state=42
                            )

                            fold_mse = []
                            fold_mape = []
                            fold_mae = []
                            fold_r2 = []
                            fold_rmse = []

                            # Manuelle Cross-Validation
                            for fold in range(n_splits):
                                start_idx = fold * fold_size
                                end_idx = (fold + 1) * fold_size if fold != n_splits - 1 else len(X_scaled)

                                X_train = np.concatenate([X_scaled[:start_idx], X_scaled[end_idx:]], axis=0)
                                y_train = np.concatenate([y_log[:start_idx], y_log[end_idx:]], axis=0)
                                X_val = X_scaled[start_idx:end_idx]
                                y_val = y_log[start_idx:end_idx]

                                # Modell trainieren
                                model.fit(X_train, y_train)

                                # Vorhersagen
                                y_pred_log = model.predict(X_val)

                                # Metriken berechnen
                                mse = mean_squared_error(y_val, y_pred_log)
                                mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(
                                    y_pred_log))  # Rücktransformieren auf Originalskala
                                mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_log))
                                r2 = r2_score(np.expm1(y_val), np.expm1(y_pred_log))
                                rmse = np.sqrt(mse)

                                fold_mse.append(mse)
                                fold_mape.append(mape)
                                fold_mae.append(mae)
                                fold_r2.append(r2)
                                fold_rmse.append(rmse)

                            # Durchschnittswerte der Metriken für das aktuelle Modell
                            avg_mse = np.mean(fold_mse)
                            avg_mape = np.mean(fold_mape)
                            avg_mae = np.mean(fold_mae)
                            avg_r2 = np.mean(fold_r2)
                            avg_rmse = np.mean(fold_rmse)

                            print(f"Hyperparameter: n_estimators={n_estimators}, max_depth={max_depth}, "
                                  f"learning_rate={learning_rate}, subsample={subsample}, "
                                  f"colsample_bytree={colsample_bytree}, min_child_weight={min_child_weight}")
                            print(f"Durchschnittlicher MSE: {avg_mse:.2f}, Durchschnittlicher MAPE: {avg_mape:.2f}%, "
                                  f"Durchschnittlicher MAE: {avg_mae:.2f}, Durchschnittlicher R²: {avg_r2:.2f}, "
                                  f"Durchschnittlicher RMSE: {avg_rmse:.2f}")

                            # Speichern der besten Hyperparameter
                            if avg_mse < best_mse:
                                best_mse = avg_mse
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'min_child_weight': min_child_weight
                                }

    print("Beste Hyperparameter:", best_params)

    # Finale Modell-Trainierung mit den besten Hyperparametern
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        **best_params,
        random_state=42
    )
    final_model.fit(X_scaled, y_log)

    # Modell speichern
    final_model.save_model(model_output_file)
    print(f"Modell wurde gespeichert unter {model_output_file}")

    # Textdatei mit den Metriken und besten Hyperparametern speichern
    with open(metrics_output_file, 'w') as f:
        f.write("Modelltraining mit XGBoost: Leistungsmetriken und Hyperparameter\n\n")

        f.write("Beste Hyperparameter:\n")
        f.write(f"n_estimators: {best_params['n_estimators']} - Anzahl der Bäume im Modell.\n")
        f.write(f"max_depth: {best_params['max_depth']} - Maximale Tiefe der Entscheidungsbäume.\n")
        f.write(f"learning_rate: {best_params['learning_rate']} - Lernrate des Modells.\n")
        f.write(
            f"subsample: {best_params['subsample']} - Prozentsatz der Trainingsdaten, die in jedem Baum verwendet werden.\n")
        f.write(
            f"colsample_bytree: {best_params['colsample_bytree']} - Prozentsatz der Features, die bei jedem Baum verwendet werden.\n")
        f.write(
            f"min_child_weight: {best_params['min_child_weight']} - Mindestanzahl an Instanzen, um einen neuen Split zu erzeugen.\n\n")

        f.write("Leistungsmetriken:\n")
        f.write(
            f"Durchschnittlicher MSE (Mean Squared Error): {avg_mse:.2f} - Misst den durchschnittlichen quadratischen Fehler zwischen Vorhersage und tatsächlichem Wert.\n")
        f.write(
            f"Durchschnittlicher MAPE (Mean Absolute Percentage Error): {avg_mape:.2f}% - Misst den durchschnittlichen prozentualen Fehler.\n")
        f.write(
            f"Durchschnittlicher MAE (Mean Absolute Error): {avg_mae:.2f} - Misst den durchschnittlichen absoluten Fehler.\n")
        f.write(
            f"Durchschnittlicher R² (R-Quadrat): {avg_r2:.2f} - Misst, wie gut das Modell die Varianz der Zielvariable erklärt.\n")
        f.write(
            f"Durchschnittlicher RMSE (Root Mean Squared Error): {avg_rmse:.2f} - Misst die Quadratwurzel des durchschnittlichen quadratischen Fehlers.\n")

    print(f"Modell- und Metrikeninformationen wurden gespeichert unter {metrics_output_file}")


# Wenn dieses Skript direkt ausgeführt wird, wird das Modell trainiert.
if __name__ == "__main__":
    input_file = 'MLWS_BOP_Dataset_encoded.csv'  # Encodierte CSV-Datei
    target_column = 'Box Office'  # Zielvariable
    output_model_file = 'xgboost_model_with_cv.json'  # Name der gespeicherten Modelldatei
    metrics_output_file = 'model_metrics.txt'  # Name der Datei für Metriken und Hyperparameter

    train_xgboost_model_with_cv(input_file, target_column, output_model_file, metrics_output_file)
