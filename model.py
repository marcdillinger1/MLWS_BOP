import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_xgboost_model_with_cv(data_file, target_column, model_output_file):
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
    mse_scores = []
    mape_scores = []
    mae_scores = []
    r2_scores = []
    rmse_scores = []

    # Hyperparameter für das Modell definieren (manuelles Tuning)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
    }

    best_mse = float("inf")
    best_params = None

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
                                mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred_log))  # Rücktransformieren auf Originalskala
                                mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_log))
                                r2 = r2_score(np.expm1(y_val), np.expm1(y_pred_log))
                                rmse = np.sqrt(mse)

                                # Ergebnisse für jede Metrik speichern
                                fold_mse.append(mse)
                                fold_mape.append(mape)
                                fold_mae.append(mae)
                                fold_r2.append(r2)
                                fold_rmse.append(rmse)

                            # Durchschnitt der Metriken berechnen
                            avg_mse = np.mean(fold_mse)
                            avg_mape = np.mean(fold_mape)
                            avg_mae = np.mean(fold_mae)
                            avg_r2 = np.mean(fold_r2)
                            avg_rmse = np.mean(fold_rmse)

                            # Ausgabe der Metriken
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


# Wenn dieses Skript direkt ausgeführt wird, wird das Modell trainiert.
if __name__ == "__main__":
    input_file = 'MLWS_BOP_Dataset_encoded.csv'  # Encodierte CSV-Datei
    target_column = 'Box Office'  # Zielvariable
    output_model_file = 'xgboost_model_with_cv.json'  # Name der gespeicherten Modelldatei

    train_xgboost_model_with_cv(input_file, target_column, output_model_file)
