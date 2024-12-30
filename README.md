# Projektarbeit Maschinelles-Lernen und Wissensbasierte-Systeme

## Box-Office Prediction mit XG-Boost

### Vorhersage des Box-Office Umsatzes von Kinofilmen mithilfe eines XG-Boost Modells

`MLWS_BOP_Dataset.csv` - Datensatz mit den Aufbereiteten Daten der Filme

`MLWS_BOP_Dataset_encoded.csv` - Datensatz mit den codierten Daten der Filme welcher vom XG-Boost Modell verwendet wird

`encoding.py` - Lädt den ursprünglichen Datensatz `MLWS_BOP_Dataset.csv` und wendet auf die kategorialen Features One-Hot-Encoding an und generiert daraus den codierten Datensatz `MLWS_BOP_Dataset_encoded.csv`

`model.py` - Trainiert das XG-boost modell auf dem Datensatz `MLWS_BOP_Dataset_encoded.csv` und speichert das finale Modell `xgboost_model_with_cv.json` ab und speichert ausserdem die dazugehörigen Hyperparameter und Leistungsmetriken in `model_metrics.txt`

`feature_importance.py` - Wertet die Feature Importance aus und erstellt daraus die Grafiken für Cover `feature_importance_cover.png`, Gain `feature_importance_gain.png`, und Weight `feature_importance_weight.png`