import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model_file, input_file, target_column):
    # Modell laden (direkt über Booster API von XGBoost)
    model = xgb.Booster()
    model.load_model(model_file)

    # Datensatz laden
    df = pd.read_csv(input_file)

    # Zielvariable und Features trennen
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Erstelle ein Mapping der Feature-Namen
    feature_names_mapping = {f"f{i}": col for i, col in enumerate(X.columns)}

    # Wichtigkeit der Features extrahieren (Weight, Gain, Cover)
    importance = model.get_score(importance_type='weight')
    gain = model.get_score(importance_type='gain')
    cover = model.get_score(importance_type='cover')

    # Konvertiere in DataFrames für eine übersichtliche Darstellung
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Weight'])
    gain_df = pd.DataFrame(list(gain.items()), columns=['Feature', 'Gain'])
    cover_df = pd.DataFrame(list(cover.items()), columns=['Feature', 'Cover'])

    # Verknüpfe ursprüngliche Feature-Namen mit den transformierten Feature-Namen
    importance_df['Original Feature'] = importance_df['Feature'].map(feature_names_mapping)
    gain_df['Original Feature'] = gain_df['Feature'].map(feature_names_mapping)
    cover_df['Original Feature'] = cover_df['Feature'].map(feature_names_mapping)

    # Zeige die Feature-Wichtigkeit nach Weight, Gain und Cover
    print("\nFeature Importance (Weight):")
    print(importance_df.sort_values(by='Weight', ascending=False))

    print("\nFeature Importance (Gain):")
    print(gain_df.sort_values(by='Gain', ascending=False))

    print("\nFeature Importance (Cover):")
    print(cover_df.sort_values(by='Cover', ascending=False))

    # Plot für Weight
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Weight',
        y='Original Feature',
        data=importance_df.sort_values(by='Weight', ascending=False).head(20),
        palette='viridis',
        hue='Original Feature',
        legend=False
    )
    plt.title('Top 20 Features by Weight')
    plt.tight_layout()
    plt.savefig('feature_importance_weight.png')
    plt.close()

    # Plot für Gain
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Gain',
        y='Original Feature',
        data=gain_df.sort_values(by='Gain', ascending=False).head(20),
        palette='viridis',
        hue='Original Feature',
        legend=False
    )
    plt.title('Top 20 Features by Gain')
    plt.tight_layout()
    plt.savefig('feature_importance_gain.png')
    plt.close()

    # Plot für Cover
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Cover',
        y='Original Feature',
        data=cover_df.sort_values(by='Cover', ascending=False).head(20),
        palette='viridis',
        hue='Original Feature',
        legend=False
    )
    plt.title('Top 20 Features by Cover')
    plt.tight_layout()
    plt.savefig('feature_importance_cover.png')
    plt.close()


# Beispiel Aufruf des Codes:
model_file = 'xgboost_model_with_cv.json'  # Dein Modell
input_file = 'MLWS_BOP_Dataset_encoded.csv'  # Dein Datensatz
target_column = 'Box Office'  # Zielvariable

plot_feature_importance(model_file, input_file, target_column)
