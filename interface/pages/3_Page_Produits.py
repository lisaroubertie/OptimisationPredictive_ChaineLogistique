import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Dashboard Supply Chain", layout="wide")
st.title("Page Performance par Produit")


@st.cache_data
def load_data():
    test = pd.read_csv("../notebooks/Modeles/Produits/product_risk_output.csv")
    test["date"] = pd.to_datetime(test["date"])
    return test

@st.cache_resource
def load_model():
    model    = joblib.load("../notebooks/Modeles/Produits/model_product_risk.pkl")
    features = joblib.load("../notebooks/Modeles/Produits/features_product_risk.pkl")
    rmse     = joblib.load("../notebooks/Modeles/Produits/rmse_product.pkl")
    le_item  = joblib.load("../notebooks/Modeles/Produits/label_encoder_product.pkl")
    return model, features, rmse, le_item

test                       = load_data()
model, features, rmse, le_item = load_model()

st.sidebar.header("Filtres")
item_list = sorted(test["item_id"].unique())
item      = st.sidebar.selectbox("Choisir un produit", item_list)
horizon   = st.sidebar.selectbox("Horizon de prévision (jours)", [7, 14, 28])

# Navigation entre les vues (même pattern que la page Magasins)
vue = st.radio(
    "Que voulez-vous analyser ?",
    ["Ventes réelles vs prédites", "Prévision future", "Risque de rupture", "Top produits à risque"]
)


if vue == "Ventes réelles vs prédites":
    st.subheader(f"Ventes réelles vs prédites — {item}")

    item_data = test[test["item_id"] == item].sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(item_data["date"], item_data["sales"],      label="Réel",   color="black",     linewidth=2)
    ax.plot(item_data["date"], item_data["prediction"], label="Prédit", color="steelblue", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # MAE du produit sélectionné
    mae_item = round(abs(item_data["sales"] - item_data["prediction"]).mean(), 2)
    st.metric(f"MAE pour {item}", f"{mae_item} unités")

    # Détail par magasin pour le produit sélectionné
    st.subheader(f"Détail par magasin — {item}")
    magasins     = sorted(test[test["item_id"] == item]["store_id"].unique())
    magasin_sel  = st.selectbox("Choisir un magasin", magasins)

    mag_data = test[(test["item_id"] == item) & (test["store_id"] == magasin_sel)].sort_values("date")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(mag_data["date"], mag_data["sales"],      label="Réel",   color="black",     linewidth=2)
    ax2.plot(mag_data["date"], mag_data["prediction"], label="Prédit", color="steelblue", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ventes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# Prédiction récursive sur 7, 14 ou 28 jours pour le produit sélectionné
# Même principe que la page Magasins : on prédit J, on utilise
# cette prédiction comme lag pour prédire J+1, etc.


elif vue == "Prévision future":
    st.subheader(f"Prévision des {horizon} prochains jours — {item}")

    item_data  = test[test["item_id"] == item].sort_values("date")
    last_date  = item_data["date"].max()
    historique = item_data["sales"].tolist()

    predictions = []

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=i)

        lag_7          = historique[-7]  if len(historique) >= 7  else np.mean(historique)
        rolling_mean_7 = np.mean(historique[-7:]) if len(historique) >= 7 else np.mean(historique)

        # On reconstruit une ligne avec les features du modèle
        last_row = item_data.iloc[-1:].copy()
        last_row["date"]           = next_date
        last_row["lag_7"]          = lag_7
        last_row["rolling_mean_7"] = rolling_mean_7

        pred_jour = model.predict(last_row[features])[0]
        pred_jour = max(0, pred_jour)  # pas de ventes négatives

        predictions.append({"date": next_date, "prediction": pred_jour})
        historique.append(pred_jour)

    df_pred = pd.DataFrame(predictions)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Historique des 28 derniers jours
    hist_28 = item_data.tail(28)
    ax.plot(hist_28["date"], hist_28["sales"], label="Historique", color="black", linewidth=2)

    # Prévisions futures
    ax.plot(df_pred["date"], df_pred["prediction"], marker="o", linestyle="--",
            color="steelblue", label="Prévisions")

    # Séparation passé / futur
    ax.axvline(x=last_date, color="red", linestyle=":", label="Aujourd'hui")

    # Intervalle de confiance (même logique que la page Globale)
    haut = df_pred["prediction"] + rmse
    bas  = (df_pred["prediction"] - rmse).clip(0)
    ax.fill_between(df_pred["date"], bas, haut, alpha=0.2, label="Intervalle de confiance")

    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes prévues")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Tableau détaillé
    df_pred["Jour"] = range(1, len(df_pred) + 1)
    st.dataframe(df_pred[["Jour", "date", "prediction"]].rename(
        columns={"date": "Date", "prediction": "Prévision ventes"}
    ).round(0))


# Score de risque basé sur le ratio prédiction / moyenne passée
# Même logique que le notebook de modélisation produit


elif vue == "Risque de rupture":
    st.subheader(f"Score de risque de rupture — {item}")

    item_data = test[test["item_id"] == item].sort_values("date").copy()

    # Niveau de risque basé sur le risk_ratio du notebook
    item_data["risk_ratio"] = item_data["prediction"] / (item_data["rolling_mean_7"] + 1)

    def risk_level(x):
        if x < 1.1:
            return "faible"
        elif x < 1.4:
            return "moyen"
        else:
            return "élevé"

    item_data["risk"] = item_data["risk_ratio"].apply(risk_level)

    # Distribution des niveaux de risque
    counts = item_data["risk"].value_counts().reindex(["faible", "moyen", "élevé"], fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    ax.set_title(f"Distribution du risque de rupture — {item}")
    ax.set_ylabel("Nombre de jours")
    ax.set_xlabel("Niveau de risque")
    st.pyplot(fig)

    # Évolution du risk_ratio dans le temps
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(item_data["date"], item_data["risk_ratio"], color="steelblue", linewidth=1.5)
    ax2.axhline(y=1.1, color="#f39c12", linestyle="--", label="Seuil moyen (1.1)")
    ax2.axhline(y=1.4, color="#e74c3c", linestyle="--", label="Seuil élevé (1.4)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Risk ratio")
    ax2.set_title("Évolution du risk ratio dans le temps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Résumé métrique
    risque_moyen = round(item_data["risk_ratio"].mean(), 2)
    nb_jours_eleve = (item_data["risk"] == "élevé").sum()
    col1, col2 = st.columns(2)
    col1.metric("Risk ratio moyen", risque_moyen)
    col2.metric("Jours à risque élevé", nb_jours_eleve)


# Identifie les produits qui nécessitent une attention particulière
# Même logique que le tableau top_risky du notebook

elif vue == "Top produits à risque":
    st.subheader("Top produits à risque de rupture")

    # Calcul du risk_ratio pour tous les produits
    all_data = test.copy()
    all_data["risk_ratio"] = all_data["prediction"] / (all_data["rolling_mean_7"] + 1)

    top_risky = (
        all_data.groupby("item_id")
        .agg(
            avg_risk       =("risk_ratio",  "mean"),
            avg_prediction =("prediction",  "mean"),
            avg_sales      =("sales",       "mean")
        )
        .sort_values("avg_risk", ascending=False)
        .head(10)
        .reset_index()
    )

    top_risky["avg_risk"]       = top_risky["avg_risk"].round(2)
    top_risky["avg_prediction"] = top_risky["avg_prediction"].round(1)
    top_risky["avg_sales"]      = top_risky["avg_sales"].round(1)

    # Graphe horizontal
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if r >= 1.4 else "#f39c12" if r >= 1.1 else "#2ecc71"
              for r in top_risky["avg_risk"]]
    ax.barh(top_risky["item_id"], top_risky["avg_risk"], color=colors, edgecolor="black")
    ax.axvline(x=1.1, color="#f39c12", linestyle="--", label="Seuil moyen")
    ax.axvline(x=1.4, color="#e74c3c", linestyle="--", label="Seuil élevé")
    ax.set_xlabel("Risk ratio moyen")
    ax.set_title("Top 10 produits par risk ratio moyen")
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)

    # Tableau détaillé
    st.dataframe(top_risky.rename(columns={
        "item_id":        "Produit",
        "avg_risk":       "Risk ratio moyen",
        "avg_prediction": "Prévision moyenne",
        "avg_sales":      "Ventes réelles moyennes"
    }))

    # Alerte si le produit sélectionné dans la sidebar est à risque
    risk_item = all_data[all_data["item_id"] == item]["risk_ratio"].mean()
    st.divider()
    if risk_item >= 1.4:
        st.error(f"⚠️ {item} est à risque élevé de rupture (risk ratio : {round(risk_item, 2)})")
    elif risk_item >= 1.1:
        st.warning(f"⚡ {item} est à risque moyen (risk ratio : {round(risk_item, 2)})")
    else:
        st.success(f"✅ {item} est à faible risque (risk ratio : {round(risk_item, 2)})")