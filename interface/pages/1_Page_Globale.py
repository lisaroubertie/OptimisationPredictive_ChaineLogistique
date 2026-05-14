import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt




st.set_page_config(page_title="Dashboard Supply Chain", layout="wide")

st.title("Dashboard Supply Chain - Page Globale")




# Pour la sidebar

# Filtre pour l'horizon
st.sidebar.title("Filtre")
horizon = st.sidebar.selectbox(
    "Horizon de prévision",
    [7, 14, 28]
)

# Filtre pour l'echelle
echelle = st.sidebar.radio("Echelle des graphiques", ["Linéaire", "Logarithmique"])




# Chargement des fichiers

daily = pd.read_csv("../notebooks/Modeles/Global/daily_regression.csv") # pour la courbe historique
derniers_28 = pd.read_csv("../notebooks/Modeles/Global/28derniersjours_regression.csv")
daily["date"] = pd.to_datetime(daily["date"])
derniers_28["date"] = pd.to_datetime(derniers_28["date"])

modele_regression = joblib.load("../notebooks/Modeles/Global/modele_regression.pkl")
modele_classification = joblib.load("../notebooks/Modeles/Global/modele_classification.pkl")

features = joblib.load("../notebooks/Modeles/Global/features_regression.pkl")
seuil_pic = joblib.load("../notebooks/Modeles/Global/seuil_pic_classification.pkl")
rmse = joblib.load("../notebooks/Modeles/Global/rmse_regression.pkl") # pour l'interval de confiance



# Date du jour
st.caption(f"Dernière date disponible dans le dataset : {derniers_28['date'].iloc[-1].date()}")



# Courbe de l'historique des ventes
st.subheader("Historique de demande")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(daily["date"], daily["sales"].rolling(7).mean(), color="black") # affichage lissé sur 7 jours pour etre plus lisible
ax.set_title("Ventes globales dans le temps")
ax.set_xlabel("Date")
ax.set_ylabel("Ventes")

# Changement d'échelle
if echelle == "Logarithmique":
    ax.set_yscale("log")

st.pyplot(fig)




# Pour les prévision 
# On utilise le modèle de régression 
st.subheader(f"Prévision des {horizon} prochains jours")


# Ici, on ne prédisait pas  le futur parce qu'on n'avait pas reconstruit les features pour les nouvelles dates à venir
# X_pred = derniers_28[features].tail(horizon)
# pred_ventes = modele_regression.predict(X_pred)


# Features pour les jours futurs
rows = []
historique = daily["sales"].tolist()  # ventes passées

for i in range(horizon):

    future_date = derniers_28["date"].iloc[-1] + pd.Timedelta(days=i+1)
    lag_1  = historique[-1] 
    lag_7  = historique[-7]
    lag_28 = historique[-28] 
    rolling_mean_7 = np.mean(historique[-7:])
    rolling_std_7  = np.std(historique[-7:], ddof=1)
    
    row = {
        "date": future_date,
        "dayofweek": future_date.dayofweek,
        "month": future_date.month,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7
    }
    rows.append(row)
    
    ligne_future = pd.DataFrame([row]) # transfo de la ligne en df 
    ligne_future = ligne_future[features] # on garde les features utilisées pendant l'entrainement
    pred_jour = modele_regression.predict(ligne_future) # prédiction du prochain jour
    pred_jour=pred_jour[0]
    historique.append(pred_jour) # pour les lags des jours suivants

future_df = pd.DataFrame(rows)
X_pred = future_df[features]
pred_ventes = np.array(historique[-horizon:])

forecast = pd.DataFrame({
    "Jour futur": np.arange(1, horizon + 1),
    "Prévision ventes": pred_ventes
})

future_dates = pd.date_range(
    start=derniers_28["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=horizon
)

fig2, ax2 = plt.subplots(figsize=(15,5))

# historique des 28 derniers jours
ax2.plot(derniers_28["date"], derniers_28["sales"], label="Historique", color="black")

# les prévisions futures
ax2.plot(future_dates, pred_ventes, marker="o", linestyle="--", label="Prévisions")

# pour l'axe de séparation entre passé et futur
ax2.axvline(x=derniers_28["date"].iloc[-1], color="red", linestyle=":")

# affichage du seuil sur le graphe
ax2.axhline(y=seuil_pic, linestyle="--", color="orange", label="Seuil pic") 

# pour l'interval de confiance 
haut = pred_ventes + rmse
bas = pred_ventes - rmse
ax2.fill_between(future_dates, bas, haut, alpha=0.2, label="Interval de confiance")

ax2.set_title("Historique des 28 derniers jours de ventes et prévisions de la demande à venir")
ax2.set_xlabel("Date")
ax2.set_ylabel("Ventes")

# Prise en compte de l'échelle en filtre
if echelle == "Logarithmique":
    ax2.set_yscale("log")

ax2.legend()
st.pyplot(fig2)




# Détection du nombres de pics
# Ici, on passe au modèle de classification

st.subheader("Détection du risque de pic")

pred_class = modele_classification.predict(X_pred)

nb_pic = np.sum(pred_class)

st.write("Seuil pris en compte :", seuil_pic)

if nb_pic > 0:
    st.error(f"Risque de pic détecté sur {nb_pic} jour(s) de la période")
else:
    st.success("Aucun pic détecté sur la période")




# Détails des prévisions

st.subheader(f"Prévision des {horizon} prochains jours dans le détail")

st.dataframe(forecast) # Tableau forecast