import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Dashboard Supply Chain", layout="wide")
st.title("Page Performance par Magasin")

# @st.cache_data : charge les données une seule fois et les garde en mémoire
# évite de relire le fichier CSV à chaque interaction de l'utilisateur
@st.cache_data
def load_data():
    test = pd.read_csv('../notebooks/Modeles/Magasins/test_predictions_magasins.csv')
    test['date'] = pd.to_datetime(test['date'])
    return test

# @st.cache_resource : charge le modèle une seule fois et le garde en mémoire
# évite de réentraîner le modèle à chaque interaction — indispensable pour la performance
@st.cache_resource
def load_model():
    model           = joblib.load('../notebooks/Modeles/Magasins/modele_xgboost_magasins.pkl')
    features        = joblib.load('../notebooks/Modeles/Magasins/features_magasins.pkl')
    mae             = joblib.load('../notebooks/Modeles/Magasins/mae_magasins.pkl')
    mae_par_magasin = joblib.load('../notebooks/Modeles/Magasins/mae_par_magasin.pkl')
    return model, features, mae, mae_par_magasin

test            = load_data()
model, features, mae_global, mae_par_magasin = load_model()


st.sidebar.header("Filtres")
store_list = sorted(test['store_id'].unique())
store      = st.sidebar.selectbox("Choisir un magasin", store_list)

# BOUTON RADIO https://docs.streamlit.io/develop/api-reference/widgets/st.radio
vue = st.radio(
    "Que voulez-vous analyser ?",
    ["Ventes réelles vs prédites", "MAE par magasin", "Alertes événements"]
)

# VUE 1 : RÉEL VS PRÉDIT
if vue == "Ventes réelles vs prédites":
    st.subheader(f"Ventes réelles vs prédites — {store}")
    
    store_data = test[test['store_id'] == store].groupby('date')[['sales', 'prediction']].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(store_data['date'], store_data['sales'], label='Réel', color='black', linewidth=2)
    ax.plot(store_data['date'], store_data['prediction'], label='Prédit', color='steelblue', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.metric(f"MAE global", f"{mae_global}")

# VUE 2 : MAE PAR MAGASIN 
elif vue == "MAE par magasin":
    st.subheader("MAE par magasin — plus c'est bas mieux c'est prédit")
    
    df_mae = mae_par_magasin.reset_index()
    df_mae.columns = ['Magasin', 'MAE']
    df_mae = df_mae.sort_values('MAE')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['red' if s == store else 'steelblue' for s in df_mae['Magasin']]
    ax.bar(df_mae['Magasin'], df_mae['MAE'], color=colors, edgecolor='black')
    ax.set_ylabel("MAE")
    ax.set_title("MAE par magasin (rouge = magasin sélectionné)")
    st.pyplot(fig)
    
    st.metric(f"MAE pour {store}", f"{mae_par_magasin[store]}")

#  VUE 3 : ALERTES ÉVÉNEMENTS 
elif vue == "Alertes événements":
    st.subheader("Événements à venir — mai/juin 2016")
    
    events_mai_juin = {
        "Mother's day"  : {"date": "2016-05-08", "impact": "+25% à +43% selon le magasin", "action": "Anticiper les stocks une semaine avant"},
        "MemorialDay"   : {"date": "2016-05-30", "impact": "Variable selon le magasin (-35% à +35%)", "action": "Vérifier le comportement historique du magasin"},
        "Father's day"  : {"date": "2016-06-19", "impact": "+25% à +34% selon le magasin", "action": "Anticiper les stocks une semaine avant"}
    }
    
    for event, info in events_mai_juin.items():
        st.warning(f"**{event}** le {info['date']} → {info['impact']}")
        st.info(f"Recommandation : {info['action']}")
