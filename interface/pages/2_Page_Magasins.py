import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Dashboard Supply Chain", layout="wide")
st.title("Page Performance par Magasin")

# Chargement des prédictions calculées dans le notebook de modélisation
def load_data():
    test = pd.read_csv('../notebooks/Modeles/Magasins/test_predictions_magasins.csv')
    test['date'] = pd.to_datetime(test['date'])
    return test

# Chargement du modèle XGBoost et de tous les fichiers exportés depuis le notebook
def load_model():
    model           = joblib.load('../notebooks/Modeles/Magasins/modele_xgboost_magasins.pkl')
    features        = joblib.load('../notebooks/Modeles/Magasins/features_magasins.pkl')
    mae             = joblib.load('../notebooks/Modeles/Magasins/mae_magasins.pkl')
    mae_par_magasin = joblib.load('../notebooks/Modeles/Magasins/mae_par_magasin.pkl')
    df_long         = joblib.load('../notebooks/Modeles/Magasins/df_long_magasins.pkl')
    le              = joblib.load('../notebooks/Modeles/Magasins/label_encoder_magasins.pkl')
    return model, features, mae, mae_par_magasin, df_long, le

test                                                      = load_data()
model, features, mae_global, mae_par_magasin, df_long, le = load_model()


# SIDEBAR
# Le responsable choisit son magasin et son horizon de prévision

st.sidebar.header("Filtres")
store_list = sorted(test['store_id'].unique())
store      = st.sidebar.selectbox("Choisir un magasin", store_list)
horizon    = st.sidebar.selectbox("Horizon de prévision (jours)", [7, 14, 28])

# Navigation entre les 4 vues disponibles
vue = st.radio(
    "Que voulez-vous analyser ?",
    ["Ventes réelles vs prédites", "Prévision future", "MAE par magasin", "Alertes événements"]
)


# VUE 1 : RÉEL VS PRÉDIT
# Permet de valider visuellement la qualité du modèle
# On affiche d'abord les ventes agrégées du magasin
# puis le détail par produit pour aller plus loin

if vue == "Ventes réelles vs prédites":
    st.subheader(f"Ventes réelles vs prédites — {store}")

    # Agrégation par date : on somme tous les produits pour avoir le total magasin
    store_data = test[test['store_id'] == store].groupby('date')[['sales', 'prediction']].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(store_data['date'], store_data['sales'],      label='Réel',   color='black',     linewidth=2)
    ax.plot(store_data['date'], store_data['prediction'], label='Prédit', color='steelblue', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes totales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # MAE du magasin sélectionné : erreur moyenne en unités
    st.metric(f"MAE pour {store}", f"{mae_par_magasin[store]}")

    # Détail par produit : utile pour identifier les produits mal prédits
    st.subheader(f"Détail par produit — {store}")
    produits = sorted(test[test['store_id'] == store]['item_id'].unique())
    produit  = st.selectbox("Choisir un produit", produits)

    produit_data = test[(test['store_id'] == store) & (test['item_id'] == produit)]

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(produit_data['date'], produit_data['sales'],      label='Réel',   color='black',     linewidth=2)
    ax2.plot(produit_data['date'], produit_data['prediction'], label='Prédit', color='steelblue', linestyle='--')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ventes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)


# VUE 2 : PRÉVISION FUTURE
# Prédiction récursive sur 7, 14 ou 28 jours
# Principe : on prédit jour J, on utilise cette prédiction
# comme lag pour prédire jour J+1, et ainsi de suite

elif vue == "Prévision future":
    st.subheader(f"Prévision des {horizon} prochains jours — {store}")

    # On part des dernières données connues du magasin sélectionné
    store_df  = df_long[df_long['store_id'] == store].copy()
    last_date = store_df['date'].max()

    predictions = []
    current_df  = store_df.copy()

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=i)

        # Dernière ligne connue par produit = point de départ de la prédiction
        last_rows = current_df.groupby('item_id').last().reset_index()

        # Mise à jour des variables temporelles pour le jour à prédire
        last_rows['date']      = next_date
        last_rows['dayofweek'] = next_date.dayofweek
        last_rows['month']     = next_date.month

        # Recalcul des lags depuis les données disponibles
        last_rows['lag_7']  = current_df.groupby('item_id')['sales'].nth(-7).values
        last_rows['lag_28'] = current_df.groupby('item_id')['sales'].nth(-28).values
        last_rows['rolling_mean_7'] = current_df.groupby('item_id')['sales'].apply(
            lambda x: x.iloc[-7:].mean()
        ).values

        # Pas d'événement par défaut --> les alertes events sont gérées séparément
        last_rows['event_enc'] = 0

        # Prédiction avec clip à 0 car on ne peut pas vendre des quantités négatives
        last_rows['prediction'] = model.predict(last_rows[features]).clip(0)
        predictions.append(last_rows[['date', 'item_id', 'prediction']])

        # La prédiction devient la nouvelle valeur de sales pour calculer les lags suivants
        last_rows['sales'] = last_rows['prediction']
        current_df = pd.concat([current_df, last_rows[current_df.columns.intersection(last_rows.columns)]])

    # Agrégation par jour : on somme les prédictions de tous les produits
    df_pred = pd.concat(predictions).groupby('date')['prediction'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_pred['date'], df_pred['prediction'], marker='o', color='steelblue', linewidth=2, label='Prévision')
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes prévues")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Tableau détaillé des prévisions jour par jour
    df_pred['Jour'] = range(1, len(df_pred) + 1)
    st.dataframe(df_pred[['Jour', 'date', 'prediction']].rename(
        columns={'date': 'Date', 'prediction': 'Prévision ventes'}
    ).round(0))


# VUE 3 : MAE PAR MAGASIN
# Compare la précision du modèle entre tous les magasins
# Le magasin sélectionné est mis en rouge pour le situer

elif vue == "MAE par magasin":
    st.subheader("MAE par magasin — plus c'est bas mieux c'est prédit")

    df_mae         = mae_par_magasin.reset_index()
    df_mae.columns = ['Magasin', 'MAE']
    df_mae         = df_mae.sort_values('MAE')

    fig, ax = plt.subplots(figsize=(10, 4))
    # Rouge pour le magasin sélectionné, bleu pour les autres
    colors = ['red' if s == store else 'steelblue' for s in df_mae['Magasin']]
    ax.bar(df_mae['Magasin'], df_mae['MAE'], color=colors, edgecolor='black')
    ax.set_ylabel("MAE")
    ax.set_title("MAE par magasin (rouge = magasin sélectionné)")
    st.pyplot(fig)

    st.metric(f"MAE pour {store}", f"{mae_par_magasin[store]}")


# VUE 4 : ALERTES ÉVÉNEMENTS
# Impacts calculés depuis l'analyse (notebook events)
# Chaque magasin a son propre impact car les comportements
# sont très hétérogènes d'un magasin à l'autre
# Vert = hausse des ventes, on anticipe les stocks
# Rouge = baisse des ventes, ne pas sur-stocker

elif vue == "Alertes événements":
    st.subheader("Événements à venir — mai/juin 2016")

    # Impacts par magasin issus de l'exploration des données (notebook Exploration_Events)
    events = {
        "Mother's day": {
            "date"  : "2016-05-08",
            "impact": {
                "CA_1": "+25%", "CA_2": "+28%", "CA_3": "+20%", "CA_4": "+31%",
                "TX_1": "+35%", "TX_2": "+30%", "TX_3": "+32%",
                "WI_1": "+43%", "WI_2": "+38%", "WI_3": "+29%"
            },
            "action": "Anticiper les stocks une semaine avant"
        },
        "MemorialDay": {
            "date"  : "2016-05-30",
            "impact": {
                "CA_1": "+35%", "CA_2": "+20%", "CA_3": "+15%", "CA_4": "+18%",
                "TX_1": "+29%", "TX_2": "+22%", "TX_3": "+25%",
                "WI_1": "-25%", "WI_2": "-35%", "WI_3": "-33%"
            },
            "action": "Vérifier le comportement historique — impact très variable selon le magasin"
        },
        "Father's day": {
            "date"  : "2016-06-19",
            "impact": {
                "CA_1": "+25%", "CA_2": "+27%", "CA_3": "+22%", "CA_4": "+29%",
                "TX_1": "+34%", "TX_2": "+30%", "TX_3": "+31%",
                "WI_1": "+28%", "WI_2": "-28%", "WI_3": "+26%"
            },
            "action": "Anticiper les stocks une semaine avant"
        }
    }

    for event, info in events.items():
        impact_magasin = info['impact'].get(store, "Non disponible")

        # Vert si hausse des ventes donc Sophie doit commander plus
        # Rouge si baisse des ventes donc Sophie ne doit pas sur-stocker
        if '-' in impact_magasin:
            st.error(f"**{event}** le {info['date']} → Impact pour {store} : {impact_magasin}")
        else:
            st.success(f"**{event}** le {info['date']} → Impact pour {store} : {impact_magasin}")

        st.info(f"Recommandation : {info['action']}")