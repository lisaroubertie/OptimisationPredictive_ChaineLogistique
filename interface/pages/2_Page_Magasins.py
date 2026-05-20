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

# Sidebar : filtres pour choisir le magasin et l'horizon de prévision
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

    st.metric(f"MAE pour {store}", f"{mae_par_magasin[store]}")

    # Détail par produit
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
# Prédiction récursive sur les top 50 produits du magasin

elif vue == "Prévision future":
    st.subheader(f"Prévision des {horizon} prochains jours — {store}")

    store_df  = df_long[df_long['store_id'] == store].copy()
    last_date = store_df['date'].max()

    # On garde uniquement les 50 produits les plus vendus
    top50_produits = (
        store_df.groupby('item_id')['sales']
        .sum()
        .sort_values(ascending=False)
        .head(50)
        .index.tolist()
    )
    store_df = store_df[store_df['item_id'].isin(top50_produits)]

    # Barre de progression
    progress = st.progress(0)
    status   = st.empty()

    all_predictions = []

    for idx, produit in enumerate(top50_produits):

        progress.progress(int((idx + 1) / len(top50_produits) * 100))
        status.text(f"Calcul en cours... {idx+1}/{len(top50_produits)} produits")

        prod_df = store_df[store_df['item_id'] == produit].copy().sort_values('date')
        current = prod_df.copy()

        for i in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=i)
            last_row  = current.iloc[-1].copy()

            last_row['date']      = next_date
            last_row['dayofweek'] = next_date.dayofweek
            last_row['month']     = next_date.month
            last_row['event_enc'] = 0

            if len(current) >= 7:
                last_row['lag_7']          = current['sales'].iloc[-7]
                last_row['rolling_mean_7'] = current['sales'].iloc[-7:].mean()
            if len(current) >= 28:
                last_row['lag_28'] = current['sales'].iloc[-28]

            X_pred            = pd.DataFrame([last_row[features]])
            pred              = max(0, model.predict(X_pred)[0])
            last_row['sales'] = pred

            all_predictions.append({
                'date'      : next_date,
                'item_id'   : produit,
                'prediction': pred
            })

            current = pd.concat([current, pd.DataFrame([last_row])], ignore_index=True)

    progress.empty()
    status.empty()

    df_pred = pd.DataFrame(all_predictions)

    # Agrégation par jour pour le total des top 50 produits
    daily_pred = df_pred.groupby('date')['prediction'].sum().reset_index()

    st.subheader("Total magasin (top 50 produits)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_pred['date'], daily_pred['prediction'],
            marker='o', color='steelblue', linewidth=2, label='Prévision')
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes prévues")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    daily_pred['Jour'] = range(1, len(daily_pred) + 1)
    daily_pred['date'] = daily_pred['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(daily_pred[['Jour', 'date', 'prediction']].rename(
        columns={'date': 'Date', 'prediction': 'Prévision ventes totales'}
    ).round(0))

    # Top 10 produits à commander en priorité
    st.subheader("Top 10 produits à commander en priorité")
    top10 = (
        df_pred.groupby('item_id')['prediction']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top10.columns = ['Produit', 'Prévision totale sur la période']
    top10['Prévision totale sur la période'] = top10['Prévision totale sur la période'].round(0)
    st.dataframe(top10)

    # Détail par produit
    st.subheader("Prévision par produit")
    produit_choisi = st.selectbox("Choisir un produit", sorted(df_pred['item_id'].unique()))
    prod_pred      = df_pred[df_pred['item_id'] == produit_choisi].copy()
    prod_pred      = prod_pred.reset_index(drop=True)
    prod_pred['Jour'] = range(1, len(prod_pred) + 1)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(prod_pred['date'], prod_pred['prediction'],
             marker='o', color='steelblue', linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ventes prévues")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    prod_pred['date'] = prod_pred['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(prod_pred[['Jour', 'date', 'prediction']].rename(
        columns={'date': 'Date', 'prediction': 'Prévision ventes'}
    ).round(0))


# VUE 3 : MAE PAR MAGASIN

elif vue == "MAE par magasin":
    st.subheader("MAE par magasin — plus c'est bas mieux c'est prédit")

    df_mae         = mae_par_magasin.reset_index()
    df_mae.columns = ['Magasin', 'MAE']
    df_mae         = df_mae.sort_values('MAE')

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ['red' if s == store else 'steelblue' for s in df_mae['Magasin']]
    ax.bar(df_mae['Magasin'], df_mae['MAE'], color=colors, edgecolor='black')
    ax.set_ylabel("MAE")
    ax.set_title("MAE par magasin (rouge = magasin sélectionné)")
    st.pyplot(fig)

    st.metric(f"MAE pour {store}", f"{mae_par_magasin[store]}")

# VUE 4 : ALERTES ÉVÉNEMENTS

elif vue == "Alertes événements":
    st.subheader("Événements à venir — mai/juin 2016")

    # Moyenne des ventes journalières du magasin
    store_recent        = df_long[df_long['store_id'] == store].copy()
    daily_recent        = store_recent.groupby('date')['sales'].sum()
    moyenne_journaliere = daily_recent.mean()

    # Impacts par magasin issus de l'exploration des données
    events = {
        "MemorialDay": {
            "date"  : "2016-05-30",
            "impact": {
                "CA_1": 35,  "CA_2": 20,  "CA_3": 15,  "CA_4": 18,
                "TX_1": 29,  "TX_2": 22,  "TX_3": 25,
                "WI_1": -25, "WI_2": -35, "WI_3": -33
            },
            "action": "Vérifier le comportement historique — impact très variable selon le magasin"
        },
        "Father's day": {
            "date"  : "2016-06-19",
            "impact": {
                "CA_1": 25, "CA_2": 27, "CA_3": 22, "CA_4": 29,
                "TX_1": 34, "TX_2": 30, "TX_3": 31,
                "WI_1": 28, "WI_2": -28, "WI_3": 26
            },
            "action": "Anticiper les stocks une semaine avant"
        }
    }

    for event, info in events.items():
        impact_pct = info['impact'].get(store, 0)
        unites_supplementaires = int(moyenne_journaliere * abs(impact_pct) / 100)

        if impact_pct > 0:
            st.success(
                f"**{event}** le {info['date']} → "
                f"Hausse prévue pour {store} : **+{impact_pct}%** "
                f"soit environ **+{unites_supplementaires} unités** ce jour-là"
            )
            st.info(
                f"Recommandation : {info['action']} — "
                f"Commander environ **{unites_supplementaires} unités supplémentaires**"
            )
        else:
            st.error(
                f"**{event}** le {info['date']} → "
                f"Baisse prévue pour {store} : **{impact_pct}%** "
                f"soit environ **-{unites_supplementaires} unités** ce jour-là"
            )
            st.info(
                f"Recommandation : Ne pas sur-stocker — "
                f"Réduire les commandes d'environ **{unites_supplementaires} unités**"
            )