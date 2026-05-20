import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Dashboard Supply Chain", layout="wide")
st.title("Page Performance par Magasin")

def load_data():
    test = pd.read_csv('../notebooks/Modeles/Magasins/test_predictions_magasins.csv')
    test['date'] = pd.to_datetime(test['date'])
    return test

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

st.sidebar.header("Filtres")
store_list = sorted(test['store_id'].unique())
store      = st.sidebar.selectbox("Choisir un magasin", store_list)
horizon    = st.sidebar.selectbox("Horizon de prévision (jours)", [7, 14, 28])

st.caption(f"Dernière date disponible dans le dataset : {test['date'].max().date()}")

vue = st.radio(
    "Que voulez-vous analyser ?",
    ["Ventes réelles vs prédites", "Prévision future", "MAE par magasin", "Alertes événements"]
)

# VUE 1 : RÉEL VS PRÉDIT
if vue == "Ventes réelles vs prédites":
    st.subheader(f"Ventes réelles vs prédites — {store}")

    store_data = test[test['store_id'] == store].groupby('date')[['sales', 'prediction']].sum().reset_index()

    # Métriques en haut
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventes réelles moyennes / jour", f"{store_data['sales'].mean():.0f}")
    c2.metric("Maximum historique",             f"{store_data['sales'].max():.0f}")
    c3.metric(f"MAE pour {store}",              f"{mae_par_magasin[store]}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(store_data['date'], store_data['sales'],      label='Réel',   color='black',     linewidth=2)
    ax.plot(store_data['date'], store_data['prediction'], label='Prédit', color='steelblue', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes totales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

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
elif vue == "Prévision future":
    st.subheader(f"Prévision des {horizon} prochains jours — {store}")

    if horizon > 7:
        erreur_estimee = round(0.102 * np.sqrt(horizon), 2)
        st.warning(f"Au-delà de 7 jours la précision diminue progressivement. "
                   f"Sur {horizon} jours l'erreur accumulée estimée est d'environ "
                   f"{erreur_estimee} unités par produit par jour.")

    store_df  = df_long[df_long['store_id'] == store].copy()
    last_date = store_df['date'].max()

    top50_produits = (
        store_df.groupby('item_id')['sales']
        .sum()
        .sort_values(ascending=False)
        .head(50)
        .index.tolist()
    )
    store_df = store_df[store_df['item_id'].isin(top50_produits)]

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

            if len(current) >= 1:
                last_row['lag_1'] = current['sales'].iloc[-1]
            if len(current) >= 7:
                last_row['lag_7']          = current['sales'].iloc[-7]
                last_row['rolling_mean_7'] = current['sales'].iloc[-7:].mean()
                last_row['rolling_std_7']  = current['sales'].iloc[-7:].std()
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

    df_pred    = pd.DataFrame(all_predictions)
    daily_pred = df_pred.groupby('date')['prediction'].sum().reset_index()

    # Métriques en haut
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventes prévues totales sur la période", f"{daily_pred['prediction'].sum():.0f}")
    c2.metric("Pic prévu",                             f"{daily_pred['prediction'].max():.0f}")
    c3.metric("Moyenne prévue / jour",                 f"{daily_pred['prediction'].mean():.0f}")

    st.subheader("Total magasin (top 50 produits)")

    # Historique des 28 derniers jours + prévisions sur le même graphe
    store_hist = df_long[df_long['store_id'] == store].copy()
    store_hist = store_hist[store_hist['item_id'].isin(top50_produits)]
    hist_28    = store_hist.groupby('date')['sales'].sum().reset_index().tail(28)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Historique
    ax.plot(hist_28['date'], hist_28['sales'],
            label='Historique', color='black', linewidth=2)

    # Prévisions
    ax.plot(daily_pred['date'], daily_pred['prediction'],
            marker='o', color='steelblue', linewidth=2, label='Prévision', linestyle='--')

    # Intervalle de confiance qui s'élargit avec le temps
    rmse_total = mae_global * 50
    haut = daily_pred['prediction'] + rmse_total * np.sqrt(np.arange(1, len(daily_pred) + 1))
    bas  = (daily_pred['prediction'] - rmse_total * np.sqrt(np.arange(1, len(daily_pred) + 1))).clip(0)
    ax.fill_between(daily_pred['date'], bas, haut, alpha=0.2, label='Intervalle de confiance')

    # Ligne de séparation passé / futur
    ax.axvline(x=last_date, color='red', linestyle=':', label='Aujourd\'hui')

    ax.set_xlabel("Date")
    ax.set_ylabel("Ventes prévues")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    daily_pred['Jour'] = range(1, len(daily_pred) + 1)
    daily_pred['date'] = daily_pred['date'].dt.strftime('%Y-%m-%d')

    # Tableau avec couleur rouge si pic
    moyenne_prev = daily_pred['prediction'].mean()
    def couleur_pic(ligne):
        couleur = "background-color: #ffd6d6" if ligne['Prévision ventes totales'] > moyenne_prev * 1.2 else ""
        return [couleur] * len(ligne)

    df_affichage = daily_pred[['Jour', 'date', 'prediction']].rename(
        columns={'date': 'Date', 'prediction': 'Prévision ventes totales'}
    ).round(0)
    st.dataframe(df_affichage.style.apply(couleur_pic, axis=1), use_container_width=True)

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

    # Métriques en haut
    c1, c2 = st.columns(2)
    c1.metric(f"MAE pour {store}",  f"{mae_par_magasin[store]}")
    c2.metric("MAE moyen tous magasins", f"{mae_par_magasin.mean():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ['red' if s == store else 'steelblue' for s in df_mae['Magasin']]
    ax.bar(df_mae['Magasin'], df_mae['MAE'], color=colors, edgecolor='black')
    ax.set_ylabel("MAE")
    ax.set_title("MAE par magasin (rouge = magasin sélectionné)")
    st.pyplot(fig)

# VUE 4 : ALERTES ÉVÉNEMENTS
elif vue == "Alertes événements":
    st.subheader("Événements à venir — mai/juin 2016")

    store_recent        = df_long[df_long['store_id'] == store].copy()
    daily_recent        = store_recent.groupby('date')['sales'].sum()
    moyenne_journaliere = daily_recent.mean()

    # Métriques en haut
    c1, c2 = st.columns(2)
    c1.metric("Ventes moyennes / jour", f"{moyenne_journaliere:.0f}")
    c2.metric("Nombre d'événements à venir", "2")

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
        impact_pct             = info['impact'].get(store, 0)
        unites_supplementaires = int(moyenne_journaliere * abs(impact_pct) / 100)

        if impact_pct > 0:
            st.success(
                f"**{event}** le {info['date']} --> "
                f"Hausse prévue pour {store} : **+{impact_pct}%** "
                f"soit environ **+{unites_supplementaires} unités** ce jour-là"
            )
            st.info(
                f"Recommandation : {info['action']} --> "
                f"Commander environ **{unites_supplementaires} unités supplémentaires**"
            )
        else:
            st.error(
                f"**{event}** le {info['date']} --> "
                f"Baisse prévue pour {store} : **{impact_pct}%** "
                f"soit environ **-{unites_supplementaires} unités** ce jour-là"
            )
            st.info(
                f"Recommandation : Ne pas sur-stocker --> "
                f"Réduire les commandes d'environ **{unites_supplementaires} unités**"
            )