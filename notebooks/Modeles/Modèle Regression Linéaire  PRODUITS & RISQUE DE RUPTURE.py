#!/usr/bin/env python
# coding: utf-8

# # Modèle Regression Linéaire : PRODUITS & RISQUE DE RUPTURE

# Dans cette partie, on cherche à répondre à des questions très concrètes :
# 
# - Quels produits vont avoir une forte demande dans les prochains jours ?
# - Quels produits risquent une rupture de stock ?
# - Quels produits faut-il prioriser en réapprovisionnement ?
# 
# L’objectif est d’aider Sophie Martin à éviter les ruptures et mieux gérer les stocks au niveau produit.

# ## IMPORTS + DONNÉES : 

# On commence par importer les bibliothèques nécessaires pour le projet comme pandas, numpy ou encore scikit-learn. Ensuite, on charge les deux jeux de données principaux : les ventes et le calendrier. Les données sont ensuite transformées pour obtenir une structure exploitable, où chaque ligne correspond à une vente associée à une date, un produit et un magasin.

# In[7]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("sales_train_evaluation.csv")
calendar = pd.read_csv("calendar.csv")

df = df.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="sales"
)

df = df.merge(calendar, on="d", how="left")
df["date"] = pd.to_datetime(df["date"])


# ## DATASET PRODUIT : 

# On construit ensuite un dataset plus précis au niveau produit. On regroupe les ventes par date, produit et magasin afin d’avoir une vision détaillée de chaque article dans chaque point de vente. On ajoute aussi des informations comme les promotions ou les événements pour mieux expliquer les variations de la demande.

# In[ ]:


product_daily = df.groupby(
    ["date", "item_id", "store_id", "cat_id", "dept_id", "sell_price"]
).agg(
    sales=("sales", "sum"),
    has_event=("event_name_1", lambda x: int(x.notna().any())),
    snap_CA=("snap_CA", "max"),
    snap_TX=("snap_TX", "max"),
    snap_WI=("snap_WI", "max")
).reset_index()


# ## FEATURES TEMPORELLES :

# On ajoute des variables temporelles pour aider le modèle à comprendre l’évolution des ventes dans le temps. On crée par exemple les ventes de la semaine précédente (lag 7) ou encore la moyenne des ventes récentes (rolling mean). Cela permet de capturer les tendances et les habitudes de consommation.

# In[ ]:


product_daily = product_daily.sort_values(["item_id", "store_id", "date"])

product_daily["lag_7"] = product_daily.groupby(
    ["item_id", "store_id"]
)["sales"].shift(7)

product_daily["rolling_mean_7"] = product_daily.groupby(
    ["item_id", "store_id"]
)["sales"].shift(1).rolling(7).mean()


# ## PRIX :

# On ajoute ensuite une variable liée au prix afin de voir son impact sur les ventes. Le price_ratio permet de comparer le prix d’un produit à la moyenne générale, ce qui aide à comprendre si un produit est plus cher ou moins cher que la normale.

# In[2]:


product_daily["price_ratio"] = (
    product_daily["sell_price"] / product_daily["sell_price"].mean()
)


# ## ENCODAGE VARIABLES :

# Les variables catégorielles comme les produits, les magasins ou les catégories sont transformées en valeurs numériques grâce à un encodage. Cela est nécessaire pour que le modèle puisse les utiliser correctement dans ses calculs.

# In[ ]:


le_item = LabelEncoder()
le_store = LabelEncoder()
le_cat = LabelEncoder()

product_daily["item_id_enc"] = le_item.fit_transform(product_daily["item_id"])
product_daily["store_id_enc"] = le_store.fit_transform(product_daily["store_id"])
product_daily["cat_id_enc"] = le_cat.fit_transform(product_daily["cat_id"])


# ## NETTOYAGE :

# On supprime les lignes avec des valeurs manquantes afin d’éviter les erreurs lors de l’entraînement du modèle. Cela permet de garder uniquement des données propres et exploitables.

# In[ ]:


product_daily = product_daily.dropna()


# ## FEATURES :

# On sélectionne ensuite les variables qui vont servir au modèle. On garde à la fois les informations sur les produits, les magasins, le contexte (prix, promotions) et l’historique des ventes. L’objectif est de donner au modèle le maximum d’informations utiles sans le complexifier inutilement.

# In[3]:


features = [
    "item_id_enc",
    "cat_id_enc",

    "store_id_enc",

    "sell_price",
    "price_ratio",
    "snap_CA", "snap_TX", "snap_WI",
    "has_event",

    "lag_7",
    "rolling_mean_7"
]

X = product_daily[features]
y = product_daily["sales"]


# ## TRAIN / TEST : 

# Les données sont séparées en deux parties : une pour l’entraînement et une pour le test. On respecte l’ordre chronologique pour éviter de mélanger le passé et le futur, ce qui est essentiel dans les données temporelles.

# In[ ]:


split = int(len(product_daily) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# ## MODELE : 

# On entraîne ensuite un modèle de régression linéaire pour prédire les ventes des produits. Les performances sont évaluées avec le RMSE afin de mesurer l’erreur entre les prédictions et les valeurs réelles.

# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE :", rmse)


# ## VISUALISATION :

# On affiche un graphique pour comparer les ventes réelles et les ventes prédites sur une période donnée. Cela permet de voir visuellement si le modèle suit bien les tendances.

# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Réel")
plt.plot(pred[:100], label="Prédit")
plt.title("Réel vs Prédit - Produits")
plt.legend()
plt.show()


# ## SCORE DE RISQUE : 

# On calcule ensuite un score de risque basé sur le rapport entre la prédiction et la moyenne des ventes passées. Cela permet de classer les produits en trois catégories : faible, moyen ou élevé risque de forte demande ou de rupture.

# In[ ]:


test = product_daily.iloc[split:].copy()
test["prediction"] = pred

test["risk_ratio"] = test["prediction"] / (test["rolling_mean_7"] + 1)

def risk_level(x):
    if x < 1.1:
        return "faible"
    elif x < 1.4:
        return "moyen"
    else:
        return "élevé"

test["risk"] = test["risk_ratio"].apply(risk_level)


# ## TOP 3 PRODUITS À RISQUE : 

# On identifie enfin les trois produits les plus à risque en fonction de leur score moyen. Cela permet de repérer les articles qui nécessitent une attention particulière en termes de stock.

# In[ ]:


top_risky = test.groupby("item_id").agg(
    avg_risk=("risk_ratio", "mean"),
    avg_prediction=("prediction", "mean")
).sort_values("avg_risk", ascending=False).head(3)

print("TOP 3 PRODUITS À RISQUE")
print(top_risky)


# ## EXPORT STREAMLIT : 

# Enfin, on entraîne un modèle final sur toutes les données et on l’exporte avec les features utilisées. Les résultats sont également sauvegardés pour être utilisés dans une application Streamlit et permettre une visualisation interactive.

# In[ ]:


model_final = LinearRegression()
model_final.fit(X, y)

joblib.dump(model_final, "model_product_risk.pkl")
joblib.dump(features, "features_product_risk.pkl")

test.to_csv("product_risk_output.csv", index=False)

print("Export terminé")

