# Projet Ingé2 - Optimisation prédictive de la chaine logistique en intégrant l'apprentissage automatique et la recherche opérationnelle


## Description
Ce projet a pour objectif de prédire la demande et détecter les ruptures de stocks à partir du dataset M5 (Walmart).
Notre rendu final est une application Streamlit combinant régression et classification à l'échelle globale, puis plus spécifique en analysant les magasins et les produits.


## Etudiantes
- Lisa ROUBERTIE
- Maëlys BOUAZZA
- Marie-Noëlle ANTON-GEOFFRY


## Référent de projet
Alessandro LEITE


## Architecture du repository
```
|-- datas/                # les fichiers .csv Kaggle doivent être placés dans ce dossier
|
|-- notebooks/
|    |-- Essais/          # archives de travail
|    |
|    |-- ExplorationDonnees/
|    |    |-- Global/
|    |    |    |-- AnalyseProduitsCategories.ipynb
|    |    |    |-- SaisonaliteTendances.ipynb
|    |    |    |-- AnalyseGlobale.ipynb
|    |    |
|    |    |-- Magasins/
|    |    |    |-- Comparaison exploration par magasin VS général.ipynb
|    |    |    |-- Exploration données ventes PAR MAGASIN.ipynb
|    |    |
|    |    |-- Produits/
|    |    |    |-- Analyse d'un Produit.ipynb
|    |    |
|    |
|    |-- Modeles/
|    |    |-- Global/
|    |    |    |-- ModelePrevisionDemandeClassificationFinal.ipynb
|    |    |    |-- ModelePrevisionDemandeRegressionFinal.ipynb
|    |    |    |-- modele_classification.pkl          # sauvegarde du modèle de classification (Streamlit)
|    |    |    |-- seuil_pic_classification.pkl          # pic du modèle binaire (Streamlit)
|    |    |    |-- features_regression.pkl          # sauvegarde des features (Streamlit)
|    |    |    |-- modele_regression.pkl          # sauvegarde du modèle de régression (Streamlit)
|    |    |    |-- rmse_regression.pkl          # score RMSE pour l'interval de confiance (Streamlit)
|    |    |    |-- 28derniersjours_regression.csv         # pour le graphe de prévision (Streamlit)
|    |    |    |-- daily_regression.csv          # pour le graphe d'historique (Streamlit)
|    |    |
|    |    |-- Magasins/
|    |    |    |-- # à compléter
|    |    |
|    |    |-- Produits/
|    |    |    |-- # à compléter
|
|-- Interface/
|    |-- app.py
|    |-- pages/
|    |    |-- 1_Page_Globale.py
|    |    |-- 2_Page_Magasins.py
|    |    |-- 3_Page_Produits.py
|
|-- README.md
|-- requirements.txt
```


## Correspondance notebooks / rapports

### Partie globale
| Notebooks | Partie de rapport |
|:-------- |--------:|
| AnalyseGlobale.ipynb | Rapport mi-projet - partie 7.1 |
| AnalyseProduitsCategories.ipynb | Rapport mi-projet - partie 7.2 |
| SaisonaliteTendances.ipynb | Rapport mi-projet - partie 7.4 |
| ModelePrevisionDemandeRegressionFinal.ipynb | Rapport final - partie 6.2 |
| ModelePrevisionDemandeClassificationFinal.ipynb | Rapport final - partie 6.3 |


### Partie par magasins
[A compléter avec les notebooks de modélisation]
| Notebooks | Partie de rapport |
|:-------- |--------:|
| Comparaison exploration par magasin VS général.ipynb | Rapport mi-projet - partie 7.3 |
| Exploration données ventes PAR MAGASIN.ipynb | Rapport mi-projet - partie 7.3 |


### Partie par produit
[A compléter avec les notebooks de modélisation]
| Notebooks | Partie de rapport |
|:-------- |--------:|
| Analyse d'un Produit.ipynb | Rapport mi-projet - partie 7.5 |


## Dataset
Le dataset étant trop lourd, nous l'avons chacune enregistré en local dans nos ordinateur (chemin : OptimisationPredictive_ChaineLogistique/datas). Il est disponible à l'adresse : [https://www.kaggle.com/competitions/m5-forecasting-accuracy]

Les fichiers nécessaires au lancement de l'application (.pkl et .csv) sont néanmoins enregistrés dans le repository avec les notebooks correspondants, pour que l'application puisse tourner sans relancer les modèles.


## Installation
1. Cloner le dépôt
2. Télécharger le dataset (chemin : OptimisationPredictive_ChaineLogistique/datas)
3. Installer les dépendances :
```bash
   pip install -r requirements.txt
```

## Lancer l'application
```bash
cd interface
streamlit run app.py
```
