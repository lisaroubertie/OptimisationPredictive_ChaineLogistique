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
|-- notebooks/
|    |-- Essais/          # contient des essais que nous ne voulions pas pour autant supprimer
|    |
|    |-- ExplorationDonnees/
|    |    |-- Global/
|    |    |    |-- AnalyseProduit.ipynb
|    |    |    |-- SaisonaliteTendances.ipynb
|    |    |    |-- AnalyseGlobale.ipynb
|    |    |
|    |    |-- Magasins/
|    |    |-- Produits/
|    |
|    |-- Modeles/
|    |    |-- Global/
|    |    |    |-- ModelePrevisionDemandeClassificationFinal.ipynb
|    |    |    |-- ModelePrevisionDemandeRegressionFinal.ipynb
|    |    |    |-- modele_classification.pkl
|    |    |    |-- seuil_pic_classification.pkl
|    |    |    |-- features_regression.pkl
|    |    |    |-- modele_regression.pkl
|    |    |    |-- rmse_regression.pkl
|    |    |    |-- 28derniersjours_regression.csv
|    |    |    |-- daily_regression.csv
|    |    |
|    |    |-- Magasins/
|    |    |-- Produits/
|
|-- Interface/
|    |-- app.py
|    |-- pages/
|    |    |-- 1_Page_Globale.py
|    |    |-- 2_Page_Magasins.py
|    |    |-- 3_Page_Produits.py
|
|-- README.md
```

## Correspondance notebooks / rapports
[A compléter]
| Notebooks | Partie de rapport |
|:-------- |--------:|
| AnalyseGlobale.ipynb | Rapport mi-projet - partie 7.1 |
| AnalyseProduit.ipynb | Rapport mi-projet - partie 7.2 |
| SaisonaliteTendances.ipynb | Rapport mi-projet - partie 7.4 |
| ModelePrevisionDemandeRegressionFinal.ipynb | Rapport final - partie 5.2 |
| ModelePrevisionDemandeClassificationFinal.ipynb | Rapport final - partie 5.3 |

## Dataset
Le dataset étant trop lourd, nous l'avons chacune enregistré en local dans nos ordinateur. Il est disponible à l'adresse : [https://www.kaggle.com/competitions/m5-forecasting-accuracy]
Les fichiers nécessaires au lancement de l'application sont néanmoins enregistrés dans le repository, dans les dossiers avec les notebooks correspondants.

## Lancer l'application
```bash
cd interface
streamlit run app.py
```
