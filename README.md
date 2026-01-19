# 🚗 Car Insurance Claim Predictor
> **Optimiser l'évaluation des risques et la tarification d'assurance grâce au Machine Learning.**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)

## 📖 Contexte & Motivation
Dans le secteur de l'assurance automobile, la capacité à prédire avec précision la probabilité de sinistre d'un conducteur est cruciale pour la rentabilité et l'équité des tarifs.
Ce projet de Data Science vise à **développer un modèle prédictif** capable de classifier les clients selon leur risque de sinistre (*outcome*), en se basant sur des données démographiques et historiques (âge, expérience, infractions, etc.). L'objectif final est de permettre une tarification plus juste et une meilleure gestion des risques.

## 🏗️ Aperçu Technique
Le projet suit une architecture classique de **Pipeline de Science des Données** :
1.  **Ingestion & Nettoyage** : Chargement des données brutes et traitement des valeurs manquantes.
2.  **EDA (Exploratory Data Analysis)** : Analyse statistique et visuelle pour comprendre les facteurs de risque.
3.  **Preprocessing** : Encodage des variables catégorielles et normalisation des données numériques.
4.  **Modélisation & Évaluation** : Entraînement de plusieurs algorithmes (Régression Logistique, KNN) et comparaison de leurs performances via validation croisée.

## ✨ Fonctionnalités Clés
*   **Analyse Exploratoire Approfondie** : Visualisation des corrélations entre les profils conducteurs (âge, genre, score de crédit) et la fréquence des accidents.
*   **Pipeline de Prétraitement Robuste** : Gestion intelligente des données manquantes (notamment sur `credit_score` et `annual_mileage`) et encodage des variables ordinales.
*   **Comparaison Multimodèle** : Mise en compétition de plusieurs classifieurs (Logistic Regression, KNN, Perceptron) pour identifier le plus performant.
*   **Validation Croisée Rigoureuse** : Utilisation de K-Fold Cross-Validation pour assurer la généralisation du modèle et éviter le surapprentissage.

## 🛠️ Stack Technique

| Catégorie | Technologies |
| :--- | :--- |
| **Langage & Environnement** | Python, Jupyter Notebook |
| **Manipulation de Données** | Pandas, NumPy, DuckDB |
| **Machine Learning** | Scikit-learn (Model Selection, Preprocessing, Linear Models) |
| **Visualisation** | Matplotlib |
| **Format de Données** | CSV |

## 🚀 Installation & Usage

Pour explorer le projet et lancer les modèles localement :

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/car-insurance-predictor.git
cd car-insurance-predictor

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le Notebook
jupyter notebook car_insurance_project.ipynb
```

## 🧠 Challenge & Apprentissage

Un défi technique majeur de ce projet a été la **gestion des données manquantes** sur des features critiques comme le `credit_score` et le `annual_mileage`.
*   **Problème** : Supprimer simplement les lignes incomplètes aurait réduit significativement la taille du dataset et potentiellement introduit un biais.
*   **Solution** : Une analyse a été menée pour décider de la meilleure stratégie d'imputation (remplacement par la moyenne/médiane) en fonction de la distribution des données, préservant ainsi l'intégrité statistique du dataset pour l'entraînement des modèles. De plus, le choix des métriques d'évaluation (Accuracy vs Precision/Recall) a été déterminant pour ne pas simplement maximiser la précision globale au détriment de la détection des "vrais" sinistres.
