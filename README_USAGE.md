# Guide de l'Utilisateur - XGBoost Congestion Pipeline

Ce projet permet de prédire le niveau de congestion des ports en utilisant un modèle XGBoost. 

## 1. Installation
Avant de commencer, installez les dépendances nécessaires :
```bash
pip install -r requirements.txt
```

## 2. Structure du Projet
- **`train_xgboost.py`**: Script pour réentraîner le modèle sur vos données originales.
- **`xgboost_congestion_model.json`**: Le modèle entraîné (votre "cerveau" IA).
- **`generate_test_dataset.py`**: Génère un dataset synthétique réaliste pour tester le modèle.
- **`run_evaluation.py`**: Évalue le modèle sur un fichier CSV et génère un rapport de performance.

## 2. Comment tester avec une NOUVELLE (Dataset) ?

Si vous avez un nouveau fichier CSV (ex: `ma_nouvelle_data.csv`), voici comment faire :

1. **Préparez votre CSV** : Assurez-vous qu'il contient les colonnes suivantes :
   - `hours`
   - `fishing_hours`
   - `mmsi_present`
   - `congestion_ratio` (calculé comme `fishing_hours / hours`)
   - `true_label` (Optionnel, si vous voulez voir la précision)

2. **Lancez l'évaluation** :
   Ouvrez `run_evaluation.py` et modifiez la dernière ligne :
   ```python
   evaluate_model("xgboost_congestion_model.json", "ma_nouvelle_data.csv")
   ```
   Ou lancez-le en ligne de commande :
   ```bash
   python run_evaluation.py
   ```

3. **Consultez les résultats** :
   Le script générera un fichier `evaluation_report.txt` et une image `confusion_matrix.png` montrant la performance.

## 3. Pour générer un nouveau test
Si vous voulez juste créer un nouveau fichier de test aléatoire :
```bash
python generate_test_dataset.py
```
Cela créera `test_dataset_large.csv`.
