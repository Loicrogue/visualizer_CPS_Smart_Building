# Setup du projet FUTURE Dashboard

## Structure des fichiers

```
ton_projet/
├── app.py
├── requirements.txt
└── data/
    ├── cleaned_2019_Floor3.csv
    ├── cleaned_2019_Floor4.csv
    ├── cleaned_2019_Floor5.csv
    ├── cleaned_2019_Floor6.csv
    └── cleaned_2019_Floor7.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python -m streamlit run app.py
```

Puis ouvrir : http://localhost:8501

## Ajouter de nouvelles données

Déposer n'importe quel fichier `cleaned_YYYY_FloorX.csv` dans `data/`
et relancer — l'app le détecte automatiquement.

## Formats de noms acceptés

- `cleaned_2019_Floor3.csv`  ← format nettoyé (recommandé)
- `2019Floor3.csv`           ← format brut Kaggle (nettoyé à la volée)
