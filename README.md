# 🏢 Cyber-Physical System (CPS) - Dashboard d'Analyse du Bâtiment

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

Ce projet est une application web interactive développée avec **Streamlit**. Elle fait office de **Système Cyber-Physique (CPS)** dédié à la supervision, la validation de données multi-capteurs, et l'optimisation énergétique (confort et gaspillage) d'un bâtiment intelligent.

L'application intègre une approche hybride combinant des règles physiques (**Théorie de Dempster-Shafer**) et des modèles statistiques (**Isolation Forest**) pour monitorer les étages sur plusieurs années.

---

## 📑 Sommaire

- [Fonctionnalités Principales](#fonctionnalites-principales)
- [Structure du Projet](#structure-du-projet)
- [Installation et Lancement Local](#installation-et-lancement-local)
- [Déploiement](#deploiement)
- [Technologies Utilisées](#technologies-utilisees)
- [Modules du Projet](#modules-du-projet)

---

<a id="fonctionnalites-principales"></a>
## 📊 Fonctionnalités Principales

* **Nettoyage Automatisé** : Détection et élimination à la volée des lignes corrompues ou des données parasites dans les fichiers sources.
* **Analyse d'Incertitude (DST)** : Fusion des données de Température et d'Humidité via la théorie des fonctions de croyance (Dempster-Shafer) pour évaluer le Confort Global (BetP) et le niveau de Conflit (K) entre capteurs.
* **Validation Croisée IA** : Comparaison en temps réel entre l'approche DST et un modèle *Isolation Forest* pour valider la convergence des diagnostics d'inconfort.
* **Indicateur d'Efficacité Énergétique (ECMI)** : Évaluation fine de la sur-climatisation et calcul des pertes thermiques évitables (gaspillage en kWh).
* **Moteur Prédictif (+4h)** : Anticipation des dérives d'ambiance à court terme pour permettre un ajustement préventif des consignes de la GTB.
* **Génération de Rapport PDF** : Exportation instantanée et allégée d'un bilan de performance au format PDF, prêt pour le déploiement Cloud.

---

<a id="structure-du-projet"></a>
## 📁 Structure du Projet

```text
📁 ton-projet-cps/
├── 📁 data/                  # Fichiers de données temporelles
│   ├── 📁 raw/               # Fichiers CSV d'origine (ex: 2019Floor6.csv)
│   └── 📄 cleaned_...csv     # Fichiers nettoyés après traitement
├── 📄 .gitignore             # Exclusion des fichiers temporaires (__pycache__, .pyc)
├── 📄 app.py                 # Code principal de l'interface Streamlit (Dashboard)
├── 📄 dst_engine.py          # Moteur de calculs mathématiques et physiques (DST, IF, ECMI)
├── 📄 rapport.py             # Script de génération dynamique du rapport FPDF2
├── 📄 README.md              # Documentation du projet
└── 📄 requirements.txt       # Dépendances Python requises pour l'application
```

---

<a id="installation-et-lancement-local"></a>
## 🚀 Installation et Lancement Local

### 1. Prérequis

Assurez-vous d'avoir **Python 3.9** ou supérieur installé sur votre machine.

```bash
python --version
```

### 2. Cloner le projet & Installer les dépendances

Ouvrez votre terminal à la racine du projet et exécutez la commande suivante pour installer les bibliothèques requises :

```bash
git clone https://github.com/votre-utilisateur/ton-projet-cps.git
cd ton-projet-cps
pip install -r requirements.txt
```

### 3. Exécuter l'application

Pour lancer le serveur local Streamlit, exécutez :

```bash
python -m streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse [http://localhost:8501](http://localhost:8501).

---

<a id="deploiement"></a>
## 🌐 Déploiement

Ce projet est configuré pour être déployé en un clic sur **Streamlit Community Cloud** :

1. Déposez le code sur un dépôt GitHub (le fichier `.gitignore` est déjà configuré pour masquer le dossier `__pycache__`).
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io/).
3. Sélectionnez votre dépôt et lancez le déploiement avec le fichier principal `app.py`.

---

<a id="technologies-utilisees"></a>
## 🛠️ Technologies Utilisées

| Catégorie | Technologie |
|---|---|
| Interface Graphique | **Streamlit** |
| Analyses & Data Science | **Pandas**, **NumPy**, **Scikit-Learn** (Isolation Forest) |
| Visualisation Spatio-temporelle | **Plotly** (Subplots & Double Axe Y) |
| Édition de Rapports | **FPDF2** (Génération PDF binaire à la volée) |
| Fondements théoriques | Théorie de Dempster-Shafer, Détection d'anomalies |

---

<a id="modules-du-projet"></a>
## 🧩 Modules du Projet

| Fichier | Rôle |
|---|---|
| `app.py` | Interface principale Streamlit : navigation, affichage des graphiques et interactions utilisateur. |
| `dst_engine.py` | Cœur analytique : calculs de fusion de croyances (DST), Isolation Forest, calcul de l'ECMI et du module prédictif. |
| `rapport.py` | Génération du rapport PDF synthétique (FPDF2) à partir des résultats calculés par `dst_engine.py`. |