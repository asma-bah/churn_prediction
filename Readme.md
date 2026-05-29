# 📊 Prédiction du départ des clients (Churn)

Modèle de machine learning qui prédit si un client d'un opérateur télécom 
risque de résilier son abonnement, exposé via une API et déployé en ligne.

## 🔗 Démo en ligne

L'API est accessible ici : **<https://esme-bba-churn-prediction-api.hf.space/docs>**

(Interface interactive : on peut tester une prédiction directement depuis le navigateur.)

## 🎯 Le projet

À partir des données de 7043 clients que j'ai téléchargé sur le site Kaggle, le modèle apprend à repérer ceux qui sont sur le point de partir, afin que l'entreprise puisse agir pour les retenir.

Le projet couvre toute la chaîne :
- exploration et nettoyage des données
- entraînement et comparaison de plusieurs modèles
- mise en place d'une API
- conteneurisation avec Docker
- déploiement en ligne

## 📈 Résultat

Comme les clients qui partent sont minoritaires (27%), j'ai évalué le modèle 
sur sa capacité à les repérer (recall) plutôt que sur la précision globale.

Modèle retenu : **régression logistique** (avec gestion du déséquilibre des classes)  
→ **recall de 0,79** sur les clients qui partent (le modèle en repère ~8 sur 10).

## 🛠️ Technologies 

Python · pandas · scikit-learn · FastAPI · Docker · Hugging Face Spaces


## 📁 Fichiers

- `exploration.ipynb` — analyse, nettoyage et entraînement du modèle
- `app.py` — l'API (routes `/predict` et `/predict_batch`)
- `churn_model.pkl` — le modèle entraîné
- `Dockerfile` — configuration du conteneur
- `requirements.txt` — dépendances


## 🚀 Lancer en local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Puis ouvrir http://127.0.0.1:8000/docs

## 👤 Auteur

Asmaou Bah — <Github : https://github.com/asma-bah / Linkedin:www.linkedin.com/in/asmaou-bah >