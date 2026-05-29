from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

"""#on crée une application
app = FastAPI(title="API de prédiction du churn")

#on definit une premiere "route" : la page d'accueil
@app.get("/")
def accueil():
    return {"message": "Bienvenue sur l'API de prédiction du churn ! "}

"""

#on charge les données que l'on a sauvegardé 
pipeline = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

#creation de l'api
app = FastAPI(title="API de prédiction du churn")

#description des infos que l'on attend d'un client, avec le type de chaque champ
class Client(BaseModel):
    gender : str
    SeniorCitizen: int
    Partner: str
    Dependents: str 
    tenure: int 
    PhoneService: str 
    MultipleLines: str
    InternetService: str 
    OnlineSecurity: str
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingTV: str
    StreamingMovies: str 
    Contract: str 
    PaperlessBilling: str
    PaymentMethod: str 
    MonthlyCharges: float    
    TotalCharges: float


#la page d'accueil
@app.get("/")
def accueil():
    return {"message": "Bienvenu sur l'API de prédiction du départ ou non des clients d'une société de téléphonie !"}


#page de prédiction 
@app.post("/predict")
def predire(client: Client):
    #mettre les infos du client dans un tableau d'une seule ligne
    donnees = pd.DataFrame([client.model_dump()])

    #
    donnees = pd.get_dummies(donnees)

    #alignement sur 30 colonnes (comme sur le modele)
    # les colonnes manquantes sont ajoutées et remplis de 0 
    donnees = donnees.reindex(columns=model_columns, fill_value=0)

    #predictions sur les jeux de données fournis par le client
    prediction = pipeline.predict(donnees)[0]  # 0 -> reste et 1 -> part
    probabilite = pipeline.predict_proba(donnees)[0][1]

    #renvoyer un resultat lisible
    return {
        "prediction": "Risque de départ " if prediction == 1 else "Va probablement rester",
        "probabilité depart" :  round(float(probabilite), 2)
    }