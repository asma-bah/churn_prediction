#Image de depart 
FROM python:3.11-slim 

#dossier de travail à l'interieur de la boite
WORKDIR /app

#la liste des dependances 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copier ensuite le reste du projet
COPY . .

#port sur lequel l'API va ecouter (ICI j'utilise Hugging face)
EXPOSE 7860

#commande lancée au démarrage de la boite 
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

