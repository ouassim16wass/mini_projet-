# Mini projet MLOps — Wine Classifier

**Auteur :** Megrad Ouassim
**Dataset :** Wine (`sklearn.datasets.load_wine`) — 178 échantillons, 13 features chimiques, 3 classes de cépages
**Stack :** Python 3.11, scikit-learn, FastAPI, Docker, GitHub Actions, GHCR

## Structure

```
.
├── app/
│   ├── train.py       # Entraînement + sauvegarde model.pkl
│   └── api.py         # FastAPI: /health + /predict
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Dataset

Wine Dataset intégré dans scikit-learn. Aucun fichier à télécharger.

- **Échantillons :** 178
- **Features (13) :** alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315, proline
- **Classes (3) :** class_0, class_1, class_2 (cultivars de vins italiens)

## Lancer en local

```bash
pip install -r requirements.txt
python -m app.train
uvicorn app.api:app --reload
```

API sur `http://localhost:8000` — doc interactive sur `/docs`.

## Lancer avec Docker

```bash
docker build -t wine-api .
docker run -p 8000:8000 wine-api
```

## Endpoints

### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14,
    "alcalinity_of_ash": 11.2, "magnesium": 100, "total_phenols": 2.65,
    "flavanoids": 2.76, "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28,
    "color_intensity": 4.38, "hue": 1.05, "od280_od315": 3.4, "proline": 1050
  }'
```
Réponse :
```json
{"prediction": 0, "label": "class_0"}
```

## Pipeline CI

- **Push sur `feature/**`** → installe les dépendances + entraîne le modèle.
- **Push sur `develop`** → entraîne + build l'image Docker + publie sur `ghcr.io/ouassim16wass/wine-api`.
