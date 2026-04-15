# Mini projet MLOps — Iris Classifier

**Auteur :** Megrad Ouassim
**Dataset :** Iris (`sklearn.datasets.load_iris`)
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

## Lancer en local

```bash
pip install -r requirements.txt
python -m app.train
uvicorn app.api:app --reload
```

API sur `http://localhost:8000` — doc interactive sur `/docs`.

## Lancer avec Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
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
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
Réponse :
```json
{"prediction": 0, "label": "setosa"}
```

## Pipeline CI

- **Push sur `feature/**`** → installe les dépendances + entraîne le modèle.
- **Push sur `develop`** → entraîne + build l'image Docker + publie sur `ghcr.io/ouassim16wass/iris-api`.
