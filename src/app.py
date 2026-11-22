"""Servidor de inferência FastAPI para o modelo MLP de churn."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

try:  # execução como pacote (fastapi dev src/app.py)
    from .schemas import ClienteEntrada, PredicaoSaida
except ImportError:  # fallback caso o módulo seja carregado diretamente
    from schemas import ClienteEntrada, PredicaoSaida

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "pipeline_rede_neural.joblib"

ROTULOS = {0: "Permanece (Sem churn)", 1: "Cancelamento (Churn)"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Único ponto de leitura em disco: startup da aplicação.
    app.state.modelo = joblib.load(CAMINHO_MODELO)
    yield
    app.state.modelo = None


app = FastAPI(
    title="API de Predição de Churn (MLP)",
    description="Inferência com Pipeline Scikit-Learn (ColumnTransformer + MLPClassifier).",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo")
def diagnostico_modelo():
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


@app.post("/predict", response_model=PredicaoSaida, status_code=200)
def predict(entrada: ClienteEntrada, request: Request):
    modelo = request.app.state.modelo  # já carregado no lifespan

    df = pd.DataFrame([entrada.model_dump()])
    probabilidades = modelo.predict_proba(df)[0]
    indice = int(probabilidades.argmax())
    classe = int(modelo.classes_[indice])

    return PredicaoSaida(
        categoria_prevista=ROTULOS.get(classe, str(classe)),
        score_confianca=round(float(probabilidades[indice]), 4),
    )
