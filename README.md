# AT2 — Redes Neurais MLP e Deploy de API em Produção

Previsão de cancelamento de clientes (**churn**) de uma operadora de telecom com uma
**MLP (MLPClassifier)** encapsulada em um `Pipeline` do Scikit-Learn e servida por uma
API REST com **FastAPI**.

- **Dataset (Kaggle):** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Problema:** classificação binária (`Churn`: 0 = permanece, 1 = cancela)
- **Features (9):** `tenure`, `MonthlyCharges`, `TotalCharges` (numéricas) e
  `Contract`, `InternetService`, `PaymentMethod`, `PaperlessBilling`, `TechSupport`, `OnlineSecurity` (categóricas)
- **Modelo:** `ColumnTransformer` (StandardScaler + OneHotEncoder) → `MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=77)`

## Estrutura

```
am-at2-redes-neurais/
├── data/dataset.csv                     # CSV original do Kaggle (você adiciona)
├── models/pipeline_rede_neural.joblib   # Pipeline serializado (gerado pelo notebook)
├── notebooks/analise_treinamento.ipynb  # EDA, treinamento e métricas
├── src/
│   ├── __init__.py
│   ├── schemas.py                       # Contratos Pydantic (entrada/saída)
│   └── app.py                           # FastAPI + lifespan + rotas
├── pyproject.toml                       # Dependências (UV)
├── requisicoes.http                     # Testes HTTP
└── README.md
```

## Como reproduzir

Pré-requisito: [UV](https://docs.astral.sh/uv/) instalado.

```bash
# 1. Instalar dependências
uv sync

# 2. Baixar o dataset no Kaggle e salvar como data/dataset.csv

# 3. Executar o notebook notebooks/analise_treinamento.ipynb (Run All)
#    usando o kernel do ambiente .venv criado pelo uv.
#    Isso gera models/pipeline_rede_neural.joblib

# 4. Subir a API
uv run fastapi dev src/app.py
```

- Swagger UI: http://127.0.0.1:8000/docs
- Testes HTTP: abra `requisicoes.http` (extensão REST Client ou Thunder Client).

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/diagnostico/modelo` | Status do serviço e metadados do modelo |
| POST | `/predict` | Recebe os dados do cliente e retorna `categoria_prevista` e `score_confianca` |

Exemplo de resposta de `/predict`:

```json
{
  "categoria_prevista": "Cancelamento (Churn)",
  "score_confianca": 0.87
}
```

Entradas inválidas (tipos errados, valores fora dos limites ou categorias inexistentes)
retornam **HTTP 422** automaticamente via Pydantic.

## Decisões de projeto

- O pipeline **completo** é serializado, então a API recebe dados brutos e aplica o mesmo pré-processamento do treino.
- O modelo é carregado **uma única vez** no `lifespan` (startup) e guardado em `app.state.modelo`; nenhuma rota lê disco.
- Use a mesma versão do `scikit-learn` no treino e na API (garantido ao usar o mesmo ambiente `uv`).

## Vídeo de demonstração

Link: _(cole aqui o link do vídeo não listado/público)_
