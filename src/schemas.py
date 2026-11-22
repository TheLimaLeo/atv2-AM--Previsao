"""Contratos de dados (Pydantic) da API de predição de churn."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClienteEntrada(BaseModel):
    """Atributos brutos de um cliente de telecom (mesmos nomes do dataset)."""

    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Meses de permanência do cliente na empresa.",
        examples=[12],
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Valor da mensalidade (US$).",
        examples=[70.35],
    )
    TotalCharges: float = Field(
        ...,
        ge=0.0,
        le=20000.0,
        description="Valor total já cobrado do cliente (US$).",
        examples=[840.5],
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato.", examples=["Month-to-month"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de serviço de internet.", examples=["Fiber optic"]
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Forma de pagamento.", examples=["Electronic check"])
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Fatura digital (sem papel)?", examples=["Yes"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui suporte técnico?", examples=["No"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui segurança online?", examples=["No"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 2,
                "MonthlyCharges": 95.7,
                "TotalCharges": 191.4,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "PaperlessBilling": "Yes",
                "TechSupport": "No",
                "OnlineSecurity": "No",
            }
        }
    )


class PredicaoSaida(BaseModel):
    """Resposta da predição."""

    categoria_prevista: str = Field(
        ..., description="Classe inferida pelo modelo.", examples=["Cancelamento (Churn)"]
    )
    score_confianca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade (predict_proba) da classe prevista.",
        examples=[0.87],
    )
