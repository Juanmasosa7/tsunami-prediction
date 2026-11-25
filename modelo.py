"""Las piezas del modelo de la Entrega 3, en un solo lugar.

Las usan build_app.py (para entrenar y calcular las predicciones) y app.py (para cargar el modelo
guardado). Es el mismo código que el notebook de la Entrega 3: si se cambia algo acá, hay que
cambiarlo también allá, o la aplicación deja de coincidir con el notebook.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

SEMILLA = 42                 # una sola semilla, pasada a todo lo que la acepta
CORTE = "2026-04-01"         # desde abril de 2026 la USGS todavía no cargó la marca de tsunami
UMBRAL = 0.27                # elegido por validación cruzada en la Entrega 3 (sección 8.2)

# Las columnas que recibe el pipeline (las candidatas de la Entrega 3).
# El pipeline usa sólo mag, depth, latitude, longitude, magType y el signo de dist_costa_km;
# nst, dmin, gap y rms se descartaron en la ablación y el ColumnTransformer las ignora.
CANDIDATAS = ["mag", "depth", "latitude", "longitude", "magType", "dist_costa_km",
              "nst", "dmin", "gap", "rms"]
NUMERICAS = ["mag", "depth", "latitude", "longitude"]
CATEGORICAS = ["magType"]
BINARIAS = ["es_oceanico"]

# Hiperparámetros ganadores del GridSearchCV de la Entrega 3 (sección 7)
HIPERPARAMETROS = {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "class_weight": None}


class EsOceanico(BaseEstimator, TransformerMixin):
    """Agrega es_oceanico (1 si el epicentro está en el mar, 0 si está en tierra), a partir del signo de dist_costa_km."""

    def fit(self, X, y=None):
        self.columnas_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        X["es_oceanico"] = (X["dist_costa_km"] > 0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columnas_ + ["es_oceanico"])


def armar(clasificador, numericas=NUMERICAS, categoricas=CATEGORICAS, binarias=BINARIAS, escalar=True):
    """El pipeline completo: derivar, imputar, escalar, codificar y clasificar, en un solo objeto."""
    pasos_num = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    preparar = ColumnTransformer([
        ("numericas", Pipeline(pasos_num), list(numericas)),
        ("categoricas", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                                  ("codificar", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]),
         list(categoricas)),
        ("binarias", "passthrough", list(binarias))])
    return Pipeline([("derivar", EsOceanico()), ("preparar", preparar), ("clasificador", clasificador)])


def modelo_elegido():
    """El gradient boosting elegido en la Entrega 3, con sus hiperparámetros ajustados."""
    return armar(HistGradientBoostingClassifier(random_state=SEMILLA, **HIPERPARAMETROS))


def grupos_de(datos):
    """Grupo = celda de 2° x 2° y mes: las réplicas de una secuencia caen en el mismo grupo."""
    return (np.floor(datos["latitude"] / 2).astype(int).astype(str) + "_" +
            np.floor(datos["longitude"] / 2).astype(int).astype(str) + "_" +
            datos["time"].dt.strftime("%Y-%m"))


def validacion():
    """La misma validación de la Entrega 3: 5 pliegues por grupo, estratificados."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def caso(mag, depth, latitude, longitude, magType, en_el_mar):
    """Arma la fila que espera el pipeline para un sismo cargado a mano.

    El modelo sólo usa el signo de dist_costa_km (es_oceanico), así que alcanza con +1 (mar) o -1 (tierra).
    Las columnas descartadas van vacías.
    """
    return pd.DataFrame([{"mag": mag, "depth": depth, "latitude": latitude, "longitude": longitude,
                          "magType": magType, "dist_costa_km": 1.0 if en_el_mar else -1.0,
                          "nst": np.nan, "dmin": np.nan, "gap": np.nan, "rms": np.nan}])[CANDIDATAS]
