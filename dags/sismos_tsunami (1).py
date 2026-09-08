"""
DAG: sismos_tsunami
=====================
Proyecto: Análisis de sismos asociados a tsunamis (Ciencia de Datos - UTN FRM)

Pregunta del proyecto:
    ¿Qué características de un sismo permiten estimar si está asociado a un tsunami?

Fuente:
    API USGS - https://earthquake.usgs.gov/fdsnws/event/1/query

Diseño del dataset:
    - Una fila  = un evento sísmico
    - Clave     = id (identificador único de USGS)
    - Objetivo  = tsunami (0/1)
    - Ventana   = 2019-01-01 a 2026-07-31, magnitud >= 5  (~7.500 filas esperadas)
    - Derivada  = dist_costa_km, distancia con signo a la costa
                  (negativa = epicentro en tierra, positiva = epicentro en el mar)

Capas (medallion):
    - bronce : GeoJSON crudo tal como llega de la API -> include/output/bronze/
    - plata  : CSV limpio, una fila por sismo         -> include/output/silver/
    - entrega: copia fechada de la plata validada     -> include/output/

Estructura de tareas (igual espíritu que fifa_ingest):
    extract >> transform >> enrich >> validate >> save
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Parámetros del proyecto (fijos -> pipeline reproducible)
# ---------------------------------------------------------------------------
DAG_ID = "sismos_tsunami"

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
START_TIME = "2019-01-01"
END_TIME = "2026-07-31"
MIN_MAGNITUDE = 5

# Columnas que nos quedamos del GeoJSON (mezcla numéricas / categóricas / fecha)
COLUMNAS = [
    "id",         # clave primaria
    "time",       # fecha/hora del evento (UTC)
    "mag",        # magnitud
    "magType",    # escala de magnitud usada
    "place",      # descripción textual de la ubicación
    "latitude",   # epicentro
    "longitude",  # epicentro
    "depth",      # profundidad del hipocentro (km)
    "tsunami",    # OBJETIVO: 0/1
    "type",       # tipo de evento (earthquake, quarry blast, ...)
    "status",     # reviewed / automatic
    "sig",        # significancia calculada por USGS
    "gap",        # cobertura azimutal de estaciones
    "dmin",       # distancia a la estación más cercana
    "nst",        # cantidad de estaciones que registraron
    "rms",        # error del ajuste
    "net",        # red sismológica que reportó
]

# Rutas de salida 
BASE_DIR = Path(__file__).resolve().parent.parent / "include" / "output"
BRONZE_DIR = BASE_DIR / "bronze"
SILVER_DIR = BASE_DIR / "silver"

RAW_PATH = BRONZE_DIR / "sismos_raw.json"
CSV_PATH = SILVER_DIR / "sismos.csv"

# Capa geográfica: activo estático y congelado del proyecto.
# Son los mapas Natural Earth 1:50m (dominio público), versionados junto al
# código. No se descargan en cada corrida a propósito: si mañana Natural Earth
# cambia el trazado de una costa, dist_costa_km cambiaría sin que nosotros
# hayamos tocado nada, y la corrida dejaría de ser reproducible.
GEO_DIR = Path(__file__).resolve().parent.parent / "include" / "geo"
COSTA_PATH = GEO_DIR / "ne_50m_coastline.geojson"
TIERRA_PATH = GEO_DIR / "ne_50m_land.geojson"

RADIO_TIERRA_KM = 6371.0088  # radio medio terrestre, para la fórmula de haversine


# ---------------------------------------------------------------------------
# TAREA 1 - extract: traer el crudo de la API y guardarlo (capa BRONCE)
# ---------------------------------------------------------------------------
def extract():
    """Pega a la API de USGS y guarda el GeoJSON crudo, sin tocarlo."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "format": "geojson",
        "starttime": START_TIME,
        "endtime": END_TIME,
        "minmagnitude": MIN_MAGNITUDE,
    }
    resp = requests.get(USGS_URL, params=params, timeout=120)
    resp.raise_for_status()  # corta la corrida si la API devuelve error

    data = resp.json()

    # Se guarda tal cual llegó: si mañana descubrimos que interpretamos mal
    # una columna, podemos rehacer el transform sin volver a pegarle a la API.
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    n = len(data.get("features", []))
    print(f"[extract] {n} sismos guardados en {RAW_PATH}")


# ---------------------------------------------------------------------------
# TAREA 2 - transform: aplanar el GeoJSON al CSV (capa PLATA)
# ---------------------------------------------------------------------------
def transform():
    """Convierte el GeoJSON anidado en una tabla plana: una fila por sismo."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    filas = []
    for feature in data["features"]:
        props = feature["properties"]
        # geometry.coordinates = [longitude, latitude, depth]
        lon, lat, depth = feature["geometry"]["coordinates"]

        filas.append(
            {
                "id": feature["id"],
                "time": props.get("time"),
                "mag": props.get("mag"),
                "magType": props.get("magType"),
                "place": props.get("place"),
                "latitude": lat,
                "longitude": lon,
                "depth": depth,
                "tsunami": props.get("tsunami"),
                "type": props.get("type"),
                "status": props.get("status"),
                "sig": props.get("sig"),
                "gap": props.get("gap"),
                "dmin": props.get("dmin"),
                "nst": props.get("nst"),
                "rms": props.get("rms"),
                "net": props.get("net"),
            }
        )

    df = pd.DataFrame(filas, columns=COLUMNAS)

    # time viene en milisegundos UTC -> lo pasamos a fecha real
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"[transform] CSV escrito en {CSV_PATH} ({len(df)} filas, {df.shape[1]} columnas)")


# ---------------------------------------------------------------------------
# TAREA 3 - enrich: derivar dist_costa_km (sigue dentro de la capa PLATA)
#
# Por qué existe esta tarea:
#   latitude y longitude son geografía disfrazada de número. Un modelo que las
#   recibe crudas memoriza coordenadas en vez de aprender el fenómeno. La
#   distancia a la costa es la misma información, pero interpretable.
#
#   El signo es lo importante: un sismo 5 km tierra adentro y uno 5 km mar
#   adentro dan los dos ~5 km, y sólo uno puede generar un tsunami. Sin signo
#   la variable mezcla los dos casos.
# ---------------------------------------------------------------------------
def _anillos(geometria):
    """Devuelve cada anillo o tramo de una geometría GeoJSON como array (N, 2) lon/lat."""
    tipo, coords = geometria["type"], geometria["coordinates"]
    if tipo == "Polygon":
        return [np.asarray(a, dtype="float64") for a in coords]
    if tipo == "MultiPolygon":
        return [np.asarray(a, dtype="float64") for poli in coords for a in poli]
    if tipo == "LineString":
        return [np.asarray(coords, dtype="float64")]
    if tipo == "MultiLineString":
        return [np.asarray(a, dtype="float64") for a in coords]
    return []


def _cargar_geojson(ruta):
    """Lee una capa Natural Earth y devuelve la lista plana de sus anillos."""
    with open(ruta, "r", encoding="utf-8") as f:
        capa = json.load(f)
    return [a for feat in capa["features"] for a in _anillos(feat["geometry"])]


def _distancia_a_la_costa(lat, lon, vertices):
    """Distancia haversine al vértice de costa más cercano, en km.

    Haversine y no Pitágoras: sobre lat/lon un grado de longitud mide 111 km en
    el ecuador y casi nada cerca de los polos. La distancia euclídea sobre
    coordenadas daría números sin sentido físico.

    Se calcula por bloques de 200 sismos porque la matriz completa
    (8.826 sismos x 60.416 vértices) son ~533 millones de distancias: de una
    sola vez no entra en memoria."""
    v_lat = np.radians(vertices[:, 1])
    v_lon = np.radians(vertices[:, 0])
    cos_v_lat = np.cos(v_lat)

    p_lat = np.radians(lat)
    p_lon = np.radians(lon)

    salida = np.empty(len(lat))
    for i in range(0, len(lat), 200):
        tramo = slice(i, i + 200)
        a = (
            np.sin((v_lat[None, :] - p_lat[tramo, None]) / 2) ** 2
            + np.cos(p_lat[tramo, None])
            * cos_v_lat[None, :]
            * np.sin((v_lon[None, :] - p_lon[tramo, None]) / 2) ** 2
        )
        salida[tramo] = (
            2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))).min(axis=1)
        )
    return salida


def _esta_en_tierra(lon_p, lat_p, poligonos, cajas):
    """Punto en polígono por ray casting: cuenta cruces de una semirrecta al este.

    Número impar de cruces = el punto está dentro. Las 'cajas' son los
    rectángulos que envuelven a cada polígono: descartan de entrada los ~1.400
    anillos que ni siquiera están cerca, que es lo que hace que esto corra en
    segundos y no en horas."""
    candidatos = np.where(
        (cajas[:, 0] <= lon_p)
        & (lon_p <= cajas[:, 1])
        & (cajas[:, 2] <= lat_p)
        & (lat_p <= cajas[:, 3])
    )[0]

    dentro = False
    for k in candidatos:
        anillo = poligonos[k]
        x, y = anillo[:, 0], anillo[:, 1]
        x_sig, y_sig = np.roll(x, -1), np.roll(y, -1)

        cruza = (y > lat_p) != (y_sig > lat_p)
        if not cruza.any():
            continue

        x_corte = x[cruza] + (lat_p - y[cruza]) * (x_sig[cruza] - x[cruza]) / (
            y_sig[cruza] - y[cruza]
        )
        if int((x_corte > lon_p).sum()) % 2 == 1:
            dentro = not dentro  # un anillo interior (lago) invierte el resultado
    return dentro


def enrich():
    """Agrega la columna dist_costa_km al CSV de plata."""
    df = pd.read_csv(CSV_PATH)

    costa = np.vstack(_cargar_geojson(COSTA_PATH))
    distancia = _distancia_a_la_costa(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(), costa
    )

    tierra = _cargar_geojson(TIERRA_PATH)
    cajas = np.array(
        [[p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()] for p in tierra]
    )
    en_tierra = np.array(
        [
            _esta_en_tierra(lon, lat, tierra, cajas)
            for lon, lat in zip(df["longitude"], df["latitude"])
        ]
    )

    df["dist_costa_km"] = np.round(np.where(en_tierra, -distancia, distancia), 2)
    df.to_csv(CSV_PATH, index=False)

    print(f"[enrich] dist_costa_km calculada para {len(df)} sismos")
    print(
        f"[enrich] {int(en_tierra.sum())} en tierra (signo negativo), "
        f"{int((~en_tierra).sum())} en el mar (signo positivo)"
    )
    print(
        f"[enrich] rango: {df['dist_costa_km'].min():.1f} km a "
        f"{df['dist_costa_km'].max():.1f} km"
    )


# ---------------------------------------------------------------------------
# TAREA 4 - validate: los 7 criterios de calidad
# ---------------------------------------------------------------------------
def validate():
    """Corre los criterios medibles. Si algo no da, corta la corrida."""
    df = pd.read_csv(CSV_PATH)

    # 1) Clave sin duplicados
    assert df["id"].is_unique, "La clave 'id' tiene duplicados"

    # 2) Volumen suficiente (> 1.000 filas)
    assert len(df) > 1000, f"Muy pocas filas: {len(df)}"

    # 3) Ancho suficiente (>= 5 columnas)
    assert df.shape[1] >= 5, f"Muy pocas columnas: {df.shape[1]}"

    # 6) Ninguna columna 100% nula
    vacias = df.columns[df.isna().all()].tolist()
    assert not vacias, f"Columnas totalmente vacías: {vacias}"

    # Reportes informativos (criterios 4 y 5: se explican, no cortan)
    print("[validate] filas x columnas:", df.shape)
    print("[validate] tipos de dato:\n", df.dtypes.value_counts())
    print("[validate] % de nulos por columna:\n",
          df.isna().mean().sort_values(ascending=False))
    print("[validate] distribución del objetivo tsunami:\n",
          df["tsunami"].value_counts(dropna=False))
    print("[validate] OK: el dataset pasa los criterios medibles")


# ---------------------------------------------------------------------------
# TAREA 5 - save: entregable fechado de la corrida
# ---------------------------------------------------------------------------
def save(**context):
    """Copia la plata ya validada a include/output/ con la fecha de la corrida.

    Va **después** de validate a propósito: sólo se publica un dataset que
    pasó los chequeos. Si validate corta, en include/output/ queda el
    entregable de la corrida anterior, no uno roto.

    El nombre lleva fecha y hora para no pisar entregas viejas: cada corrida
    deja su propio archivo y se puede comparar contra el anterior.

    Ojo con `ds`: sólo existe cuando el DAG tiene schedule y por lo tanto
    intervalo de datos. Este corre a demanda, así que la fecha sale del
    DagRun. Es un tropiezo clásico al pasar de Airflow 2 a 3.
    """
    dag_run = context["dag_run"]
    momento = dag_run.logical_date or dag_run.run_after

    sello = momento.strftime("%Y-%m-%d_%H-%M-%S")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    destino = BASE_DIR / f"sismos_{sello}.csv"
    shutil.copy(CSV_PATH, destino)

    print(f"[save] entregable publicado en {destino}")
    return str(destino)


# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id=DAG_ID,
    description="Pipeline de sismos asociados a tsunamis (fuente USGS)",
    start_date=datetime(2024, 1, 1),
    schedule=None,          # se dispara a mano; ventana de datos fija
    catchup=False,
    tags=["ciencia-de-datos", "usgs", "tsunamis"],
) as dag:

    t_extract = PythonOperator(task_id="extract", python_callable=extract)
    t_transform = PythonOperator(task_id="transform", python_callable=transform)
    t_enrich = PythonOperator(task_id="enrich", python_callable=enrich)
    t_validate = PythonOperator(task_id="validate", python_callable=validate)
    t_save = PythonOperator(task_id="save", python_callable=save)

    t_extract >> t_transform >> t_enrich >> t_validate >> t_save
