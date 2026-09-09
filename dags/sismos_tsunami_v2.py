"""
DAG: sismos_tsunami
=====================
Proyecto Integrador: Análisis de sismos asociados a tsunamis
Cátedra: Ciencia de Datos — UTN FRM (5.º año, Ingeniería en Sistemas de Información)

Pregunta de investigación:
    ¿Qué características de un sismo permiten estimar si está asociado a un tsunami?

Fuente de datos:
    API USGS - https://earthquake.usgs.gov/fdsnws/event/1/query

Diseño del dataset (Modelo Medallón):
    - Una fila  = un evento sísmico
    - Clave     = id (identificador único de USGS)
    - Objetivo  = tsunami (0/1)
    - Bronce    = GeoJSON crudo tal cual llega de la API -> include/output/bronze/
    - Plata     = CSV limpio, aplanado y enriquecido con dist_costa_km -> include/output/silver/
    - Entrega   = Copia fechada del dataset de plata validado -> include/output/

Refactorizado bajo el estándar Airflow 3 / TaskFlow API (@dag, @task, @task.sensor)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
import requests

from airflow.sdk import Param, PokeReturnValue, dag, task

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración y Rutas del Proyecto
# ---------------------------------------------------------------------------
DAG_ID = "sismos_tsunami"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Rutas de salida en arquitectura Medallón
BASE_DIR = Path("/usr/local/airflow/include/output")
BRONZE_DIR = BASE_DIR / "bronze"
SILVER_DIR = BASE_DIR / "silver"

RAW_PATH = BRONZE_DIR / "sismos_raw.json"
CSV_PATH = SILVER_DIR / "sismos.csv"

# Activos geográficos estáticos
GEO_DIR = Path("/usr/local/airflow/include/geo")
COSTA_PATH = GEO_DIR / "ne_50m_coastline.geojson"
TIERRA_PATH = GEO_DIR / "ne_50m_land.geojson"

RADIO_TIERRA_KM = 6371.0088  # Radio medio terrestre IUGG

# Esquema canónico de columnas
COLUMNAS = [
    "id", "time", "mag", "magType", "place", "latitude", "longitude",
    "depth", "tsunami", "type", "status", "sig", "gap", "dmin",
    "nst", "rms", "net"
]

# ---------------------------------------------------------------------------
# Funciones auxiliares de Geometría Espacial
# ---------------------------------------------------------------------------
def _anillos(geometria: dict) -> list[np.ndarray]:
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


def _cargar_geojson(ruta: Path) -> list[np.ndarray]:
    """Lee una capa Natural Earth y devuelve la lista plana de sus anillos."""
    with open(ruta, "r", encoding="utf-8") as f:
        capa = json.load(f)
    return [a for feat in capa["features"] for a in _anillos(feat["geometry"])]


def _distancia_a_la_costa(lat: np.ndarray, lon: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Calcula la distancia Haversine al vértice de costa más cercano en km por bloques."""
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


def _esta_en_tierra(lon_p: float, lat_p: float, poligonos: list[np.ndarray], cajas: np.ndarray) -> bool:
    """Clasificación Continental / Marítima mediante el algoritmo de Ray Casting."""
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
            dentro = not dentro
    return dentro


# ---------------------------------------------------------------------------
# Definición del DAG con TaskFlow API
# ---------------------------------------------------------------------------
@dag(
    dag_id=DAG_ID,
    schedule=None,  # Se ejecuta a demanda o según requerimiento
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["ciencia-de-datos", "usgs", "tsunamis", "utn-frm"],
    doc_md=__doc__,
    params={
        "start_time": Param(
            "2019-01-01",
            type="string",
            title="Fecha Inicial",
            description="Fecha de inicio de la ventana de extracción (YYYY-MM-DD)",
        ),
        "end_time": Param(
            "2026-07-31",
            type="string",
            title="Fecha Final",
            description="Fecha de fin de la ventana de extracción (YYYY-MM-DD)",
        ),
        "min_magnitude": Param(
            5.0,
            type="number",
            title="Magnitud Mínima",
            description="Filtro sísmico ($M \\ge 5.0$ para eventos con potencial de tsunami)",
        ),
        "force": Param(
            False,
            type="boolean",
            title="Forzar Descarga",
            description="Fuerza la descarga desde la API ignorando la caché local en bronce",
        ),
    },
)
def sismos_tsunami():

    @task.sensor(poke_interval=60, timeout=600, mode="reschedule", soft_fail=True)
    def wait_for_source(**context) -> PokeReturnValue:
        """Sensor: Verifica disponibilidad y respuesta de la API de USGS."""
        params = context["params"]
        query_params = {
            "format": "geojson",
            "starttime": params["start_time"],
            "endtime": params["end_time"],
            "minmagnitude": params["min_magnitude"],
            "limit": 1,  # Petición liviana para prueba de vida
        }
        try:
            resp = requests.get(USGS_URL, params=query_params, timeout=30)
            if resp.status_code == 200:
                log.info("API de USGS responde correctamente.")
                return PokeReturnValue(is_done=True, xcom_value=True)
            log.warning("USGS devolvió código HTTP %s. Reintentando...", resp.status_code)
            return PokeReturnValue(is_done=False)
        except Exception as e:
            log.warning("No se pudo conectar a la API de USGS (%s). Reintentando...", e)
            return PokeReturnValue(is_done=False)

    @task
    def extract(**context) -> str:
        """Capa Bronce: Descarga el GeoJSON crudo y lo persiste en disco sin modificar."""
        params = context["params"]
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)

        if RAW_PATH.exists() and not params["force"]:
            log.info("Capa Bronce reutilizada desde caché en %s", RAW_PATH)
            return str(RAW_PATH)

        query_params = {
            "format": "geojson",
            "starttime": params["start_time"],
            "endtime": params["end_time"],
            "minmagnitude": params["min_magnitude"],
        }
        log.info("Descargando eventos sísmicos desde la USGS...")
        resp = requests.get(USGS_URL, params=query_params, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        n = len(data.get("features", []))
        log.info("%s sismos descargados e ingestados en Capa Bronce: %s", n, RAW_PATH)
        return str(RAW_PATH)

    @task
    def transform(raw_json_path: str) -> str:
        """Capa Plata (Paso 1): Transforma el JSON anidado en un CSV aplanado."""
        SILVER_DIR.mkdir(parents=True, exist_ok=True)

        with open(raw_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filas = []
        for feature in data["features"]:
            props = feature["properties"]
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
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

        df.to_csv(CSV_PATH, index=False)
        log.info("Aplanamiento completado: %s filas x %s columnas -> %s", len(df), df.shape[1], CSV_PATH)
        return str(CSV_PATH)

    @task
    def enrich(silver_csv_path: str) -> str:
        """Capa Plata (Paso 2): Calcula dist_costa_km con signo (Feature Engineering)."""
        df = pd.read_csv(silver_csv_path)

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
        df.to_csv(silver_csv_path, index=False)

        log.info(
            "Enriquecimiento completado: %s en tierra (-), %s en mar (+)",
            int(en_tierra.sum()), int((~en_tierra).sum())
        )
        return silver_csv_path

    @task
    def validate(silver_csv_path: str) -> str:
        """Control de Calidad (Data Quality Gate): Valida reglas de negocio."""
        df = pd.read_csv(silver_csv_path)

        problemas = []
        if not df["id"].is_unique:
            problemas.append("La clave primaria 'id' contiene duplicados")
        if len(df) <= 1000:
            problemas.append(f"Volumen de datos insuficiente: {len(df)} filas")
        if df.shape[1] < 5:
            problemas.append(f"Cantidad de columnas insuficiente: {df.shape[1]}")

        vacias = df.columns[df.isna().all()].tolist()
        if vacias:
            problemas.append(f"Columnas completamente nulas detectadas: {vacias}")

        if problemas:
            raise ValueError("Validación de calidad fallida:\n - " + "\n - ".join(problemas))

        log.info("Validación OK: %s filas x %s columnas. Distribución Tsunami:\n%s",
                 len(df), df.shape[1], df["tsunami"].value_counts(dropna=False))
        return silver_csv_path

    @task
    def save(silver_csv_path: str, **context) -> str:
        """Capa Gold / Entrega: Publica el entregable validado con sello de fecha."""
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        sello = momento.strftime("%Y-%m-%d_%H-%M-%S")

        BASE_DIR.mkdir(parents=True, exist_ok=True)
        destino = BASE_DIR / f"sismos_{sello}.csv"
        shutil.copy(silver_csv_path, destino)

        log.info("Entregable final publicado con éxito en: %s", destino)
        return str(destino)

    # ---------------------------------------------------------------------------
    # Orquestación del Grafo de Dependencias
    # ---------------------------------------------------------------------------
    ready = wait_for_source()
    raw_json = extract()
    csv_plata = transform(raw_json)
    csv_enriquecido = enrich(csv_plata)
    csv_validado = validate(csv_enriquecido)
    entregable = save(csv_validado)

    # Sensor antecede a la extracción
    ready >> raw_json


# Instanciación del DAG
sismos_tsunami()