"""
DAG: sismos_tsunami
=====================
Proyecto Integrador: Análisis de sismos asociados a tsunamis
Cátedra: Ciencia de Datos — UTN FRM (5.º año, Ingeniería en Sistemas de Información)

Refactorizado: Ingesta Incremental Diaria (Schedule 02:00 AM) con validación de omisión.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
import requests

from airflow.sdk import Param, PokeReturnValue, dag, task
from airflow.exceptions import AirflowSkipException

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración y Rutas del Proyecto
# ---------------------------------------------------------------------------
DAG_ID = "sismos_tsunami"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"

BASE_DIR = Path("/usr/local/airflow/include/output")
BRONZE_DIR = BASE_DIR / "bronze"
SILVER_DIR = BASE_DIR / "silver"

# El entregable final consolidado
MASTER_CSV_PATH = BASE_DIR / "sismos_master.csv"

GEO_DIR = Path("/usr/local/airflow/include/geo")
COSTA_PATH = GEO_DIR / "ne_50m_coastline.geojson"
TIERRA_PATH = GEO_DIR / "ne_50m_land.geojson"

RADIO_TIERRA_KM = 6371.0088

COLUMNAS = [
    "id", "time", "mag", "magType", "place", "latitude", "longitude",
    "depth", "tsunami", "type", "status", "sig", "gap", "dmin",
    "nst", "rms", "net"
]

# ---------------------------------------------------------------------------
# Funciones auxiliares de Geometría Espacial (Se mantienen sin cambios)
# ---------------------------------------------------------------------------
def _anillos(geometria: dict) -> list[np.ndarray]:
    tipo, coords = geometria["type"], geometria["coordinates"]
    if tipo == "Polygon": return [np.asarray(a, dtype="float64") for a in coords]
    if tipo == "MultiPolygon": return [np.asarray(a, dtype="float64") for poli in coords for a in poli]
    if tipo == "LineString": return [np.asarray(coords, dtype="float64")]
    if tipo == "MultiLineString": return [np.asarray(a, dtype="float64") for a in coords]
    return []

def _cargar_geojson(ruta: Path) -> list[np.ndarray]:
    with open(ruta, "r", encoding="utf-8") as f: capa = json.load(f)
    return [a for feat in capa["features"] for a in _anillos(feat["geometry"])]

def _distancia_a_la_costa(lat: np.ndarray, lon: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    v_lat, v_lon = np.radians(vertices[:, 1]), np.radians(vertices[:, 0])
    cos_v_lat = np.cos(v_lat)
    p_lat, p_lon = np.radians(lat), np.radians(lon)
    salida = np.empty(len(lat))
    for i in range(0, len(lat), 200):
        tramo = slice(i, i + 200)
        a = (np.sin((v_lat[None, :] - p_lat[tramo, None]) / 2) ** 2
             + np.cos(p_lat[tramo, None]) * cos_v_lat[None, :]
             * np.sin((v_lon[None, :] - p_lon[tramo, None]) / 2) ** 2)
        salida[tramo] = 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))).min(axis=1)
    return salida

def _esta_en_tierra(lon_p: float, lat_p: float, poligonos: list[np.ndarray], cajas: np.ndarray) -> bool:
    candidatos = np.where((cajas[:, 0] <= lon_p) & (lon_p <= cajas[:, 1]) & (cajas[:, 2] <= lat_p) & (lat_p <= cajas[:, 3]))[0]
    dentro = False
    for k in candidatos:
        anillo = poligonos[k]
        x, y = anillo[:, 0], anillo[:, 1]
        x_sig, y_sig = np.roll(x, -1), np.roll(y, -1)
        cruza = (y > lat_p) != (y_sig > lat_p)
        if not cruza.any(): continue
        x_corte = x[cruza] + (lat_p - y[cruza]) * (x_sig[cruza] - x[cruza]) / (y_sig[cruza] - y[cruza])
        if int((x_corte > lon_p).sum()) % 2 == 1: dentro = not dentro
    return dentro

# ---------------------------------------------------------------------------
# Definición del DAG con TaskFlow API
# ---------------------------------------------------------------------------
@dag(
    dag_id=DAG_ID,
    schedule="0 2 * * *",  # Ejecución diaria a las 02:00 AM
    #schedule="*/2 * * * *",  # Expresión cron: cada 2 minutos
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["ciencia-de-datos", "usgs", "tsunamis", "utn-frm"],
    doc_md=__doc__,
    params={
        "min_magnitude": Param(5.0, type="number", title="Magnitud Mínima"),
        "force_historical": Param(
            False, type="boolean", title="Forzar Carga Histórica",
            description="Si es True, usa las fechas manuales e ignora la ventana diaria."
        ),
        "manual_start": Param("2019-01-01", type="string", title="Fecha Inicio Manual"),
        "manual_end": Param(
            "",  # Valor por defecto vacío
            type=["string", "null"], 
            title="Fecha Fin Manual (Opcional)",
            description="Si se deja vacío, el pipeline utilizará automáticamente la fecha de hoy."
        ),
    },
)
def sismos_tsunami():

    def _obtener_ventana(context) -> tuple[str, str]:
        """Resuelve si se usa la ventana diaria automática o la manual histórica."""
        params = context["params"]
        
        if params["force_historical"]:
            start = params["manual_start"]
            end = params.get("manual_end")
            
            if not end or end.strip() == "":
                end = pendulum.now("UTC").to_date_string()
                
            return start, end
            
        # Ventana diaria ROBUSTA para las últimas 24 horas (funciona manual y programado)
        fin = context["logical_date"]
        inicio = fin.subtract(days=1)
        
        # Enviamos la fecha y la HORA exacta (ISO8601) para que USGS no asuma 00:00:00
        return inicio.to_iso8601_string(), fin.to_iso8601_string()
    
    @task.sensor(poke_interval=60, timeout=600, mode="reschedule", soft_fail=True)
    def wait_for_source(**context) -> PokeReturnValue:
        start_str, end_str = _obtener_ventana(context)
        query_params = {
            "starttime": start_str,
            "endtime": end_str,
            "minmagnitude": context["params"]["min_magnitude"],
        }
        
        try:
            # Consulta ultra liviana al endpoint /count
            resp = requests.get(USGS_COUNT_URL, params=query_params, timeout=30)
            resp.raise_for_status()
            total_eventos = int(resp.text.strip())
            
            if total_eventos == 0:
                log.info("Cero sismos M>=5.0 detectados entre %s y %s. Omitiendo pipeline.", start_str, end_str)
                raise AirflowSkipException("Ventana sin sismos: ejecución omitida correctamente.")
                
            log.info("API operativa. Se detectaron %s eventos en la ventana.", total_eventos)
            return PokeReturnValue(is_done=True, xcom_value=total_eventos)
            
        except requests.exceptions.RequestException as e:
            log.warning("Fallo de conexión a la API USGS: %s. Reintentando...", e)
            return PokeReturnValue(is_done=False)

    @task
    def extract(**context) -> str:
        start_str, end_str = _obtener_ventana(context)
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Archivo particionado por día para no pisar el historial
        raw_path = BRONZE_DIR / f"sismos_raw_{start_str}_{end_str}.json"

        query_params = {
            "format": "geojson",
            "starttime": start_str,
            "endtime": end_str,
            "minmagnitude": context["params"]["min_magnitude"],
        }
        log.info("Descargando ventana %s a %s...", start_str, end_str)
        resp = requests.get(USGS_URL, params=query_params, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return str(raw_path)

    @task
    def transform(raw_json_path: str) -> str:
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = SILVER_DIR / Path(raw_json_path).name.replace(".json", ".csv")

        with open(raw_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filas = []
        for feature in data["features"]:
            props = feature["properties"]
            lon, lat, depth = feature["geometry"]["coordinates"]
            filas.append({
                "id": feature["id"], "time": props.get("time"), "mag": props.get("mag"),
                "magType": props.get("magType"), "place": props.get("place"),
                "latitude": lat, "longitude": lon, "depth": depth, "tsunami": props.get("tsunami"),
                "type": props.get("type"), "status": props.get("status"), "sig": props.get("sig"),
                "gap": props.get("gap"), "dmin": props.get("dmin"), "nst": props.get("nst"),
                "rms": props.get("rms"), "net": props.get("net"),
            })

        df = pd.DataFrame(filas, columns=COLUMNAS)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @task
    def enrich(silver_csv_path: str) -> str:
        df = pd.read_csv(silver_csv_path)
        costa = np.vstack(_cargar_geojson(COSTA_PATH))
        distancia = _distancia_a_la_costa(df["latitude"].to_numpy(), df["longitude"].to_numpy(), costa)

        tierra = _cargar_geojson(TIERRA_PATH)
        cajas = np.array([[p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()] for p in tierra])
        en_tierra = np.array([_esta_en_tierra(lon, lat, tierra, cajas) for lon, lat in zip(df["longitude"], df["latitude"])])

        df["dist_costa_km"] = np.round(np.where(en_tierra, -distancia, distancia), 2)
        df.to_csv(silver_csv_path, index=False)
        return silver_csv_path

    @task
    def validate(silver_csv_path: str) -> str:
        df = pd.read_csv(silver_csv_path)
        problemas = []
        
        if not df["id"].is_unique: problemas.append("Duplicados detectados en 'id'")
        if df.empty: problemas.append("Dataset validado vacío (debió atraparse en el sensor)")
        if df.shape[1] < 5: problemas.append(f"Faltan columnas (actual: {df.shape[1]})")
        vacias = df.columns[df.isna().all()].tolist()
        if vacias: problemas.append(f"Columnas completamente nulas: {vacias}")

        if problemas:
            raise ValueError("Data Quality Gate falló:\n - " + "\n - ".join(problemas))
            
        return silver_csv_path

    @task
    def save(silver_csv_path: str, **context) -> str:
        """Capa Gold: Upsert diario o Sobrescritura total en carga histórica."""
        df_lote = pd.read_csv(silver_csv_path)
        params = context["params"]
        
        # Si forzamos la carga histórica, sobrescribimos todo el archivo maestro
        if params["force_historical"]:
            df_lote.to_csv(MASTER_CSV_PATH, index=False)
            log.info("Carga Histórica: Dataset maestro sobrescrito completamente. Total: %s filas", len(df_lote))
            return str(MASTER_CSV_PATH)

        # Si es una carga diaria (incremental), anexamos al maestro existente
        if MASTER_CSV_PATH.exists():
            df_master = pd.read_csv(MASTER_CSV_PATH)
            df_final = pd.concat([df_master, df_lote], ignore_index=True)
            # Deduplicación crítica para mantener idempotencia
            df_final.drop_duplicates(subset=["id"], keep="last", inplace=True)
            df_final.to_csv(MASTER_CSV_PATH, index=False)
            log.info("Lote diario anexado. Dataset maestro actualizado: %s filas", len(df_final))
        else:
            # Si el maestro no existía por algún motivo, lo crea
            df_lote.to_csv(MASTER_CSV_PATH, index=False)
            log.info("Dataset maestro creado desde cero con %s filas", len(df_lote))
            
        return str(MASTER_CSV_PATH)
    @task
    def notify(**context):
        """Envía una alerta a un canal externo al finalizar el DAG."""
        fecha = context["logical_date"].to_date_string()
        
        # Podés generar este link gratis creando un Webhook en la configuración de un canal de Discord o Slack
        # El link del webhook es secreto: se lee de la variable de entorno DISCORD_WEBHOOK_URL (archivo .env de Astro)
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook_url:
            print("Sin DISCORD_WEBHOOK_URL: no se envía la notificación.")
            return
        
        mensaje = {
            "content": f"✅ **ETL Sismos & Tsunami**\nLa actualización de datos del día {fecha} se procesó y guardó correctamente en el archivo maestro."
        }
        
        try:
            # Enviamos el POST al webhook de la misma forma que consultamos una API
            resp = requests.post(webhook_url, json=mensaje, timeout=10)
            if resp.status_code in [200, 204]:
                print("Notificación enviada con éxito.")
            else:
                print(f"Error en webhook. Código: {resp.status_code}")
        except Exception as e:
            print(f"Fallo de conexión al intentar notificar: {e}")
    
    # ---------------------------------------------------------------------------
    # Orquestación
    # ---------------------------------------------------------------------------
    ready = wait_for_source()
    raw_json = extract()
    csv_plata = transform(raw_json)
    csv_enriquecido = enrich(csv_plata)
    csv_validado = validate(csv_enriquecido)
    entregable = save(csv_validado)

    # 1. Instanciás la nueva tarea
    alerta = notify()

    # 2. Actualizás las dependencias explícitas (modificando tu línea 305)
    ready >> raw_json
    entregable >> alerta

sismos_tsunami()