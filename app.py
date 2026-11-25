"""¿Qué sismos marca la USGS con alerta de tsunami? — Proyecto Integrador, Grupo 9 (UTN FRM, 2026).

Aplicación de Streamlit con forma de copa de martini:
  portada → tallo (los datos, lo que encontramos, el modelo) → copa (explorá, probá el modelo).

Lee lo que genera build_app.py en app_data/. Para correrla:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

import modelo  # define EsOceanico: joblib la necesita para cargar el pipeline

BASE = Path(__file__).parent
APP_DATA = BASE / "app_data"
MUNDO = alt.topo_feature("https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json", "countries")
alt.data_transformers.disable_max_rows()   # los mapas tienen ~13.600 sismos

# --- Colores: significan lo mismo en toda la aplicación -----------------------------------------
AZUL, AZUL_CLARO = "#2a78d6", "#9cc3ef"          # con marca (claro: con marca y magnitud 6 o más, en el mapa)
GRIS_CLARO, GRIS = "#cfcac3", "#7a7974"          # sin marca · sin marca cargada todavía
NARANJA, VIOLETA = "#d95926", "#8a5cc2"          # falso negativo · falsa alarma
MAR, TIERRA = "#1b9e8f", "#b8862b"               # dónde está el epicentro
NEUTRO = "#4a5560"                               # barras que no son una clase (distribuciones, importancias)

RESULTADOS = ["acierto: con marca", "falso negativo", "falsa alarma", "acierto: sin marca", "sin marca cargada"]
ESCALA_RESULTADO = alt.Scale(domain=RESULTADOS, range=[AZUL, NARANJA, VIOLETA, GRIS_CLARO, GRIS])
CON, SIN = "con marca de la USGS", "sin marca de la USGS"   # las dos clases, con el recordatorio de qué es la marca
ESCALA_MARCA = alt.Scale(domain=[CON, SIN], range=[AZUL, GRIS_CLARO])
ESCALA_ZONA = alt.Scale(domain=["mar", "tierra"], range=[MAR, TIERRA])

# Regiones donde la USGS usa un umbral regional (sección 2.1 de la Entrega 3)
REGION_ALASKA_OESTE = ["Alaska", "Oregon", "off the coast of Oregon", "Canada", "CA", "California", "OR",
                       "south of Alaska"]

# Rangos de magnitud y de distancia que usan los gráficos de «Lo que encontramos»
CORTES_MAG = [4.99, 5.5, 6.0, 6.5, 7.0, 7.5, 9.5]
RANGOS_MAG = ["5,0–5,5", "5,5–6,0", "6,0–6,5", "6,5–7,0", "7,0–7,5", "más de 7,5"]
CORTES_PROF = [-1, 70, 300, 1000]
RANGOS_PROF = ["superficial (menos de 70 km)", "intermedio (70 a 300 km)", "profundo (más de 300 km)"]

# Tipos de magnitud de la USGS: el nombre que muestra el selector de «Probá el modelo» y la explicación de su cuadro
NOMBRES_MAGTYPE = {
    "mww": "Magnitud de momento, fase W (la estándar de la USGS)",
    "mb": "Magnitud de ondas de cuerpo (sismos moderados y lejanos)",
    "mw": "Magnitud de momento, informada por otra red",
    "mwr": "Magnitud de momento regional",
    "ml": "Magnitud local, tipo Richter (redes regionales)",
    "mwb": "Magnitud de momento con ondas de cuerpo",
}
EXPLICACION_MAGTYPE = {  # sigla: (cómo se calcula, cuándo la usa la USGS)
    "mww": ("Calcula el momento sísmico (qué tan grande es la falla que se rompió y cuánto se deslizó) con la fase W, "
            "una onda de período muy largo que registran estaciones de todo el mundo.",
            "Es la que la USGS prefiere para la mayoría de los sismos medianos y grandes, y la primera que publica en "
            "los grandes. Por eso es la de casi todos los sismos con marca."),
    "mb": ("Mide la amplitud de las ondas P (las primeras en llegar) en sismógrafos lejanos al sismo.",
           "Sismos moderados y lejanos que todavía no tienen magnitud de momento. Se satura cerca de 6: en sismos "
           "más grandes deja de crecer, así que no sirve para medirlos."),
    "mw": ("Calcula el momento sísmico, pero el dato lo aporta otra agencia o red sismológica.",
           "Cuando la magnitud preferida del catálogo viene de un contribuyente externo, por ejemplo una red de "
           "California."),
    "mwr": ("Calcula el momento sísmico con ondas registradas por estaciones de la región del sismo.",
            "Sismos moderados en zonas con buena red regional (en los datos: Chile, Alaska y Taiwán, entre otras)."),
    "ml": ("Es la escala original de Richter: mide la amplitud de las ondas en estaciones cercanas.",
           "La informan redes regionales (en los datos: Alaska, Puerto Rico, Hawái y Nevada). También se satura en "
           "sismos grandes."),
    "mwb": ("Calcula el momento sísmico con ondas de cuerpo de período largo.",
            "Una alternativa a mww para sismos moderados a grandes."),
}


# --- Formato en castellano ----------------------------------------------------------------------
def pct(x, dec=0):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def grafico(chart):
    """Todos los gráficos van sin el tema de Streamlit, para que los colores sean los de arriba."""
    st.altair_chart(chart, theme=None)


def tarjeta_kpi(col, valor, descripcion):
    """Un número grande con su explicación debajo (los resultados de «Probá el modelo»)."""
    col.markdown(f"<div style='background-color:rgba(127,127,127,0.08); border-radius:8px; padding:1rem 1.15rem; "
                 f"height:100%'><div style='color:{AZUL}; font-size:2.1rem; font-weight:800; line-height:1.1; "
                 f"margin-bottom:0.4rem'>{valor}</div><div style='opacity:0.8; font-size:0.9rem; line-height:1.35'>"
                 f"{descripcion}</div></div>", unsafe_allow_html=True)


def _nombre_principal(serie):
    """La región que más se repite en una zona (para el nombre que muestra el mapa de «Probá el modelo»)."""
    moda = serie.dropna().mode()
    return moda.iloc[0] if len(moda) else "Océano"


# --- Las funciones de la cátedra (Entrega 2) ----------------------------------------------------
CORTES_SEMAFORO = {"separacion": (0.2, 0.8), "correlacion": (0.2, 0.6)}


def separacion(x, y):
    """Distancia entre dos muestras, en desviaciones estándar."""
    x, y = x.dropna(), y.dropna()
    return abs(x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)


def semaforo(valor, medida="separacion"):
    bajo, alto = CORTES_SEMAFORO[medida]
    v = abs(valor)
    return "roja" if v < bajo else "verde" if v > alto else "amarilla"


def asociacion(x, y):
    """Pearson, Spearman y la brecha entre ambos."""
    p, s = x.corr(y), x.corr(y, method="spearman")
    return p, s, s - p


# --- Carga ---------------------------------------------------------------------------------------
# Cada función recibe la fecha de modificación de su archivo. Streamlit guarda en memoria lo que leyó y lo reutiliza
# mientras la función y sus argumentos no cambien: sin este argumento, cuando build_app.py (o un git pull en Streamlit
# Cloud) reemplaza el archivo, la app sigue usando la copia vieja. Con la fecha como argumento, la vuelve a leer.
def modificado(archivo):
    return (APP_DATA / archivo).stat().st_mtime


@st.cache_data
def cargar_sismos(fecha_archivo):
    return pd.read_csv(APP_DATA / "sismos_app.csv")


@st.cache_data
def cargar_resumen(fecha_archivo):
    with open(APP_DATA / "resumen.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo(fecha_archivo):
    return joblib.load(APP_DATA / "modelo.joblib")


st.set_page_config(page_title="Sismos y marca de tsunami · Grupo 9", layout="wide")

if not (APP_DATA / "sismos_app.csv").exists():
    st.error("Faltan los datos de la aplicación: correr primero `python build_app.py`.")
    st.stop()

sismos = cargar_sismos(modificado("sismos_app.csv")).round({"latitude": 2, "longitude": 2, "depth": 1})   # menos decimales = página más liviana
resumen = cargar_resumen(modificado("resumen.json"))
guardado = cargar_modelo(modificado("modelo.joblib"))
exp = resumen["experimentos"]
UMBRAL = guardado["umbral"]
modelados = sismos[sismos["conjunto"] != "sin marca cargada"].copy()   # los que usa el modelo (hasta marzo de 2026)
modelados["marca"] = modelados["tsunami"].map({1: CON, 0: SIN})
con_marca, sin_marca = modelados[modelados["tsunami"] == 1], modelados[modelados["tsunami"] == 0]
test = resumen["test"]
vc = resumen["validacion_cruzada"]
vc05 = resumen["validacion_cruzada_umbral_05"]
fondo = alt.Chart(MUNDO).mark_geoshape(fill="#eeeeee", stroke="white", strokeWidth=0.5)

if guardado.get("version_sklearn") != sklearn.__version__:
    st.warning(f"El modelo se guardó con scikit-learn {guardado.get('version_sklearn')} y acá corre "
               f"{sklearn.__version__}. Instalar la versión de requirements.txt.")


# =================================================================================================
# PORTADA
# =================================================================================================
st.caption("Ciencia de Datos · UTN FRM · 2026 · Proyecto Integrador · Grupo 9 · Comisión 5K09")
st.title("¿Qué características de un sismo permiten estimar si está asociado a un tsunami?")

# La respuesta a la pregunta, en un párrafo y con letra más grande que el resto
st.markdown(
    "<div style='font-size:1.2rem; line-height:1.6; margin-bottom:1rem'>"
    "<b>Con nuestros datos, la pregunta no se puede responder tal como la planteamos.</b> El catálogo de la USGS (Servicio Geológico de Estados Unidos) no "
    "registra qué sismos generaron un tsunami: la columna <code>tsunami</code> es la alerta que la USGS emite cuando un "
    "sismo cumple su criterio. Lo más cerca que pudimos llegar es estimar qué sismos reciben esa alerta."
    "</div>", unsafe_allow_html=True)

# Tres números del test que sostienen la respuesta (salen de resumen.json)
alertas_test = test["aciertos_con_marca"] + test["falsas_alarmas"]
con_marca_test = test["aciertos_con_marca"] + test["falsos_negativos"]
umbral_txt = num(resumen["umbral"], 2)
numeros_portada = [
    (num(test["f1"], 2), AZUL,
     f"F1 en el test (clase con marca), contra {num(resumen['piso_test']['f1'], 2)} del piso que dice siempre "
     "«sin marca»"),
    (pct(test["recall_tsunami"]), AZUL,
     f"de los {con_marca_test} sismos con marca del test detectados ({test['aciertos_con_marca']} aciertos, "
     f"umbral {umbral_txt})"),
    (str(test["falsas_alarmas"]), VIOLETA,
     f"falsas alarmas entre las {alertas_test} alertas del test (1 de cada "
     f"{round(alertas_test / test['falsas_alarmas'])}). Es el precio de bajar el umbral a {umbral_txt} para no "
     "perder sismos con marca"),
]
for col, (valor, color, texto) in zip(st.columns(3), numeros_portada):
    col.markdown(f"<div style='font-size:2.6rem; font-weight:700; line-height:1.1; color:{color}'>{valor}</div>"
                 f"<div style='color:{GRIS}; line-height:1.4; margin-bottom:1rem'>{texto}</div>",
                 unsafe_allow_html=True)

st.subheader("`tsunami = 1` no significa que hubo una ola: es la marca de alerta de la USGS")
st.markdown("La marca la pone la USGS cuando un sismo cumple su criterio de alerta, y ese criterio "
            "depende de la magnitud **y de la región**: cerca de Alaska y de la costa oeste de Norteamérica, donde la "
            "USGS opera un sistema de alerta propio, marca sismos más chicos que en el resto del mundo. Por eso toda la "
            "aplicación habla de **sismos con marca de la USGS**, y el modelo predice esa marca, no la ola.")

st.header("De qué se trata")
st.markdown(f"""
Un **tsunami** es una serie de olas que se forma cuando un sismo grande mueve de golpe el fondo del mar. No todo sismo
fuerte genera uno: depende de cuán grande es, de si ocurre bajo el mar o cerca de la costa, y de cuán profundo es.

El **Servicio Geológico de Estados Unidos (USGS)** registra todos los sismos del mundo en un catálogo público, con su
magnitud, su ubicación y su profundidad. Cuando un sismo cumple su criterio de alerta, le pone una **marca de tsunami**.

**Qué hicimos nosotros:** tomamos los {miles(len(sismos))} sismos de magnitud 5 o más registrados entre 2019 y 2026,
buscamos qué distingue a los que tienen esa marca de los que no, y entrenamos un modelo que estima la probabilidad de
que un sismo la reciba.
""")
pasos = [("1 · Los datos", "Un pipeline descarga todos los días los sismos del catálogo de la USGS."),
         ("2 · El análisis", "Buscamos qué separa a los sismos con marca de los sin marca."),
         ("3 · El modelo", "Aprende a estimar la probabilidad de que un sismo reciba la marca."),
         ("4 · La aplicación", "Muestra lo que encontramos y deja explorar los datos y probar el modelo.")]
for col, (titulo, texto) in zip(st.columns(4), pasos):
    with col.container(border=True):
        st.markdown(f"**{titulo}**  \n{texto}")

st.divider()


# =================================================================================================
# TALLO 1 · LOS DATOS
# =================================================================================================
st.header("Los datos")
n_total, n_marca = len(sismos), int(sismos["tsunami"].sum())
st.subheader(f"De {miles(n_total)} sismos fuertes registrados desde 2019, sólo {miles(n_marca)} llevan la marca de "
             f"tsunami: 1 de cada {round(n_total / n_marca)}")

clases = pd.DataFrame({"marca": [CON, SIN], "sismos": [n_marca, n_total - n_marca], "orden": [0, 1]})
clases["parte"] = clases["sismos"] / n_total
clases["numero"] = [pct(n_marca / n_total, 1), ""]   # sólo la fila con marca lleva texto (si no, se escribe dos veces)
clases["texto"] = [CON, ""]
# La dona se gira para que la porción con marca quede arriba, centrada, y la etiqueta va en el centro del círculo.
# La etiqueta se ubica con theta y radio 0 (no con x e y fijos), así queda en el centro aunque cambie el ancho.
inicio = -np.pi * n_marca / n_total
base_dona = alt.Chart(clases).encode(
    theta=alt.Theta("sismos:Q", stack=True, scale=alt.Scale(range=[inicio, inicio + 2 * np.pi])),
    order=alt.Order("orden:Q"))
dona = base_dona.mark_arc(innerRadius=100, outerRadius=155, stroke="white", strokeWidth=2).encode(
    color=alt.Color("marca:N", scale=ESCALA_MARCA, legend=None),
    tooltip=[alt.Tooltip("marca:N", title=" "), alt.Tooltip("sismos:Q"), alt.Tooltip("parte:Q", format=".1%")])
etiqueta_num = base_dona.mark_text(radius=0, dy=-12, fontSize=38, fontWeight="bold", color=AZUL).encode(text="numero:N")
etiqueta_txt = base_dona.mark_text(radius=0, dy=22, fontSize=13, color="#444").encode(text="texto:N")
k1, k2 = st.columns([1, 1.25])
with k1:
    grafico((dona + etiqueta_num + etiqueta_txt).properties(width=340, height=340))
with k2:
    st.metric(f"sismos {CON}", miles(n_marca), pct(n_marca / n_total, 1), delta_color="off")
    st.metric(f"sismos {SIN}", miles(n_total - n_marca), pct(1 - n_marca / n_total, 1), delta_color="off")
    st.markdown(f"Cada fila es un sismo de magnitud 5 o más del catálogo de la USGS, y la columna `tsunami` dice si "
                f"tiene la marca. **La diferencia es enorme**: un modelo que diga siempre «sin marca» acierta el "
                f"{pct(1 - n_marca / n_total)} sin aprender nada. Por eso la accuracy no sirve para medir el modelo, y "
                "se usa el **F1 de la clase con marca**: combina cuántos sismos con marca encuentra el modelo y cuántas "
                "de sus alertas son correctas. Decir siempre «sin marca» da un F1 de 0.")

d1, d2, d3, d4 = st.columns(4)
d1.metric("columnas", resumen["datos"]["columnas"])
d2.metric("desde", pd.Timestamp(resumen["datos"]["desde"]).strftime("%d/%m/%Y"))
d3.metric("hasta", pd.Timestamp(resumen["datos"]["hasta"]).strftime("%d/%m/%Y"))
d4.metric("sismos duplicados", resumen["datos"]["ids_duplicados"])

st.markdown("##### Las variables con las que trabajamos")
st.caption("De las 18 columnas del catálogo, éstas son las que usa el modelo. Las demás, y por qué no se usan, están "
           "más abajo.")
rango_de = lambda c, dec=1: (f"{num(sismos[c].min(), dec)} a {num(sismos[c].max(), dec)}" if dec else
                             f"{miles(round(sismos[c].min()))} a {miles(round(sismos[c].max()))}")
cuota_tipo = sismos["magType"].value_counts(normalize=True)
tarjetas = [
    ("Magnitud", "mag", "Cuánta energía liberó el sismo, en escala logarítmica.", f"De {rango_de('mag')}"),
    ("Profundidad", "depth", "A cuántos kilómetros bajo la superficie empezó el sismo.", f"De {rango_de('depth', 0)} km"),
    ("Ubicación", "latitude, longitude", "Las coordenadas del epicentro, el punto de la superficie sobre el sismo.",
     "Todo el mundo"),
    ("Tipo de magnitud", "magType", "Con qué método calculó la USGS la magnitud.",
     f"mww ({pct(cuota_tipo.get('mww', 0))}), mb ({pct(cuota_tipo.get('mb', 0))}) y otros"),
    ("Mar o tierra", "dist_costa_km", "Nuestra variable: distancia a la costa, negativa en tierra y positiva en el mar. "
     "El modelo usa sólo el signo.", f"De {rango_de('dist_costa_km', 0)} km"),
]
cols_var = st.columns(3) + st.columns(3)
for col, (nombre, columna, que_es, rango_txt) in zip(cols_var, tarjetas):
    with col.container(border=True):
        st.markdown(f"**{nombre}** · `{columna}`  \n{que_es}  \n*{rango_txt}*")
cols_var[5].info(f"**La marca** · `tsunami`  \n**Lo que queremos predecir:** 1 si la USGS marcó el sismo con alerta "
                 f"de tsunami, 0 si no.  \n*{miles(n_marca)} de {miles(n_total)} sismos*")

st.markdown("##### Por qué la validación del modelo se arma por grupos")
st.markdown("**Los sismos no son independientes: uno grande trae decenas de réplicas.** Cada fila es un sismo, pero "
            "después de un sismo grande vienen decenas de réplicas en el mismo lugar y en los días siguientes. Para medir "
            "el modelo, los datos se parten en 5 pedazos: se entrena con cuatro y se evalúa con el quinto, cinco veces. "
            "Si una réplica queda en los pedazos de entrenamiento y otra de la misma secuencia en el de evaluación, el "
            "modelo «reconoce» la secuencia y el resultado sale mejor de lo que es. Por eso los pedazos se arman **por "
            "grupo**: una celda de 2° × 2° de latitud y longitud y un mes calendario. Todas las réplicas cercanas caen "
            "en el mismo grupo, y el grupo entero queda en un solo pedazo. Así se ve con la secuencia más grande del "
            "dataset:")

conteo_grupos = modelados["grupo"].value_counts()
grupo_mayor = conteo_grupos.index[0]
secuencia = modelados[modelados["grupo"] == grupo_mayor].copy()
principal = secuencia.loc[secuencia["mag"].idxmax()]
fecha_principal = pd.Timestamp(principal["time"])
lat0, lon0 = int(np.floor(principal["latitude"] / 2) * 2), int(np.floor(principal["longitude"] / 2) * 2)
lat_a, lat_b = lat0 - 4, lat0 + 6
lon_a, lon_b = max(lon0 - 4, -180), max(lon0 - 4, -180) + 10
zona = modelados[(modelados["mes"] == principal["mes"]) & modelados["latitude"].between(lat_a, lat_b)
                 & modelados["longitude"].between(lon_a, lon_b)].copy()

# La zona dividida en las celdas de 2° × 2° que usan los grupos, con cuántos sismos tuvo cada una ese mes
zona["celda_lat"] = (np.floor(zona["latitude"] / 2) * 2).astype(int)
zona["celda_lon"] = (np.floor(zona["longitude"] / 2) * 2).astype(int)
por_celda = zona.groupby(["celda_lat", "celda_lon"]).size()
celdas = pd.DataFrame([(la, lo) for la in range(lat_a, lat_b, 2) for lo in range(lon_a, lon_b, 2)],
                      columns=["celda_lat", "celda_lon"])
celdas["sismos"] = [int(por_celda.get((la, lo), 0)) for la, lo in zip(celdas["celda_lat"], celdas["celda_lon"])]
celdas["lat2"], celdas["lon2"] = celdas["celda_lat"] + 2, celdas["celda_lon"] + 2
celdas["centro_lat"], celdas["centro_lon"] = celdas["celda_lat"] + 1, celdas["celda_lon"] + 1
celdas["oscura"] = celdas["sismos"] > conteo_grupos.iloc[0] * 0.4
GRADOS = "datum.value + '°'"
eje_x = alt.X("celda_lon:Q", title="longitud", scale=alt.Scale(domain=[lon_a, lon_b], nice=False),
              axis=alt.Axis(values=list(range(lon_a, lon_b + 1, 2)), labelExpr=GRADOS, grid=False))
eje_y = alt.Y("celda_lat:Q", title="latitud", scale=alt.Scale(domain=[lat_a, lat_b], nice=False),
              axis=alt.Axis(values=list(range(lat_a, lat_b + 1, 2)), labelExpr=GRADOS, grid=False))
tablero = alt.Chart(celdas).mark_rect(stroke="white", strokeWidth=2).encode(
    x=eje_x, x2="lon2:Q", y=eje_y, y2="lat2:Q",
    color=alt.Color("sismos:Q", legend=None, scale=alt.Scale(domain=[0, int(celdas["sismos"].max())],
                                                             range=["#f1f0ec", "#2f3a44"])),
    tooltip=[alt.Tooltip("celda_lat:Q", title="latitud desde"), alt.Tooltip("celda_lon:Q", title="longitud desde"),
             alt.Tooltip("sismos:Q", title="sismos en el mes")])
numeros = alt.Chart(celdas[celdas["sismos"] > 0]).mark_text(fontSize=15, fontWeight="bold").encode(
    x="centro_lon:Q", y="centro_lat:Q", text="sismos:Q",
    color=alt.condition("datum.oscura", alt.value("white"), alt.value("#555")))
marca_principal = alt.Chart(pd.DataFrame({"x": [principal["longitude"]], "y": [principal["latitude"]]})).mark_point(
    shape="diamond", filled=True, color="#e3242b", size=160).encode(x="x:Q", y="y:Q")
tablero_zona = (tablero + numeros + marca_principal).properties(
    width=430, height=430, title=f"Sismos por celda de 2° × 2° en {fecha_principal.strftime('%m/%Y')}")

# Un mapamundi chico, centrado en el Pacífico, con la zona marcada en rojo
recuadro = pd.DataFrame({"lon": [lon_a, lon_b, lon_a, lon_a], "lat": [lat_a, lat_a, lat_b, lat_a],
                         "lon2": [lon_b, lon_b, lon_b, lon_a], "lat2": [lat_a, lat_b, lat_b, lat_b]})
lados_zona = alt.Chart(recuadro).mark_rule(color="#e3242b", strokeWidth=2.5).encode(
    longitude="lon:Q", latitude="lat:Q", longitude2="lon2:Q", latitude2="lat2:Q")
ubicacion = alt.layer(fondo, lados_zona).project(type="equirectangular", rotate=[180, 0, 0]).properties(
    width=430, height=215, title="Dónde queda la zona (recuadro rojo)")

secuencia["dia"] = pd.to_datetime(secuencia["time"]).dt.day
por_dia = secuencia.groupby("dia").size().reindex(range(1, 32), fill_value=0).rename_axis("dia").reset_index(
    name="sismos")
por_dia["principal"] = np.where(por_dia["dia"] == fecha_principal.day, "día del sismo grande", "otro día")
barras_dia = alt.Chart(por_dia).mark_bar().encode(
    x=alt.X("dia:O", title=f"día de {fecha_principal.strftime('%m/%Y')}", axis=alt.Axis(labelAngle=0, values=[1, 5, 10, 15, 20, 25, 30])),
    y=alt.Y("sismos:Q", title=f"sismos en la celda de {conteo_grupos.iloc[0]}"),
    color=alt.Color("principal:N", legend=None,
                    scale=alt.Scale(domain=["día del sismo grande", "otro día"], range=["#e3242b", NEUTRO])),
    tooltip=[alt.Tooltip("dia:O", title="día"), "sismos:Q"]).properties(
    width=430, height=430, title="Sismos por día en esa celda")
grafico(alt.vconcat(ubicacion, alt.hconcat(tablero_zona, barras_dia, spacing=40), spacing=20))
st.markdown(f"En {fecha_principal.strftime('%m/%Y')} hubo {len(zona)} sismos en esta zona, y **{len(secuencia)} cayeron "
            f"en la misma celda**, la oscura (el rombo rojo es el sismo de {num(principal['mag'], 1)} del día "
            f"{fecha_principal.day}). Casi todos son sus réplicas, y se concentran en los días que le siguen. Cada "
            "cuadrado es una celda de 2° × 2° de latitud y longitud: esa celda, en ese mes, es un grupo.")

reparto_azar = secuencia["pedazo_azar"].value_counts().sort_index()
st.markdown(f"""
**Qué pasa con esa secuencia según cómo se parten los datos.** Para medir el modelo, los sismos se reparten en 5
pedazos: se entrena con cuatro y se evalúa con el quinto.

- **Al azar**, los {len(secuencia)} sismos de la celda quedan repartidos en los 5 pedazos
  ({', '.join(str(v) for v in reparto_azar.values[:-1])} y {reparto_azar.values[-1]}): cuando el modelo se evalúa en
  uno, ya vio a sus «hermanas» en los otros cuatro.
- **Por grupo**, los {len(secuencia)} quedan en el mismo pedazo: el modelo nunca ve la secuencia mientras se lo evalúa
  con ella.
""")

g1, g2, g3, g4 = st.columns(4)
g1.metric("grupos", miles(len(conteo_grupos)))
g2.metric("grupos con un solo sismo", miles(int((conteo_grupos == 1).sum())))
grandes_g = conteo_grupos[conteo_grupos >= 10]
g3.metric("grupos de 10 sismos o más", len(grandes_g),
          f"{pct(grandes_g.sum() / len(modelados))} de los sismos", delta_color="off")
g4.metric("el grupo más grande", f"{conteo_grupos.iloc[0]} sismos")

st.markdown("##### Las otras columnas del catálogo, y por qué no se usan")
filas_columnas = [
    ("sig", "numérica", "«Significancia» del evento según la USGS", "Fuga: la USGS la calcula con la alerta de tsunami"),
    ("nst, dmin, gap, rms", "numéricas", "Cómo registró el sismo la red de estaciones",
     "No mejoran el modelo, y la USGS las recalcula al revisar el evento"),
    ("time", "temporal", "Fecha y hora (UTC)", "No describe al sismo; sólo arma los grupos de validación"),
    ("place", "texto libre", "Descripción del lugar (de ahí sale la región)", "Identificador"),
    ("id", "identificador", "Código del sismo en el catálogo", "Identificador"),
    ("type, status, net", "categóricas", "Tipo de evento, estado de revisión y red que lo informó",
     "Constantes o casi: no varían entre sismos"),
]
tabla_columnas = pd.DataFrame(filas_columnas, columns=["columna", "tipo", "qué es", "por qué no se usa"])
st.dataframe(tabla_columnas, hide_index=True)
nulos_nst = sismos["nst"].isna().mean()
st.caption(f"Los únicos nulos del dataset están en las cuatro columnas de la red de estaciones (`nst` {pct(nulos_nst)}, "
           "el resto menos de 1 %): son de instrumentación, no errores de carga. Las otras 14 columnas están completas.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**De dónde salen.** De la API pública de la USGS (`earthquake.usgs.gov/fdsnws/event/1/query`), en GeoJSON y sin clave.
Los baja un pipeline de Airflow con arquitectura Medallón:

1. `wait_for_source`: un sensor que pregunta cuántos sismos hay en la ventana; si hay cero, saltea el resto.
2. `extract` (Bronce): guarda el GeoJSON crudo, sin tocarlo, con las fechas de la ventana en el nombre.
3. `transform` (Plata): lo aplana en una tabla de 17 columnas, un sismo por fila.
4. `enrich` (Plata): agrega nuestra columna, `dist_costa_km`.
5. `validate`: corta el pipeline si el `id` se repite, si hay menos de 1.000 filas, menos de 5 columnas o alguna
   columna vacía del todo.
6. `save` (Oro): agrega los sismos a `sismos_master.csv` y reemplaza los que ya estaban por `id` (*upsert*), así
   las correcciones de la USGS pisan la versión vieja.
7. `notify`: avisa por Discord que terminó.

Corre **todos los días a las 02:00** y trae los sismos del día anterior. Esta aplicación usa una copia congelada de esa
salida (`{resumen['datos']['archivo']}`, del {resumen['datos']['desde']} al {resumen['datos']['hasta']}), para que todos
los números se puedan reproducir.

**Qué es una fila.** Un sismo de magnitud 5 o más. Por debajo de 5 la posibilidad de tsunami es prácticamente nula.

**Nuestra columna, `dist_costa_km`.** La distancia del epicentro a la costa más cercana, con signo:
- la distancia sale de la fórmula de Haversine, que tiene en cuenta la curvatura de la Tierra, contra la línea de costa
  de Natural Earth (escala 50 m);
- tierra o mar sale de *ray casting*: se traza un rayo desde el epicentro y se cuenta cuántas veces cruza el borde de
  los polígonos de tierra (impar = adentro de un continente);
- negativa en tierra, positiva en el mar.

**Qué se filtró y qué quedó.**
""")
    st.dataframe(pd.DataFrame(resumen["filtros"]).rename(
        columns={"paso": "Paso", "filas": "Sismos", "con_marca": "Con marca"}), hide_index=True)
    recientes_ = sismos[sismos["conjunto"] == "sin marca cargada"]
    st.markdown(f"**Por qué el corte en el 31/03/2026.** Desde abril de 2026 la proporción de sismos con marca cae a "
                f"casi cero: de los {miles(len(recientes_))} sismos de abril a julio, sólo "
                f"{int(recientes_['tsunami'].sum())} la tienen, y hay sismos de 7,2 a 7,8 en el mar sin marca. No dejó "
                "de haber tsunamis: la USGS carga la marca cuando revisa el evento, y la descarga tomó esos sismos "
                "antes. Si se dejaban, el modelo aprendía que esos sismos no se marcan.")
    sep_sig = separacion(con_marca["sig"], sin_marca["sig"])
    _, s_ms, _ = asociacion(modelados["mag"], modelados["sig"])
    con_sig = {a["variante"]: a for a in exp["ablacion"]}["+ sig (fuga)"]
    base_ = {a["variante"]: a for a in exp["ablacion"]}["base"]
    st.markdown(f"""
**La columna descartada por fuga: `sig`.** Es un índice de la USGS que separa muy bien las dos clases (separación
{num(sep_sig, 2)}, zona {semaforo(sep_sig)}), pero porque la USGS lo calcula **usando la alerta de tsunami**: contiene la
respuesta. El número solo no lo delataba: su correlación con la marca es apenas
{num(modelados['tsunami'].corr(modelados['sig']), 2)}, y es casi una copia de la magnitud (Spearman {num(s_ms, 2)}). La
fuga se detecta leyendo cómo se construye la columna. En el modelo la medimos igual: con `sig` el F1 da
{num(con_sig['f1'])} y sin ella {num(base_['f1'])}; sale por cómo se construye, no por el número.

**Las demás columnas que salen:** `id`, `place` y `time` son identificadores (`time` sólo arma los grupos de
validación); `status` tiene un solo valor, `type` es «earthquake» en casi todas las filas y `net` es «us» en el
{pct((sismos['net'] == 'us').mean(), 1)}; `nst`, `dmin`, `gap` y `rms` no mejoran el modelo y la USGS las recalcula al
revisar el evento.
""")

st.divider()


# =================================================================================================
# TALLO 2 · LO QUE ENCONTRAMOS
# =================================================================================================
st.header("Lo que encontramos")
st.subheader("La magnitud es la que más separa, pero no es una regla: ni los sismos más grandes llevan siempre la marca")

modelados["rango_mag"] = pd.cut(modelados["mag"], CORTES_MAG, labels=RANGOS_MAG)
por_mag = (modelados.groupby(["rango_mag", "en_el_mar"], observed=True)["tsunami"]
           .agg(sismos="size", con_marca="sum").reset_index())
por_mag["proporcion"] = por_mag["con_marca"] / por_mag["sismos"]
por_mag["rango_mag"] = por_mag["rango_mag"].astype(str)
lineas_mag = alt.Chart(por_mag).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=3).encode(
    x=alt.X("rango_mag:O", sort=RANGOS_MAG, title="magnitud", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("proporcion:Q", title="sismos con marca de la USGS", axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("en_el_mar:N", title="epicentro", scale=ESCALA_ZONA, legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("rango_mag:O", title="magnitud"), alt.Tooltip("en_el_mar:N", title="epicentro"),
             alt.Tooltip("sismos:Q"), alt.Tooltip("con_marca:Q", title="con marca"),
             alt.Tooltip("proporcion:Q", title="con marca", format=".1%")])
etiquetas_mag = lineas_mag.mark_text(dy=-12, fontSize=11).encode(text=alt.Text("proporcion:Q", format=".0%"))
regla = alt.Chart(pd.DataFrame({"y": [1.0], "texto": ["si la magnitud fuera una regla, todo llegaría acá (100 %)"]}))
regla_linea = regla.mark_rule(color=GRIS, strokeDash=[6, 4]).encode(y="y:Q")
regla_texto = regla.mark_text(align="left", x=5, dy=-8, color=GRIS, fontSize=11).encode(y="y:Q", text="texto:N")
grafico((lineas_mag + etiquetas_mag + regla_linea + regla_texto).properties(width=900, height=330))

ult = por_mag[por_mag["rango_mag"] == RANGOS_MAG[-1]].set_index("en_el_mar")
st.markdown(f"La marca sube con la magnitud: la tienen menos de 2 de cada 100 sismos de 5,0 a 5,5 y "
            f"{pct(ult.loc['mar', 'proporcion'])} de los de más de 7,5 en el mar. Pero **nunca llega al 100 %**, y en "
            f"tierra queda más abajo ({pct(ult.loc['tierra', 'proporcion'])}, sobre sólo "
            f"{int(ult.loc['tierra', 'sismos'])} sismos).")

st.markdown("##### ¿Qué separa a los sismos con marca de los sin marca?")
mar_ = modelados[modelados["dist_costa_km"] > 0]
seps = pd.DataFrame([
    ("Magnitud", separacion(con_marca["mag"], sin_marca["mag"]), "se usa"),
    ("sig (índice de la USGS)", separacion(con_marca["sig"], sin_marca["sig"]), "fuga: se descarta"),
    ("Distancia a la costa, sólo en el mar",
     separacion(mar_.loc[mar_["tsunami"] == 1, "dist_costa_km"], mar_.loc[mar_["tsunami"] == 0, "dist_costa_km"]),
     "se usa"),
    ("Distancia a la costa", separacion(con_marca["dist_costa_km"], sin_marca["dist_costa_km"]), "se usa"),
    ("Profundidad", separacion(con_marca["depth"], sin_marca["depth"]), "se usa"),
], columns=["variable", "separacion", "estado"])
seps["texto"] = [f"{num(v, 2)}" + ("  · fuga: se descarta" if e != "se usa" else "")
                 for v, e in zip(seps["separacion"], seps["estado"])]
orden_vars = seps["variable"].tolist()
x_max = 1.6
franjas = pd.DataFrame({"desde": [0, 0.2, 0.8], "hasta": [0.2, 0.8, x_max],
                        "zona": ["zona roja", "zona amarilla", "zona verde"]})
fondo_sem = alt.Chart(franjas).mark_rect(opacity=0.35).encode(
    x=alt.X("desde:Q", title="separación (desvíos entre el promedio con marca y sin marca)",
            scale=alt.Scale(domain=[0, x_max])), x2="hasta:Q",
    color=alt.Color("zona:N", title="semáforo de la cátedra", legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["zona roja", "zona amarilla", "zona verde"],
                                    range=["#f4b6b0", "#f7e3a1", "#bfe3c0"])))
barras_sep = alt.Chart(seps).mark_bar(size=22).encode(
    y=alt.Y("variable:N", sort=orden_vars, title=None, axis=alt.Axis(labelFontSize=12)),
    x="separacion:Q",
    color=alt.Color("estado:N", scale=alt.Scale(domain=["se usa", "fuga: se descarta"], range=[NEUTRO, "#b9b6b0"]),
                    legend=None),
    tooltip=["variable:N", alt.Tooltip("separacion:Q", format=".3f"), "estado:N"])
texto_sep = barras_sep.mark_text(align="left", dx=5, color="#222", fontSize=12).encode(text="texto:N")
grafico((fondo_sem + barras_sep + texto_sep).properties(width=900, height=230))

# --- La profundidad: lo que la intuición física decía y no se sostuvo
st.markdown("##### La profundidad no distingue: los sismos con marca no son más superficiales")
st.caption("La hipótesis física decía lo contrario. Cada par de barras compara el mismo tramo de profundidad en los "
           "dos grupos, sobre el mismo eje: si la hipótesis se sostuviera, la barra azul sería más larga que la gris "
           "en «superficial» y más corta en las otras dos.")
sep_prof = separacion(con_marca["depth"], sin_marca["depth"])
prof = modelados.assign(rango=pd.cut(modelados["depth"], CORTES_PROF, labels=RANGOS_PROF))
reparto = prof.groupby(["marca", "rango"], observed=True).size().reset_index(name="sismos")
reparto["parte"] = reparto["sismos"] / reparto.groupby("marca")["sismos"].transform("sum")
reparto["rango"] = reparto["rango"].astype(str)
reparto["orden"] = reparto["rango"].map({r_: i for i, r_ in enumerate(RANGOS_PROF)})
sup = reparto[reparto["orden"] == 0].set_index("marca")["parte"]
EJE_PROF = alt.Y("rango:N", sort=RANGOS_PROF, title=None,
                 axis=alt.Axis(labelFontSize=13, labelLimit=240, ticks=False, domain=False))
PAR_PROF = alt.YOffset("marca:N", sort=[CON, SIN])
X_PARTE = alt.X("parte:Q", title="parte de los sismos de cada grupo", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]))   # dominio fijo: las dos capas comparten la misma escala
barras_p = alt.Chart(reparto).mark_bar(cornerRadiusEnd=3).encode(
    y=EJE_PROF, yOffset=PAR_PROF, x=X_PARTE,
    color=alt.Color("marca:N", sort=[CON, SIN], scale=ESCALA_MARCA, legend=None),
    tooltip=[alt.Tooltip("marca:N", title="grupo"), alt.Tooltip("rango:N", title="profundidad"),
             alt.Tooltip("sismos:Q", title="sismos"),
             alt.Tooltip("parte:Q", title="parte del grupo", format=".1%")])
texto_p = alt.Chart(reparto).mark_text(align="left", dx=6, color="#222", fontSize=13).encode(
    y=EJE_PROF, yOffset=PAR_PROF, x=X_PARTE, text=alt.Text("parte:Q", format=".1%"))
# Rótulos directos dentro del par más largo, en lugar de una leyenda aparte
rotulos_p = alt.Chart(reparto[reparto["orden"] == 0]).mark_text(align="left", x=8, fontSize=13,
                                                               fontWeight="bold").encode(
    y=EJE_PROF, yOffset=PAR_PROF, text=alt.Text("marca:N"),
    color=alt.condition(alt.datum.marca == CON, alt.value("white"), alt.value("#4a4a46")))
grafico((barras_p + texto_p + rotulos_p).properties(width=900, height=230))
st.markdown(f"Un tsunami lo genera un sismo que mueve el fondo del mar, y eso pasa con los sismos superficiales. Por eso "
            f"esperábamos que los sismos con marca fueran más superficiales. **No es así:** {pct(sup[CON], 1)} de los "
            f"sismos con marca tiene menos de 70 km de profundidad, pero también {pct(sup[SIN], 1)} de los sin marca. "
            "Las dos barras son casi iguales, porque los sismos profundos son raros en general (separación "
            f"{num(sep_prof, 3)}, zona {semaforo(sep_prof)}). Y {int((con_marca['depth'] > 300).sum())} sismos de más de "
            "300 km tienen la marca, aunque un sismo tan profundo casi no mueve el fondo del mar: la marca de la USGS no "
            "mira la profundidad.")

# --- La matriz de correlación de la Entrega 2 (sobre los 13.602 sismos, igual que el notebook)
st.markdown("##### ¿Hay columnas que repiten la misma información?")
COLS_CORR = ["mag", "depth", "dist_costa_km", "latitude", "longitude", "nst", "gap", "dmin", "rms", "sig", "tsunami"]
matriz = sismos[COLS_CORR].corr()
largo = matriz.reset_index().melt(id_vars="index", var_name="columna", value_name="r").rename(columns={"index": "fila"})
posicion = {c: i for i, c in enumerate(COLS_CORR)}
largo = largo[[posicion[f] >= posicion[c] for f, c in zip(largo["fila"], largo["columna"])]].copy()  # triángulo inferior
largo["fuerte"] = largo["r"].abs() >= 0.6
celdas_c = alt.Chart(largo).mark_rect(stroke="white", strokeWidth=1).encode(
    x=alt.X("columna:N", sort=COLS_CORR, title=None, axis=alt.Axis(labelAngle=-45, orient="bottom")),
    y=alt.Y("fila:N", sort=COLS_CORR, title=None),
    color=alt.Color("r:Q", title="correlación de Pearson",
                    scale=alt.Scale(scheme="redblue", reverse=True, domain=[-1, 1]),
                    legend=alt.Legend(orient="right", gradientLength=300)),
    tooltip=["fila:N", "columna:N", alt.Tooltip("r:Q", format=".2f")])
textos_c = alt.Chart(largo).mark_text(fontSize=10).encode(
    x=alt.X("columna:N", sort=COLS_CORR), y=alt.Y("fila:N", sort=COLS_CORR), text=alt.Text("r:Q", format=".2f"),
    color=alt.condition("datum.fuerte", alt.value("white"), alt.value("#222")))
pares_clave = [("mag", "sig"), ("dist_costa_km", "dmin"), ("nst", "gap"), ("mag", "nst"), ("depth", "dist_costa_km")]
filas_pares = []
for a_, b_ in pares_clave:
    p_, s_, br_ = asociacion(sismos[a_], sismos[b_])
    filas_pares.append({"par": f"{a_} – {b_}", "Pearson": round(p_, 2), "Spearman": round(s_, 2),
                        "brecha": round(br_, 2), "zona (Spearman)": semaforo(s_, "correlacion")})
mc1, mc2 = st.columns([1.35, 1])
with mc1:
    grafico((celdas_c + textos_c).properties(width=470, height=470,
                                             title="Matriz de correlación entre columnas numéricas"))
with mc2:
    st.markdown("**Los pares que se destacan**, con Pearson (relación en línea recta), Spearman (relación que sube o "
                "baja aunque no sea recta) y la brecha entre los dos:")
    st.dataframe(pd.DataFrame(filas_pares), hide_index=True)
    st.markdown(f"""
- **`sig` es casi una copia de la magnitud** (Spearman {num(filas_pares[0]['Spearman'], 2)}): otra señal de que no trae
  información nueva, sólo la alerta.
- **`dist_costa_km` y `dmin` se parecen** ({num(filas_pares[1]['Pearson'], 2)}): las estaciones están en tierra, así que
  cuanto más mar adentro, más lejos queda la estación más cercana.
- **Ningún par de columnas candidatas pasa 0,6**, así que no se descartó nada por redundancia.
- **En la fila de `tsunami`, ninguna columna se acerca a la marca por sí sola:** la más alta es la magnitud
  ({num(matriz.loc['tsunami', 'mag'], 2)}).
""")
st.caption(f"Calculada sobre los {miles(n_total)} sismos, igual que en el notebook de la Entrega 2. Rojo: correlación "
           "positiva; azul: negativa. Sólo se muestra la mitad de abajo, porque la de arriba repite los mismos valores.")

# --- El resumen
tipos_m = modelados.groupby("magType")["tsunami"].agg(sismos="size", con_marca="sum")
st.markdown(
    f"**En resumen: la magnitud es la única variable física que separa bien.** Las otras dos que dicta la intuición "
    f"física casi no separan: los sismos con marca no son más superficiales, y la distancia a la costa distingue algo "
    f"sólo si se mira dentro del mar. `sig` separa muy bien, pero porque la USGS la calcula con la alerta: es una fuga y "
    f"se descartó (sección «Los datos»). Ninguna columna repite a otra, salvo `sig` a la magnitud. Además, el **tipo de "
    f"magnitud** cambia mucho las cosas: de los {miles(int(tipos_m.loc['mb', 'sismos']))} sismos medidos en `mb` sólo "
    f"{int(tipos_m.loc['mb', 'con_marca'])} tienen marca, contra "
    f"{pct(tipos_m.loc['mww', 'con_marca'] / tipos_m.loc['mww', 'sismos'], 1)} en `mww`.  \n"
    "Lo que el análisis no podía explicar es por qué hay sismos grandes sin marca y sismos chicos con marca. "
    "Eso lo resolvió el modelo: **el lugar**.")

with st.expander("Cómo lo sabemos"):
    st.markdown("**La separación** es la herramienta de la cátedra para medir una hipótesis con un número:")
    st.latex(r"\text{separación} = \frac{|\bar{x}_{\text{con marca}} - \bar{x}_{\text{sin marca}}|}"
             r"{\sqrt{(s^2_{\text{con marca}} + s^2_{\text{sin marca}})/2}}")
    grandes = modelados[modelados["mag"] > 7.5]
    excepciones = grandes[grandes["tsunami"] == 0]
    sep_mag = separacion(con_marca["mag"], sin_marca["mag"])
    st.markdown(f"""
Es la distancia entre los dos promedios medida en desvíos. Menos de 0,2 es zona roja, de 0,2 a 0,8 amarilla y más de 0,8
verde. Un verde no alcanza para usar una columna (`sig`), y un rojo no alcanza para descartarla (la profundidad entró
igual, por motivo físico).

**La hipótesis que se sostuvo, como tendencia: la magnitud** (separación {num(sep_mag, 2)}, zona {semaforo(sep_mag)}).
La pregunta era si todo sismo de más de 7,5 tiene marca: {len(excepciones)} de los {len(grandes)}
({pct(len(excepciones) / len(grandes), 1)}) no la tienen. Buena variable, mal umbral fijo.

**La que no se sostuvo: la profundidad** (separación {num(sep_prof, 3)}, zona {semaforo(sep_prof)}). La pregunta era si
los sismos con marca son más superficiales: el gráfico de arriba muestra que no.
""")
    st.markdown(f"""
**Sobre los números.** En la Entrega 2 las hipótesis se midieron sobre los {miles(n_total)} sismos. Acá se miden sobre
los {miles(len(modelados))} que usa el modelo, porque desde la Entrega 3 sabemos que los de abril a julio de 2026 no
tienen la marca cargada. Las separaciones cambian en el tercer decimal; lo único que se mueve es la proporción de sismos
de más de 7,5 sin marca: 21,7 % en la Entrega 2 y {pct(len(excepciones) / len(grandes), 1)} acá. La matriz de
correlación sí se muestra sobre los {miles(n_total)} sismos, igual que en el notebook.
""")

st.divider()


# =================================================================================================
# TALLO 3 · EL MODELO
# =================================================================================================
st.header("El modelo")
st.subheader(f"El modelo detecta {test['aciertos_con_marca']} de cada "
             f"{test['aciertos_con_marca'] + test['falsos_negativos']} sismos con marca, "
             f"a cambio de una falsa alarma cada "
             f"{round((test['verdaderos_negativos'] + test['falsas_alarmas']) / test['falsas_alarmas'])} sismos sin marca")

r1, r2, r3 = st.columns(3)
r1.metric("sismos con marca detectados en el test",
          f"{test['aciertos_con_marca']} de {test['aciertos_con_marca'] + test['falsos_negativos']}",
          help=f"Recall de la clase con marca, con umbral {num(UMBRAL, 2)}")
r2.metric("F1 en el test", num(test["f1"]),
          f"el piso (siempre «sin marca») da {num(resumen['piso_test']['f1'])}", delta_color="off",
          help="F1 de la clase con marca: combina el recall y la precisión de esa clase")
r3.metric("alertas correctas", f"{round(test['precision_tsunami'] * 100)} de cada 100",
          help="Precisión de la clase con marca en el test")

# Matriz de confusión del test: filas = lo que pasó, columnas = lo que dijo el modelo, diagonal = aciertos
conf = (sismos[sismos["conjunto"] == "test"]
        .assign(real=lambda d: d["tsunami"].map({1: CON, 0: SIN}),
                dijo=lambda d: d["alerta"].map({1: "alerta", 0: "sin alerta"}))
        .groupby(["real", "dijo", "resultado"], as_index=False).size())
totales_conf = conf.groupby("real")["size"].sum()
conf["fila"] = [f"{r} · {miles(totales_conf[r])} sismos" for r in conf["real"]]
conf["caso"] = ["acierto" if r.startswith("acierto") else r for r in conf["resultado"]]
conf["cuenta"] = conf["size"].map(miles)
conf["parte"] = conf["size"] / conf["real"].map(totales_conf)
FILAS_CONF = [f"{c} · {miles(totales_conf[c])} sismos" for c in (CON, SIN)]
Y_CONF = alt.Y("fila:N", sort=FILAS_CONF, title="de verdad, el sismo…",
               axis=alt.Axis(labelFontSize=13, labelLimit=260, ticks=False, domain=False))
X_CONF = alt.X("dijo:N", sort=["alerta", "sin alerta"], title="lo que dijo el modelo",
               axis=alt.Axis(labelFontSize=13, labelAngle=0, orient="top", ticks=False, domain=False))
TINTA_CONF = alt.condition(alt.datum.resultado == "acierto: sin marca",   # la única celda de fondo claro
                           alt.value("#2b2b29"), alt.value("white"))
celdas_conf = alt.Chart(conf).mark_rect(stroke="white", strokeWidth=3).encode(
    x=X_CONF, y=Y_CONF,
    color=alt.Color("resultado:N", scale=ESCALA_RESULTADO, legend=None),   # el color significa lo mismo que en el mapa
    tooltip=[alt.Tooltip("resultado:N", title="caso"), alt.Tooltip("size:Q", title="sismos"),
             alt.Tooltip("parte:Q", title="parte de la fila", format=".1%")])
cuenta_conf = alt.Chart(conf).mark_text(dy=-12, fontSize=26, fontWeight="bold").encode(
    x=X_CONF, y=Y_CONF, text=alt.Text("cuenta:N"), color=TINTA_CONF)
caso_conf = alt.Chart(conf).mark_text(dy=18, fontSize=13).encode(
    x=X_CONF, y=Y_CONF, text=alt.Text("caso:N"), color=TINTA_CONF)
grafico((celdas_conf + cuenta_conf + caso_conf).properties(width=640, height=260))
st.markdown(f"Es un gradient boosting que usa la magnitud, la ubicación, la profundidad, el tipo de magnitud y si el "
            f"epicentro está en el mar. Con umbral {num(UMBRAL, 2)}, de los "
            f"{test['aciertos_con_marca'] + test['falsos_negativos']} sismos con marca del test detecta "
            f"{test['aciertos_con_marca']} ({pct(test['recall_tsunami'])}) y pierde {test['falsos_negativos']}: ese es "
            f"el error caro. A cambio da {test['falsas_alarmas']} falsas alarmas entre "
            f"{miles(test['verdaderos_negativos'] + test['falsas_alarmas'])} sismos sin marca: de cada 100 alertas, "
            f"{round(test['precision_tsunami'] * 100)} son correctas.")

chicos = modelados[(modelados["tsunami"] == 1) & (modelados["mag"] < 6)]
st.markdown("##### En qué se apoya: la ubicación pesa tanto como la magnitud")
nombres_col = {"longitude": "longitud", "mag": "magnitud", "latitude": "latitud", "magType": "tipo de magnitud",
               "dist_costa_km": "mar o tierra", "depth": "profundidad"}
imp = pd.DataFrame(exp["importancia"])
imp["columna"] = imp["columna"].map(nombres_col)
imp["bajo"], imp["alto"] = imp["caida"] - imp["desvio"], imp["caida"] + imp["desvio"]
orden_imp = imp.sort_values("caida", ascending=False)["columna"].tolist()
barras_i = alt.Chart(imp).mark_bar(color=NEUTRO).encode(
    y=alt.Y("columna:N", sort=orden_imp, title=None),
    x=alt.X("caida:Q", title="cuánto cae el F1 al desordenarla"),
    tooltip=["columna:N", alt.Tooltip("caida:Q", format=".3f"), alt.Tooltip("desvio:Q", format=".3f")])
bigotes_i = alt.Chart(imp).mark_rule(color="#222").encode(y=alt.Y("columna:N", sort=orden_imp), x="bajo:Q", x2="alto:Q")

cortes = exp["arbol"]["cortes"]
lat_c, lon_c, mag_c = cortes.get("latitude", 40.2), cortes.get("longitude", -116.6), cortes.get("mag", 6.05)
mapa = modelados.assign(grupo=lambda d: d["tsunami"].map({0: "sin marca", 1: "con marca, magnitud 6 o más"}).where(
    ~((d["tsunami"] == 1) & (d["mag"] < 6)), "con marca, magnitud menor a 6"))
dominio_g = ["sin marca", "con marca, magnitud 6 o más", "con marca, magnitud menor a 6"]
puntos_g = alt.Chart(mapa[["longitude", "latitude", "grupo", "tsunami", "place", "mag"]]).mark_circle().encode(
    longitude="longitude:Q", latitude="latitude:Q",
    color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=dominio_g, range=[GRIS_CLARO, AZUL_CLARO, AZUL])),
    size=alt.Size("grupo:N", legend=None, scale=alt.Scale(domain=dominio_g, range=[3, 14, 26])),
    order=alt.Order("tsunami:Q"),
    tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud")])
lados = pd.DataFrame({"lon": [-179.9, lon_c, -179.9, -179.9], "lat": [lat_c, lat_c, 75, lat_c],
                      "lon2": [lon_c, lon_c, lon_c, -179.9], "lat2": [lat_c, 75, 75, 75]})
caja = alt.Chart(lados).mark_rule(color="#e3242b", strokeWidth=2, strokeDash=[6, 3]).encode(
    longitude="lon:Q", latitude="lat:Q", longitude2="lon2:Q", latitude2="lat2:Q")
texto_caja = alt.Chart(pd.DataFrame({"lon": [lon_c + 3], "lat": [72.0], "t": ["la «caja» del árbol"]})).mark_text(
    align="left", color="#e3242b", fontSize=12, fontWeight="bold").encode(longitude="lon:Q", latitude="lat:Q", text="t:N")
izq, der = st.columns([1, 1.9])
with izq:
    grafico((barras_i + bigotes_i).properties(width=330, height=230))
with der:
    grafico(alt.layer(fondo, puntos_g, caja, texto_caja).project(type="equirectangular")
            .properties(width=620, height=310))
f1_elegido = {c["modelo"]: c for c in exp["comparacion"]}["Gradient boosting ajustado (elegido)"]["f1"]
st.markdown(f"**A la izquierda:** si se desordena una columna y el modelo empeora mucho, es que se apoyaba en ella. "
            "Longitud, magnitud y latitud pesan parecido; el tipo de magnitud, algo; la profundidad y mar o tierra, "
            "nada. La ubicación no es fuga: las coordenadas se conocen apenas ocurre el sismo y no se calculan con la "
            "marca.  \n"
            f"**A la derecha, por qué pesa el lugar:** los puntos azul oscuro son los {len(chicos)} sismos con marca y "
            f"magnitud menor a 6, y **{chicos['region'].isin(REGION_ALASKA_OESTE).sum()} de ellos están en Alaska y la "
            "costa oeste**, donde la USGS tiene un sistema de alerta "
            f"propio con un umbral más bajo. El rectángulo rojo es la regla que encuentra un árbol de sólo tres preguntas "
            f"(F1 {num(exp['arbol']['f1'])}): si la magnitud es mayor a {num(mag_c, 2)}, casi siempre hay marca; si no, "
            f"la hay al norte de {num(lat_c, 1)}° y al oeste de {num(lon_c, 1)}°. El boosting llega a "
            f"{num(f1_elegido)} combinando muchos cortes de este tipo.")

recientes = sismos[sismos["conjunto"] == "sin marca cargada"]
grandes_recientes = recientes[recientes["mag"] >= 7.2]
with st.container(border=True):
    st.markdown("#### Las conclusiones")
    st.markdown(f"""
- **La respuesta a la pregunta:** lo que permite estimar si un sismo lleva la marca de tsunami de la USGS es **su
  magnitud y su ubicación**, casi por partes iguales. El tipo de magnitud aporta algo; la profundidad, la distancia a la
  costa y las métricas de la red de estaciones, nada.
- **Lo que el modelo explicó y el análisis no:** por qué hay sismos grandes sin marca y chicos con marca. La marca sigue
  la regla regional de alerta de la USGS, además de la física del sismo.
- **Lo que predice es la marca, no la ola.** El modelo aprendió el criterio de la USGS, incluido su sesgo regional: donde
  la USGS casi no marca sismos (por ejemplo, el Caribe), el modelo tampoco.
- **Para comprobarlo con datos nuevos:** de los {miles(len(recientes))} sismos de abril a julio de 2026, que la USGS
  todavía no marcó, el modelo da alerta en {int(recientes['alerta'].sum())}, entre ellos los {len(grandes_recientes)} de
  magnitud 7,2 o más. Cuando la USGS cargue esas marcas se va a poder ver cuántos acertó.
""")

with st.expander("Cómo lo sabemos"):
    g = resumen["grupos"]
    comparacion = pd.DataFrame(exp["comparacion"])
    piso = comparacion.loc[comparacion["modelo"] == "Piso: siempre «sin marca»", "f1"].iloc[0]
    ablacion = {a["variante"]: a for a in exp["ablacion"]}
    base_f1, base_desvio = ablacion["base"]["f1"], ablacion["base"]["desvio"]
    f1_02_05 = [c["F1"] for c in exp["curva_umbral"] if 0.2 <= c["umbral"] <= 0.5]
    st.markdown(f"""
**La métrica: el F1 de la clase con marca.** Combina la **precisión** (de las alertas que da el modelo, cuántas son
correctas) y el **recall** (de los sismos con marca, cuántos encuentra): sólo es alto si las dos lo son. Mira únicamente la
clase que interesa, así que los aciertos en los sismos sin marca, que son casi todos, no lo inflan. El error caro es el
falso negativo (un sismo con alerta que el modelo no detecta); una falsa alarma cuesta menos.

**El piso:** {num(piso)}. Un `DummyClassifier` que dice siempre «sin marca» acierta el 96 % de los sismos, pero no
encuentra ninguno con marca.

**Sobre el cambio de métrica.** En la Entrega 3 las decisiones se tomaron con el f1 macro (el promedio del F1 de las dos
clases). Repetidas con el F1 de la clase con marca, no cambia ninguna: el mismo modelo, los mismos hiperparámetros y el
mismo umbral.

**La validación.** Un sismo grande trae réplicas en el mismo lugar durante días: si una réplica queda en entrenamiento y
otra en validación, el modelo «reconoce» la secuencia y el número sale inflado. Por eso los pedazos se arman con
`StratifiedGroupKFold` (5 pedazos), con grupos de **celda de 2° × 2° de latitud y longitud y mes**: {miles(g['cantidad'])}
grupos; el más grande son los {g['mayor_sismos']} sismos de la secuencia de Kermadec de marzo de 2021. El test (20 %) se
separó con el mismo criterio y se miró una sola vez. ¿Importa cómo se arman los pedazos?
""")
    part = pd.DataFrame(exp["particiones"])
    part["valor"] = [f"{num(f)} ± {num(d)}" for f, d in zip(part["f1"], part["desvio"])]
    st.dataframe(part.pivot(index="modelo", columns="forma", values="valor")[["Al azar", "Por grupo", "Por fecha"]])
    st.markdown("Dan lo mismo, siempre dentro del desvío: **las réplicas no inflan el número**, porque son casi todas "
                "chicas y sin marca. Igual nos quedamos con la validación por grupo, porque imita el uso real.")

    st.markdown("**La comparación de modelos** (F1 en validación cruzada, media ± desvío):")
    modelos = comparacion[~comparacion["modelo"].str.startswith("Piso")].copy()
    modelos["bajo"], modelos["alto"] = modelos["f1"] - modelos["desvio"], modelos["f1"] + modelos["desvio"]
    modelos["elegido"] = np.where(modelos["modelo"].str.contains("elegido"), "elegido", "probado")
    modelos["texto"] = [f"{num(f)} ± {num(d)}" for f, d in zip(modelos["f1"], modelos["desvio"])]
    orden_modelos = modelos.sort_values("f1")["modelo"].tolist()
    color_elegido = alt.Color("elegido:N", scale=alt.Scale(domain=["elegido", "probado"], range=[AZUL, NEUTRO]),
                              legend=None)
    eje_m = alt.Y("modelo:N", sort=orden_modelos, title=None)
    bigotes = alt.Chart(modelos).mark_rule(strokeWidth=2).encode(
        x=alt.X("bajo:Q", title="F1", scale=alt.Scale(domain=[0, 0.9])), x2="alto:Q", color=color_elegido,
        y=eje_m)
    puntos_m = alt.Chart(modelos).mark_circle(size=110).encode(x="f1:Q", color=color_elegido, y=eje_m,
                                                               tooltip=["modelo:N", "texto:N"])
    textos_m = alt.Chart(modelos).mark_text(align="left", dx=10, color="#222").encode(x="alto:Q", text="texto:N", y=eje_m)
    linea_piso = alt.Chart(pd.DataFrame({"x": [piso], "t": [f"piso: {num(piso)}"]}))
    grafico((bigotes + puntos_m + textos_m + linea_piso.mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="x:Q")
             + linea_piso.mark_text(align="left", dx=4, y=8, color=GRIS).encode(x="x:Q", text="t:N"))
            .properties(width=760, height=210))
    rf = comparacion.set_index("modelo").loc["Random forest"]
    gb = comparacion.set_index("modelo").loc["Gradient boosting"]
    logi = comparacion.set_index("modelo").loc["Regresión logística"]
    st.markdown(f"""
Random forest y gradient boosting **empatan**: la diferencia ({num(rf['f1'] - gb['f1'])}) es menor que el desvío
({num(gb['desvio'])}). Se eligió el boosting porque entrena unas tres veces más rápido y da un modelo más liviano para la
app. La logística pierde porque sólo puede sumar columnas con pesos («más al norte, más marca») y no puede dibujar la caja
de Alaska: sin latitud y longitud baja de {num(logi['f1'])} a {num(exp['logistica_sin_ubicacion']['f1'])}, y el boosting
de {num(base_f1)} a {num(ablacion['− latitud y longitud']['f1'])}.

**El ajuste:** `GridSearchCV` probó 36 combinaciones (tasa de aprendizaje, cantidad de árboles, hojas por árbol y
balanceo de clases). Ganó tasa {num(resumen['hiperparametros']['learning_rate'], 2)},
{resumen['hiperparametros']['max_iter']} árboles, {resumen['hiperparametros']['max_leaf_nodes']} hojas y sin balancear
({num(f1_elegido)}). Balancear las clases empató con el mejor y fue menos estable.

**El umbral.** El modelo da una probabilidad, y el umbral decide a partir de cuánto eso es «alerta». En la Entrega 3
usamos 0,5; la corrección fue bajarlo. El F1 es casi plano entre 0,2 y 0,5 (de {num(min(f1_02_05))} a
{num(max(f1_02_05))}, una diferencia menor que el desvío), así que bajar el umbral no empeora la métrica y deja de perder
sismos marcados. Se eligió {num(UMBRAL, 2)}, el umbral con el F1 más alto, con validación cruzada y sin mirar el test:
""")
    curva = pd.DataFrame(exp["curva_umbral"]).melt(id_vars="umbral", var_name="medida", value_name="valor")
    medidas = ["recall", "precisión", "F1"]
    lineas_u = alt.Chart(curva).mark_line(strokeWidth=2.5).encode(
        x=alt.X("umbral:Q", title="umbral"),
        y=alt.Y("valor:Q", title="validación cruzada", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("medida:N", title=None, sort=medidas,
                        scale=alt.Scale(domain=medidas, range=["#222222", "#9a9a9a", "#2e7d32"]),
                        legend=alt.Legend(orient="top")),
        tooltip=["umbral:Q", "medida:N", alt.Tooltip("valor:Q", format=".3f")])
    marcas_u = alt.Chart(pd.DataFrame({"umbral": [0.5, UMBRAL], "t": ["antes: 0,5", f"elegido: {num(UMBRAL, 2)}"],
                                       "c": [GRIS, AZUL]}))
    grafico((lineas_u + marcas_u.mark_rule(strokeDash=[4, 4]).encode(x="umbral:Q", color=alt.Color("c:N", scale=None))
             + marcas_u.mark_text(align="left", dx=4, y=10).encode(x="umbral:Q", text="t:N",
                                                                    color=alt.Color("c:N", scale=None)))
            .properties(width=760, height=240))
    st.markdown(f"""
En validación cruzada el recall sube de {pct(vc05['recall_tsunami'], 1)} a {pct(vc['recall_tsunami'], 1)} y la precisión
baja de {pct(vc05['precision_tsunami'], 1)} a {pct(vc['precision_tsunami'], 1)}.

**El test, una sola vez** (umbral {num(UMBRAL, 2)}): F1 {num(test['f1'])} (validación cruzada:
{num(vc['f1'])}), recall {pct(test['recall_tsunami'])}, precisión {pct(test['precision_tsunami'])}, PR-AUC
{num(test['pr_auc'], 2)} contra {num(resumen['piso_test']['pr_auc'], 2)} del piso.

**Lo que probamos y no anduvo** (la ablación: el modelo base contra el mismo modelo agregando o sacando una columna por
vez; la franja gris es ± un desvío del base, y lo que cae adentro es un empate):
""")
    abl = pd.DataFrame([a for a in exp["ablacion"] if a["variante"] != "base"])
    abl["diferencia"] = abl["f1"] - base_f1
    abl["lectura"] = np.where(abl["diferencia"].abs() > base_desvio, "supera el desvío", "empate")
    franja = alt.Chart(pd.DataFrame({"a": [-base_desvio], "b": [base_desvio]})).mark_rect(
        color="#e8e6e1").encode(x="a:Q", x2="b:Q")
    barras_a = alt.Chart(abl).mark_bar().encode(
        y=alt.Y("variante:N", sort=abl.sort_values("diferencia")["variante"].tolist(), title=None),
        x=alt.X("diferencia:Q", title="diferencia de F1 con el modelo base"),
        color=alt.Color("lectura:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["supera el desvío", "empate"], range=[NEUTRO, "#b9b6b0"])),
        tooltip=["variante:N", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("diferencia:Q", format="+.3f")])
    grafico((franja + barras_a).properties(width=700, height=300))
    st.markdown("Sólo sacar la latitud y la longitud supera el desvío. Las métricas de la red (`nst`, `dmin`, `gap`, "
                "`rms`) no mejoran nada, ni solas ni juntas; la distancia continua empata con mar o tierra; la "
                "profundidad empata; `sig` tampoco cambia el número.")

    casos = sismos[sismos["conjunto"] == "test"]
    columnas_e = {"time": "fecha (UTC)", "place": "lugar", "mag": "magnitud", "depth": "profundidad (km)",
                  "en_el_mar": "epicentro", "probabilidad": "probabilidad"}
    fn_ = casos[casos["resultado"] == "falso negativo"].sort_values("probabilidad")
    st.markdown(f"**Dónde se equivoca.** Los {len(fn_)} falsos negativos del test:")
    st.dataframe(fn_[list(columnas_e)].rename(columns=columnas_e), hide_index=True)
    st.markdown(f"""
Son sismos de 6,3 a 6,5 fuera de Alaska (en tierra o profundos), sismos chicos en el mar frente a California y Alaska que
la USGS marcó con su umbral regional, y uno en el Caribe, una región casi sin marcas. Las falsas alarmas más seguras son
sismos fuertes en el mar que, por magnitud y lugar, son el caso típico que la USGS marca, pero esta vez no lo hizo.

**Limitaciones.** Los grupos de validación son una aproximación: una secuencia de réplicas que cruza el borde de una
celda o el cambio de mes queda partida (igual, partir al azar dio lo mismo). Los sismos desde abril de 2026 quedaron
afuera hasta que la USGS cargue su marca. Y la importancia por permutación usa la predicción por defecto del modelo, que
corta en 0,5 y no en {num(UMBRAL, 2)}; el orden de las columnas no depende del umbral.
""")

st.divider()


# =================================================================================================
# COPA · EXPLORÁ (EJES + MAPA DE CALOR + CLIC A LA UBICACIÓN)
# =================================================================================================
OPCION_CERO = "— armarlo desde cero —"   # la primera opción de «Partir de un sismo real», en «Probá el modelo»
ROJO = "#e3242b"                          # tu sismo, el umbral y la caja del árbol (no es una clase)
fondo_recortado = alt.Chart(MUNDO).mark_geoshape(fill="#eeeeee", stroke="white", strokeWidth=0.5, clip=True)


def mapa(capas, encuadre=None, centro_lon=0.0, alto=300):
    """Las capas sobre el mapamundi, en proyección equirectangular.

    `encuadre` = (lon_min, lat_min, lon_max, lat_max) es la zona que tiene que entrar en el gráfico (por defecto, el mundo
    sin la Antártida). La proyección se gira para que `centro_lon` quede en el medio: así el zoom funciona igual cerca
    de la línea de cambio de fecha (Fiyi, Tonga, Aleutianas). Lo que queda afuera del encuadre se recorta.
    """
    lon0, lat0, lon1, lat1 = encuadre or (-179.9, -60, 179.9, 80)
    lat0, lat1 = max(lat0, -89.9), min(lat1, 89.9)
    caja_geo = {"type": "Feature", "properties": {},
                "geometry": {"type": "MultiPoint", "coordinates": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]]}}
    return (alt.layer(fondo_recortado, *capas)
            .project(type="equirectangular", rotate=[-centro_lon, 0, 0], fit=caja_geo)
            .properties(height=alto))


def elegido_en(evento, seleccion, campo):
    """El valor de `campo` del punto que el usuario tocó en un gráfico de Altair con on_select (None si no tocó nada)."""
    try:
        puntos = evento["selection"][seleccion]
    except (KeyError, TypeError):
        return None
    if isinstance(puntos, list) and puntos and isinstance(puntos[0], dict):
        return puntos[0].get(campo)
    return None


def seccion_explora():
    st.header("Tu turno · Explorá")
    st.markdown("Cada punto es un sismo, con el color de lo que dijo el modelo. **Hacé clic en cualquier punto** para "
                "ver toda su información y saltar directo a su ubicación en el mapa.")
    recientes = sismos[sismos["conjunto"] == "sin marca cargada"]
    grandes_recientes = recientes[recientes["mag"] >= 7.2]
    st.markdown(f"En gris oscuro están los {miles(len(recientes))} sismos que la USGS todavía no marcó. El modelo da "
                f"alerta en {int(recientes['alerta'].sum())}, entre ellos {'los' if len(grandes_recientes) > 1 else 'el'} "
                f"{len(grandes_recientes)} de magnitud 7,2 o más "
                f"({', '.join(f'{r.region} {num(r.mag, 1)}' for r in grandes_recientes.itertuples())}). Cuando la USGS "
                "cargue esas marcas, van a servir para comprobar al modelo con datos que nunca vio.")

    if "explora_sismo_id" not in st.session_state:
        st.session_state["explora_sismo_id"] = None

    # 1. Enfoques rápidos
    opciones_enfoque = [
        "Aciertos con marca y errores",
        "Sólo aciertos con marca",
        "Sólo errores del modelo",
        "Sismos 2026 sin marca cargada",
        "Todos los sismos (catálogo completo)",
    ]
    if hasattr(st, "segmented_control"):
        enfoque = st.segmented_control("¿Qué querés inspeccionar?", opciones_enfoque, default=opciones_enfoque[0])
        if not enfoque:
            enfoque = opciones_enfoque[0]
    else:
        enfoque = st.radio("¿Qué querés inspeccionar?", opciones_enfoque, horizontal=True)

    # 2. Filtros + selector de eje Y
    ejes_y = {
        "Profundidad (km)": ("depth", "Profundidad (km)"),
        "Probabilidad del modelo (%)": ("prob_pct_num", "Probabilidad según el modelo (%)"),
        "Distancia a la costa (km)": ("dist_costa_km", "Distancia a la costa (+ mar / − tierra, km)"),
    }
    c1, c2, c3, c4 = st.columns([1.25, 1.25, 1.15, 0.95])
    eje_y_nombre = c1.selectbox("Eje vertical (Y)", list(ejes_y.keys()))
    mag_min, mag_max = c2.slider("Rango de magnitud (eje X)", 5.0, 9.0, (5.0, 9.0), step=0.1, format="%.1f")
    region = c3.selectbox("Región", ["Todas"] + sorted(sismos["region"].dropna().unique()))
    epicentro = c4.radio("Epicentro", ["Todos", "mar", "tierra"], horizontal=True)

    col_y, titulo_y = ejes_y[eje_y_nombre]

    # Aplicación de filtros
    vista = sismos[(sismos["mag"] >= mag_min) & (sismos["mag"] <= mag_max)].copy()
    if enfoque == "Aciertos con marca y errores":
        vista = vista[vista["resultado"].isin(["acierto: con marca", "falso negativo", "falsa alarma"])]
    elif enfoque == "Sólo aciertos con marca":
        vista = vista[vista["resultado"] == "acierto: con marca"]
    elif enfoque == "Sólo errores del modelo":
        vista = vista[vista["resultado"].isin(["falso negativo", "falsa alarma"])]
    elif enfoque == "Sismos 2026 sin marca cargada":
        vista = vista[vista["conjunto"] == "sin marca cargada"]

    if region != "Todas":
        vista = vista[vista["region"] == region]
    if epicentro != "Todos":
        vista = vista[vista["en_el_mar"] == epicentro]

    if len(vista) == 0:
        st.info("No hay sismos que cumplan con esta combinación de filtros. Probá ampliar el rango de magnitud o "
                "cambiar el enfoque.")
        return

    vista["marca_real"] = vista["tsunami"].map({1: "Sí", 0: "No"}).where(
        vista["conjunto"] != "sin marca cargada", "Todavía no cargada"
    )
    vista["prob_pct_num"] = (vista["probabilidad"] * 100).round(1)
    vista["prob_txt"] = (vista["probabilidad"] * 100).round(0).astype(int).astype(str) + " %"
    vista["fecha"] = vista["time"].str.slice(0, 16).str.replace("T", " ")

    # 3. Números que siguen a los filtros
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sismos filtrados", miles(len(vista)))
    k2.metric("Con marca USGS", miles(int((vista["marca_real"] == "Sí").sum())))
    k3.metric("Alertas del modelo", miles(int(vista["alerta"].sum())), help=f"Probabilidad ≥ {num(UMBRAL, 2)}")
    k4.metric("Magnitud máxima", num(vista["mag"].max(), 1))

    # El sismo elegido (si sigue dentro de los filtros)
    fila = None
    id_sel = st.session_state["explora_sismo_id"]
    if id_sel is not None:
        match = vista[vista["id"].astype(str) == str(id_sel)]
        if not match.empty:
            fila = match.iloc[0]

    # 4. Dos columnas: izquierda = gráfico de ejes; derecha = ficha + ubicación
    col_graf, col_ubic = st.columns([1.45, 1.0])

    with col_graf:
        eje_x = alt.X("mag:Q", title="Magnitud del sismo", scale=alt.Scale(zero=False))
        eje_y = alt.Y(f"{col_y}:Q", title=titulo_y)
        capas = []

        # Un punto por sismo, con el color de lo que dijo el modelo (si hay más de 2.500 grises «sin marca»,
        # se muestrean sólo esos; los errores y los aciertos con marca van todos)
        muestreo_grises = False
        orden_capas = ["acierto: sin marca", "sin marca cargada", "falsa alarma", "acierto: con marca", "falso negativo"]
        partes = []
        for i, res in enumerate(orden_capas):
            sub = vista[vista["resultado"] == res]
            if res == "acierto: sin marca" and len(sub) > 2500:
                sub = sub.sample(2500, random_state=42)
                muestreo_grises = True
            partes.append(sub.assign(capa=i))
        de_puntos = pd.concat(partes)
        gris = de_puntos["resultado"] == "acierto: sin marca"
        de_puntos["tam"] = np.where(gris, 30, 75)
        de_puntos["opac"] = np.where(gris, 0.45, 0.92)
        columnas = list(dict.fromkeys(["id", "mag", col_y, "resultado", "place", "fecha", "depth", "en_el_mar",
                                       "marca_real", "prob_txt", "tam", "opac", "capa"]))

        elegir = alt.selection_point(name="sismo", fields=["id"], on="click")
        capas.append(alt.Chart(de_puntos[columnas]).mark_circle(stroke="white", strokeWidth=0.6).encode(
            x=eje_x, y=eje_y,
            color=alt.Color("resultado:N", scale=ESCALA_RESULTADO, title="qué dijo el modelo",
                            legend=alt.Legend(orient="top", columns=3)),
            size=alt.Size("tam:Q", scale=None),
            opacity=alt.Opacity("opac:Q", scale=None),
            order=alt.Order("capa:Q"),
            tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("fecha:N", title="fecha (UTC)"),
                     alt.Tooltip("mag:Q", title="magnitud"), alt.Tooltip("depth:Q", title="profundidad (km)"),
                     alt.Tooltip("en_el_mar:N", title="mar o tierra"),
                     alt.Tooltip("marca_real:N", title="marca de la USGS"),
                     alt.Tooltip("prob_txt:N", title="probabilidad según el modelo"),
                     alt.Tooltip("resultado:N", title="resultado")],
        ).add_params(elegir))

        # Línea de referencia según el eje Y elegido
        referencia = None
        if col_y == "prob_pct_num":
            referencia = (UMBRAL * 100, f"umbral de alerta ({pct(UMBRAL)})", ROJO)
        elif col_y == "dist_costa_km":
            referencia = (0, "línea de costa (arriba mar / abajo tierra)", NEUTRO)
        if referencia:
            linea = alt.Chart(pd.DataFrame({"y": [referencia[0]], "t": [referencia[1]]}))
            capas += [linea.mark_rule(color=referencia[2], strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q"),
                      linea.mark_text(align="right", baseline="bottom", dx=-4, dy=-3, color=referencia[2]).encode(
                          y="y:Q", x=alt.value("width"), text="t:N")]

        # El sismo elegido, con un anillo negro
        if fila is not None:
            capas.append(alt.Chart(pd.DataFrame({"mag": [fila["mag"]], col_y: [fila[col_y]]})).mark_point(
                size=380, strokeWidth=3, color="#111111", filled=False).encode(x="mag:Q", y=f"{col_y}:Q"))

        ejes = alt.layer(*capas).properties(height=500)
        evento = st.altair_chart(ejes, theme=None, on_select="rerun", selection_mode=["sismo"], key="explora_ejes")
        id_tocado = elegido_en(evento, "sismo", "id")
        if id_tocado is not None and str(id_tocado) != str(st.session_state["explora_sismo_id"]):
            st.session_state["explora_sismo_id"] = str(id_tocado)
            st.rerun()

        if muestreo_grises:
            st.caption("Para mantener ágil el navegador se muestran todos los casos con marca y los errores, y una "
                       "muestra de 2.500 sismos grises sin marca (filtrá por región o magnitud para ver el 100 %). "
                       "Los números de arriba sí cuentan todos los sismos filtrados.")
        else:
            st.caption("Tocá cualquier punto para ver su ficha y ubicarlo en el mapa.")

    # 5. Panel derecho: ficha completa del sismo tocado + mapa centrado en su ubicación exacta
    with col_ubic:
        st.markdown("##### Sismo seleccionado y ubicación")
        punto_mapa = dict(longitude="longitude:Q", latitude="latitude:Q",
                          color=alt.Color("resultado:N", scale=ESCALA_RESULTADO, legend=None))

        if fila is None:
            st.info("**Tocá cualquier punto** en el gráfico de la izquierda para ver su ficha completa y acercarte "
                    "a su ubicación exacta en el mapa.")
            ref = vista[["longitude", "latitude", "resultado"]].head(1500)
            grafico(mapa([alt.Chart(ref).mark_circle(size=12, opacity=0.5, clip=True).encode(**punto_mapa)], alto=310))
        else:
            lat_s, lon_s = float(fila["latitude"]), float(fila["longitude"])
            dlat = (vista["latitude"] - lat_s).abs()
            dlon = ((vista["longitude"] - lon_s + 180) % 360 - 180).abs()
            vecinos = vista[(dlat <= 10) & (dlon <= 14) & (vista["id"].astype(str) != str(fila["id"]))].head(120)

            capa_vecinos = alt.Chart(vecinos[["longitude", "latitude", "resultado", "place", "mag"]]).mark_circle(
                size=45, opacity=0.5, clip=True).encode(
                **punto_mapa, tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud")])
            capa_elegido = alt.Chart(pd.DataFrame([fila[["longitude", "latitude", "resultado", "place"]]])).mark_circle(
                size=260, opacity=1, stroke="white", strokeWidth=3, clip=True).encode(
                **punto_mapa, tooltip=[alt.Tooltip("place:N", title="lugar")])
            grafico(mapa([capa_vecinos, capa_elegido], encuadre=(lon_s - 14, lat_s - 10, lon_s + 14, lat_s + 10),
                         centro_lon=lon_s, alto=265))
            st.caption("El sismo elegido, grande y con borde blanco; alrededor, los otros sismos filtrados de la zona, "
                       "con el mismo color de lo que dijo el modelo.")

            # Ficha completa del sismo
            with st.container(border=True):
                st.markdown(f"**{fila['place']}**")
                st.caption(f"Fecha: {str(fila['time'])[:19].replace('T', ' ')} UTC · Conjunto: `{fila['conjunto']}`")
                i1, i2, i3 = st.columns(3)
                i1.metric("Magnitud", num(fila["mag"], 1), help=f"Tipo: {fila['magType']}")
                i2.metric("Profundidad", f"{num(fila['depth'], 0)} km")
                i3.metric("Prob. modelo", fila["prob_txt"])
                st.markdown(
                    f"- **Resultado:** `{fila['resultado']}` · **Marca USGS:** **{fila['marca_real']}**\n"
                    f"- **Ubicación:** {fila['en_el_mar']} ({num(fila['dist_costa_km'], 0)} km a la costa) · "
                    f"Lat `{num(lat_s, 2)}`, Lon `{num(lon_s, 2)}`\n"
                    f"- **Tipo de magnitud:** `{fila['magType']}`"
                )

            # Botones para abrir la ubicación real o cargarlo en «Probá el modelo»
            b1, b2 = st.columns(2)
            b1.link_button("Abrir en Google Maps", f"https://www.google.com/maps?q={lat_s},{lon_s}", width="stretch")
            b2.link_button("Ficha oficial USGS",
                           f"https://earthquake.usgs.gov/earthquakes/eventpage/{fila['id']}/executive",
                           width="stretch")

            if st.button("Probar este sismo en «Probá el modelo» ↓", width="stretch", type="primary"):
                r_ = resumen["rangos"]
                tipos_ = resumen["magtypes"]
                st.session_state["_proba_pendiente"] = {
                    "proba_mag": float(np.clip(round(float(fila["mag"]), 1), r_["mag"][0], r_["mag"][1])),
                    "proba_depth": float(np.clip(round(float(fila["depth"])), r_["depth"][0], r_["depth"][1])),
                    "proba_lat": float(np.clip(round(lat_s, 1), r_["latitude"][0], r_["latitude"][1])),
                    "proba_lon": float(np.clip(round(lon_s, 1), r_["longitude"][0], r_["longitude"][1])),
                    "proba_mar": "mar" if fila["en_el_mar"] == "mar" else "tierra",
                    "proba_magtype": fila["magType"] if fila["magType"] in tipos_ else "mww",
                    "proba_zoom_zona": str(fila["region"]) if pd.notna(fila["region"]) else str(fila["place"]),
                    "proba_ejemplo_sel": OPCION_CERO,
                    "proba_ejemplo_prev": OPCION_CERO,
                    "proba_mapa_ver": st.session_state.get("proba_mapa_ver", 0) + 1,
                    "_desde_explora": True,
                }
                st.rerun()

    # 6. Tabla con los sismos filtrados
    with st.expander(f"Ver detalle tabular ({miles(len(vista))} sismos filtrados, ordenados por magnitud)"):
        cols = ["time", "place", "mag", "magType", "depth", "dist_costa_km", "en_el_mar", "marca_real",
                "prob_txt", "resultado", "conjunto"]
        renombre = {
            "time": "Fecha (UTC)", "place": "Lugar", "mag": "Magnitud", "magType": "Tipo",
            "depth": "Profundidad (km)", "dist_costa_km": "Dist. costa (km)", "en_el_mar": "Epicentro",
            "marca_real": "Marca USGS", "prob_txt": "Prob. modelo", "resultado": "Resultado", "conjunto": "Conjunto",
        }
        st.dataframe(
            vista.sort_values(["mag", "probabilidad"], ascending=False)[cols].head(300).rename(columns=renombre),
            hide_index=True,
            width="stretch",
        )


seccion_explora()
st.divider()


# =================================================================================================
# COPA · PROBÁ EL MODELO (CONTROLES A LA IZQUIERDA, MAPA CLICABLE CON ZOOM POR ZONA A LA DERECHA)
# =================================================================================================
def seccion_proba():
    st.header("Tu turno · Probá el modelo")
    st.markdown("Partí de un sismo real, **hacé clic en una zona del mapa** para acercarte y ubicar tu sismo ahí, o "
                "ajustá los controles a mano. Los controles van del mínimo al máximo que vio el entrenamiento, así que "
                "no se puede cargar un valor que el modelo no conoce.")

    r = resumen["rangos"]
    tipos = resumen["magtypes"]

    def dentro(valor, rango_):
        return float(min(max(float(valor), float(rango_[0])), float(rango_[1])))

    defaults = {
        "proba_mag": dentro(7.0, r["mag"]),
        "proba_depth": dentro(25.0, r["depth"]),
        "proba_lat": dentro(-33.5, r["latitude"]),
        "proba_lon": dentro(-72.0, r["longitude"]),
        "proba_mar": "mar",
        "proba_magtype": "mww" if "mww" in tipos else tipos[0],
        "proba_zoom_zona": None,
        "proba_ejemplo_prev": OPCION_CERO,
        "proba_mapa_ver": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Lo que dejó cargado un clic en el mapa o el botón de «Explorá» (se aplica antes de dibujar los controles)
    pend = st.session_state.pop("_proba_pendiente", None)
    if pend:
        desde_explora = pend.pop("_desde_explora", False)
        for k, v in pend.items():
            st.session_state[k] = v
        if desde_explora:
            st.toast("El sismo de «Explorá» quedó cargado en «Probá el modelo».")

    def ejemplo(filtro, por, descripcion):
        """El sismo real con el mayor valor de `por` entre los que cumplen el filtro."""
        candidatos = modelados[filtro(modelados)]
        return (descripcion, candidatos.loc[candidatos[por].idxmax()]) if len(candidatos) else None

    ejemplos = [e for e in [
        ejemplo(lambda d: d["mag"] > 0, "mag", "El más grande del catálogo"),
        ejemplo(lambda d: (d["region"] == "Alaska") & (d["tsunami"] == 1) & (d["mag"] < 5.5), "probabilidad",
                "Un sismo chico de Alaska, con marca"),
        ejemplo(lambda d: (d["en_el_mar"] == "tierra") & (d["mag"] >= 6.8), "mag", "Un sismo fuerte en tierra"),
        ejemplo(lambda d: d["depth"] > 300, "mag", "Un sismo grande y profundo"),
        ejemplo(lambda d: (d["magType"] == "mb") & (d["tsunami"] == 0), "mag", "El mayor sismo informado en mb"),
    ] if e]
    opciones = [OPCION_CERO] + [f"{d}: {s['place']} ({num(s['mag'], 1)}, {s['time'][:10]})" for d, s in ejemplos]

    # Dos columnas: los controles a la izquierda y el mapa a la derecha, para ver la latitud y la longitud junto al mapa
    izq, der = st.columns([1, 1.35], gap="large")

    with izq:
        elegido = st.selectbox("Partir de un sismo real", opciones, key="proba_ejemplo_sel")
        real = ejemplos[opciones.index(elegido) - 1][1] if elegido != OPCION_CERO else None

        if elegido != st.session_state["proba_ejemplo_prev"]:
            st.session_state["proba_ejemplo_prev"] = elegido
            if real is not None:
                st.session_state["proba_mag"] = dentro(round(float(real["mag"]), 1), r["mag"])
                st.session_state["proba_depth"] = dentro(round(float(real["depth"])), r["depth"])
                st.session_state["proba_lat"] = dentro(round(float(real["latitude"]), 1), r["latitude"])
                st.session_state["proba_lon"] = dentro(round(float(real["longitude"]), 1), r["longitude"])
                st.session_state["proba_mar"] = "mar" if real["en_el_mar"] == "mar" else "tierra"
                st.session_state["proba_magtype"] = (real["magType"] if real["magType"] in tipos
                                                     else defaults["proba_magtype"])
                st.session_state["proba_zoom_zona"] = (str(real["region"]) if pd.notna(real["region"])
                                                       else "Sismo elegido")
            else:
                st.session_state["proba_zoom_zona"] = None
            st.session_state["proba_mapa_ver"] += 1

        if real is not None:
            st.caption(f"En la realidad, la USGS {'**lo marcó**' if real['tsunami'] == 1 else '**no lo marcó**'}. "
                       "Los controles arrancan con sus valores (redondeados): movelos o cliqueá otro punto en el mapa.")

        mag = st.slider("Magnitud", float(r["mag"][0]), float(r["mag"][1]), step=0.1, format="%.1f", key="proba_mag")
        depth = st.slider("Profundidad (km)", float(r["depth"][0]), float(r["depth"][1]), step=1.0,
                          key="proba_depth")
        lat = st.slider("Latitud", float(r["latitude"][0]), float(r["latitude"][1]), step=0.1, format="%.1f",
                        key="proba_lat")
        lon = st.slider("Longitud", float(r["longitude"][0]), float(r["longitude"][1]), step=0.1, format="%.1f",
                        key="proba_lon")
        en_el_mar = st.radio("¿El epicentro está en el mar o en tierra?", ["mar", "tierra"], horizontal=True,
                             key="proba_mar") == "mar"

        with st.expander("Opciones avanzadas: tipo de magnitud"):
            mag_type = st.selectbox("Tipo de magnitud", tipos, format_func=lambda m: f"{NOMBRES_MAGTYPE.get(m, m)} ({m})",
                                    help="Sólo se ofrecen los tipos con al menos 20 sismos en el entrenamiento.",
                                    key="proba_magtype")
            cuota_mww = (modelados["magType"] == "mww").mean()
            st.caption(f"Es el método con que la USGS calculó la magnitud del sismo. Si no sabés cuál elegir, dejá "
                       f"*mww*: es la de {round(cuota_mww * 10)} de cada 10 sismos y la que la USGS usa para los sismos "
                       "grandes. Qué significa cada una está en el cuadro de abajo.")

        if mag_type == "mb" and mag >= 6.5:
            todos_mb = sismos[sismos["magType"] == "mb"]
            st.warning(f"Combinación casi inexistente: la escala *mb* se satura cerca de magnitud 6, así que la USGS "
                       f"casi no informa en mb sismos tan grandes. En los datos hay {(todos_mb['mag'] >= 6.5).sum()} "
                       f"de {miles(len(todos_mb))} (el mayor, de {num(todos_mb['mag'].max(), 1)}). El modelo casi no "
                       "vio casos así: tomá esta probabilidad con cuidado.")

        probabilidad = guardado["pipeline"].predict_proba(
            modelo.caso(mag, depth, lat, lon, mag_type, en_el_mar))[0, 1]
        alerta = probabilidad >= UMBRAL

        m1, m2 = st.columns(2)
        tarjeta_kpi(m1, pct(probabilidad), "probabilidad de que la USGS le ponga la marca de alerta")
        tarjeta_kpi(m2, "Sí · Alerta" if alerta else "No · Sin alerta",
                    f"decisión del modelo con el umbral elegido ({num(UMBRAL, 2)})")

        rango_elegido = RANGOS_MAG[int(np.digitize(mag, CORTES_MAG[1:-1], right=True))]
        mismo_rango = modelados[modelados["rango_mag"].astype(str) == rango_elegido]
        misma_zona = mismo_rango[mismo_rango["en_el_mar"] == ("mar" if en_el_mar else "tierra")]
        st.caption(f"Para comparar: en los datos, los sismos de magnitud {rango_elegido} tienen la marca el "
                   f"{pct(mismo_rango['tsunami'].mean())} de las veces ({pct(misma_zona['tsunami'].mean())} los que "
                   f"ocurren en {'el mar' if en_el_mar else 'tierra'}). La diferencia con la probabilidad del modelo la "
                   "pone, sobre todo, el lugar.")

    with der:
        # --- El mapa, en 2 niveles: el mundo con las zonas sísmicas, o el zoom en una zona con sus sismos reales ---
        zona_enfocada = st.session_state["proba_zoom_zona"]
        b1, b2 = st.columns([3, 1.3])
        if zona_enfocada is None:
            b1.markdown("##### Ubicación en el mapa · *hacé clic en una zona para acercarte*")
            if b2.button("Acercar a mi sismo", width="stretch"):
                st.session_state["proba_zoom_zona"] = "Ubicación elegida"
                st.session_state["proba_mapa_ver"] += 1
                st.rerun()
        else:
            b1.markdown(f"##### Zoom en **{zona_enfocada}** · *cliqueá un sismo cercano para mover tu punto ahí*")
            if b2.button("← Volver a vista mundial", width="stretch", type="primary"):
                st.session_state["proba_zoom_zona"] = None
                st.session_state["proba_mapa_ver"] += 1
                st.rerun()

        # La caja del árbol (la misma de «El modelo»), recortada al borde del mapa
        capas = [caja.mark_rule(color=ROJO, strokeWidth=2, strokeDash=[6, 3], clip=True),
                 texto_caja.mark_text(align="left", color=ROJO, fontSize=12, fontWeight="bold", clip=True)]

        if zona_enfocada is None:
            # NIVEL 1: el mundo, con las zonas sísmicas (celdas de 22° × 22° con 20 sismos o más) para hacer clic
            paso = 22
            tmp = modelados.assign(
                zid=(np.floor(modelados["latitude"] / paso) * paso).astype(int).astype(str)
                + "_"
                + (np.floor(modelados["longitude"] / paso) * paso).astype(int).astype(str)
            )
            zonas = (
                tmp.groupby("zid", as_index=False)
                .agg(
                    lat=("latitude", "mean"),
                    lon=("longitude", "mean"),
                    sismos=("mag", "size"),
                    con_marca=("tsunami", "sum"),
                    nombre=("region", _nombre_principal),
                    mar=("en_el_mar", lambda s: "mar" if (s == "mar").mean() >= 0.5 else "tierra"),
                )
            )
            zonas = zonas[zonas["sismos"] >= 20].copy()
            zonas["tasa"] = zonas["con_marca"] / zonas["sismos"]
            seleccion = "zona"
            capas.append(alt.Chart(zonas).mark_circle(color=NEUTRO, opacity=0.35, stroke="white", strokeWidth=1,
                                                      clip=True).encode(
                longitude="lon:Q", latitude="lat:Q",
                size=alt.Size("sismos:Q", scale=alt.Scale(type="sqrt", range=[150, 900]), legend=None),
                tooltip=[alt.Tooltip("nombre:N", title="zona"),
                         alt.Tooltip("sismos:Q", title="sismos registrados", format=","),
                         alt.Tooltip("tasa:Q", title="con marca", format=".1%")],
            ).add_params(alt.selection_point(name=seleccion, fields=["zid"], on="click")))
            encuadre, centro_lon = None, 0.0
        else:
            # NIVEL 2: zoom en la zona + sismos reales cercanos de magnitud parecida
            dlat = (modelados["latitude"] - lat).abs()
            dlon = ((modelados["longitude"] - lon + 180) % 360 - 180).abs()
            cercanos = modelados[(dlat <= 12) & (dlon <= 16) & ((modelados["mag"] - mag).abs() <= 0.6)].copy()
            if len(cercanos) > 180:
                marcados_c = cercanos[cercanos["tsunami"] == 1]
                no_marcados_c = cercanos[cercanos["tsunami"] == 0].head(max(180 - len(marcados_c), 0))
                cercanos = pd.concat([marcados_c, no_marcados_c])
            seleccion = "cercano"
            capas.append(alt.Chart(cercanos[["id", "longitude", "latitude", "marca", "place", "mag", "depth",
                                             "en_el_mar"]]).mark_circle(opacity=0.85, stroke="white", strokeWidth=0.7,
                                                                        clip=True).encode(
                longitude="longitude:Q", latitude="latitude:Q",
                color=alt.Color("marca:N", scale=ESCALA_MARCA, title="sismos reales cercanos",
                                legend=alt.Legend(orient="top")),
                size=alt.Size("marca:N", scale=alt.Scale(domain=[CON, SIN], range=[140, 70]), legend=None),
                order=alt.Order("mag:Q"),
                tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud real"),
                         alt.Tooltip("depth:Q", title="profundidad (km)"),
                         alt.Tooltip("en_el_mar:N", title="mar o tierra"), alt.Tooltip("marca:N", title="marca")],
            ).add_params(alt.selection_point(name=seleccion, fields=["id"], on="click")))
            encuadre, centro_lon = (lon - 16, lat - 12, lon + 16, lat + 12), lon

        # Tu sismo: un punto rojo con borde blanco
        tu_sismo = pd.DataFrame({"longitude": [lon], "latitude": [lat],
                                 "t": [f"Tu sismo: magnitud {num(mag, 1)}, probabilidad {pct(probabilidad)}"]})
        capas.append(alt.Chart(tu_sismo).mark_circle(size=320, color=ROJO, opacity=1, stroke="white", strokeWidth=2.5,
                                                     clip=True).encode(
            longitude="longitude:Q", latitude="latitude:Q", tooltip=alt.Tooltip("t:N", title=" ")))

        clave_mapa = f"mapa_proba_{'mundo' if zona_enfocada is None else 'zoom'}_{st.session_state['proba_mapa_ver']}"
        evento = st.altair_chart(mapa(capas, encuadre, centro_lon, alto=560), theme=None, on_select="rerun",
                                 selection_mode=[seleccion], key=clave_mapa)
        st.caption("El punto rojo es tu sismo. La línea roja punteada es la «caja» del árbol de «El modelo»: al norte y "
                   "al oeste de ella, la USGS marca sismos más chicos.")

        tocado = elegido_en(evento, seleccion, "zid" if seleccion == "zona" else "id")
        if tocado is not None:
            if seleccion == "zona":
                z = zonas[zonas["zid"] == str(tocado)]
                destino = None if z.empty else (z.iloc[0]["lat"], z.iloc[0]["lon"], z.iloc[0]["mar"], z.iloc[0]["nombre"])
            else:
                c = cercanos[cercanos["id"].astype(str) == str(tocado)]
                destino = None if c.empty else (c.iloc[0]["latitude"], c.iloc[0]["longitude"], c.iloc[0]["en_el_mar"],
                                                zona_enfocada)
            if destino is not None:
                st.session_state["_proba_pendiente"] = {
                    "proba_lat": dentro(round(float(destino[0]), 1), r["latitude"]),
                    "proba_lon": dentro(round(float(destino[1]), 1), r["longitude"]),
                    "proba_mar": "mar" if destino[2] == "mar" else "tierra",
                    "proba_zoom_zona": str(destino[3]),
                    "proba_mapa_ver": st.session_state["proba_mapa_ver"] + 1,
                }
                st.rerun()

    en_mb = modelados[modelados["magType"] == "mb"]
    with st.expander("¿Qué significa cada tipo de magnitud?"):
        conteo_tipos = modelados.groupby("magType")["tsunami"].agg(["size", "sum"])
        filas = []
        for m in tipos:
            como, cuando = EXPLICACION_MAGTYPE.get(m, ("—", "—"))
            n, k = ((int(conteo_tipos.loc[m, "size"]), int(conteo_tipos.loc[m, "sum"]))
                    if m in conteo_tipos.index else (0, 0))
            en_datos = f"{miles(n)} sismos · {pct(k / n, 1)} con marca" if n else "—"
            filas.append(f"| **{m}** | {NOMBRES_MAGTYPE.get(m, m)} | {como} | {cuando} | {en_datos} |")
        st.markdown(f"""
La **magnitud** mide el tamaño de un sismo, pero hay varias formas de calcularla. La sigla indica qué método usó la
USGS (o la red que informó el sismo). Hay dos familias:

- **Magnitudes de momento** (las que empiezan con *mw*): se calculan a partir del momento sísmico, es decir, de qué tan
  grande es la falla que se rompió y cuánto se deslizó. Es la medida física más completa y **no se satura**: sirve
  igual para un sismo de 5 que para uno de 9.
- **Magnitudes de amplitud** (*mb* y *ml*): miden qué tan fuerte se mueve un tipo de onda en el sismógrafo. Son más
  rápidas de calcular, pero **se saturan**: en los sismos grandes dejan de crecer aunque el sismo sea mayor.

| Sigla | Nombre | Cómo se calcula | Cuándo la usa la USGS | En los datos (hasta marzo de 2026) |
|---|---|---|---|---|
""" + "\n".join(filas) + f"""

**Por qué cambia la predicción.** Las magnitudes de momento dan resultados casi iguales entre sí. La que cambia todo es
*mb*: casi ningún sismo informado en mb tiene marca ({int(en_mb['tsunami'].sum())} de {miles(len(en_mb))}), porque la
USGS la usa en sismos moderados. Por eso el modelo aprendió a asociar *mb* con «sin marca». Las siglas que no aparecen
en el selector (como *mwp* o *mwc*) tienen menos de 20 sismos en el entrenamiento y no se ofrecen.
""")

    st.divider()
    st.caption(f"Datos: USGS (earthquake.usgs.gov), sismos del {resumen['datos']['desde']} al "
               f"{resumen['datos']['hasta']}. Modelo: gradient boosting entrenado con datos hasta el 31/03/2026, "
               f"semilla {resumen['semilla']}.")


seccion_proba()
