"""Construye los datos de la aplicación a partir del CSV que produce el pipeline.

Uso:
    python build_app.py                                  # usa la copia congelada datos/DatasetTsunami.csv
    python build_app.py --datos ruta/a/sismos_master.csv # usa una salida nueva del pipeline
    python build_app.py --rapido                         # sin repetir los experimentos (sólo el modelo)

Hace lo mismo que el notebook de la Entrega 3, con la misma semilla. La métrica es el F1 de la clase con alerta
(tsunami = 1):
  1. lee los sismos y corta en CORTE (las marcas desde abril de 2026 todavía no están cargadas);
  2. separa entrenamiento y test por grupo (un pliegue de StratifiedGroupKFold, 20 %);
  3. entrena el modelo elegido con los hiperparámetros ajustados;
  4. calcula la probabilidad de cada sismo con un modelo que no lo vio:
     - entrenamiento: validación cruzada (cross_val_predict), igual que la sección 8.2;
     - test y sismos sin marca cargada: el modelo entrenado;
  5. repite los experimentos de la Entrega 3 que muestra la app (comparación de modelos, formas de partir,
     ablación de columnas, umbral, importancia por permutación y el árbol chico);
  6. guarda en app_data/ lo que lee app.py: sismos_app.csv, resumen.json y modelo.joblib.

Si los datos cambian, correrlo de nuevo actualiza la aplicación. Tarda un par de minutos por los experimentos.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from modelo import (SEMILLA, CORTE, UMBRAL, CANDIDATAS, NUMERICAS, CATEGORICAS, HIPERPARAMETROS,
                    armar, modelo_elegido, grupos_de, validacion)

# Columnas del CSV del pipeline que viajan a la app (las 18 originales, más las que agrega este script)
COLUMNAS_ORIGINALES = ["id", "time", "mag", "magType", "place", "latitude", "longitude", "depth", "tsunami",
                       "type", "status", "sig", "gap", "dmin", "nst", "rms", "net", "dist_costa_km"]


def resultado(real, alerta):
    """Cómo le fue al modelo con cada sismo, en palabras."""
    return np.select([(real == 1) & (alerta == 1), (real == 1) & (alerta == 0), (real == 0) & (alerta == 1)],
                     ["acierto: con marca", "falso negativo", "falsa alarma"], default="acierto: sin marca")


def metricas(y, proba, umbral):
    pred = (proba >= umbral).astype(int)
    return {"f1": f1_score(y, pred),          # F1 de la clase con alerta (tsunami = 1)
            "recall_tsunami": pred[y == 1].mean(),
            "precision_tsunami": y[pred == 1].mean() if pred.sum() else 0.0,
            "pr_auc": average_precision_score(y, proba),
            "alertas_cada_100": pred.mean() * 100}


def experimentos(datos, X_train, y_train, g_train, cv, modelo, X_test, y_test, proba_cv):
    """Los experimentos de la Entrega 3, con el mismo código del notebook (secciones 3.5 a 10.3)."""

    def medir(pipeline, X_=None, y_=None, validacion_=cv, con_grupos=True):
        X_ = X_train if X_ is None else X_
        y_ = y_train if y_ is None else y_
        r = cross_validate(pipeline, X_, y_, groups=g_train if con_grupos else None, cv=validacion_,
                           scoring=["f1", "average_precision"], n_jobs=-1)
        return {"f1": r["test_f1"].mean(), "desvio": r["test_f1"].std(),
                "pr_auc": r["test_average_precision"].mean()}

    def boosting():
        return HistGradientBoostingClassifier(random_state=SEMILLA)

    print("  · comparación de modelos")
    comparacion = {
        "Piso: siempre «sin marca»": medir(armar(DummyClassifier(strategy="most_frequent"))),
        "Piso: al azar": medir(armar(DummyClassifier(strategy="stratified", random_state=SEMILLA))),
        "Regresión logística": medir(armar(LogisticRegression(max_iter=3000))),
        "Árbol de decisión": medir(armar(DecisionTreeClassifier(random_state=SEMILLA))),
        "Random forest": medir(armar(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMILLA))),
        "Gradient boosting": medir(armar(boosting())),
        "Gradient boosting ajustado (elegido)": medir(modelo_elegido()),
    }

    print("  · formas de partir")
    al_azar = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    orden = datos.loc[X_train.index, "time"].sort_values().index
    particiones = []
    for nombre, clf in {"Regresión logística": LogisticRegression(max_iter=3000),
                        "Gradient boosting": boosting()}.items():
        for forma, r in [("Al azar", medir(armar(clf), validacion_=al_azar, con_grupos=False)),
                         ("Por grupo", medir(armar(clf))),
                         ("Por fecha", medir(armar(clf), X_train.loc[orden], y_train.loc[orden],
                                             validacion_=TimeSeriesSplit(n_splits=5), con_grupos=False))]:
            particiones.append({"modelo": nombre, "forma": forma, **r})

    print("  · ablación de columnas")
    N = NUMERICAS
    variantes = {
        "base": armar(boosting()),
        "+ nst": armar(boosting(), numericas=N + ["nst"]),
        "+ dmin": armar(boosting(), numericas=N + ["dmin"]),
        "+ gap": armar(boosting(), numericas=N + ["gap"]),
        "+ rms": armar(boosting(), numericas=N + ["rms"]),
        "+ nst, dmin y gap": armar(boosting(), numericas=N + ["nst", "dmin", "gap"]),
        "distancia continua en vez de mar/tierra": armar(boosting(), numericas=N + ["dist_costa_km"], binarias=[]),
        "− mar/tierra": armar(boosting(), binarias=[]),
        "− tipo de magnitud": armar(boosting(), categoricas=[]),
        "− profundidad": armar(boosting(), numericas=["mag", "latitude", "longitude"]),
        "− latitud y longitud": armar(boosting(), numericas=["mag", "depth"]),
    }
    ablacion = [{"variante": n, **medir(p)} for n, p in variantes.items()]
    X_con_sig = X_train.join(datos["sig"])
    ablacion.append({"variante": "+ sig (fuga)", **medir(armar(boosting(), numericas=N + ["sig"]), X_con_sig)})
    logistica_sin_ubicacion = medir(armar(LogisticRegression(max_iter=3000), numericas=["mag", "depth"]))

    print("  · umbral, importancia y árbol chico")
    curva = []
    for u in np.round(np.arange(0.05, 0.951, 0.01), 2):
        m = metricas(y_train.to_numpy(), proba_cv, u)
        curva.append({"umbral": float(u), "F1": m["f1"], "recall": m["recall_tsunami"],
                      "precisión": m["precision_tsunami"]})

    columnas_modelo = NUMERICAS + CATEGORICAS + ["dist_costa_km"]
    imp = permutation_importance(modelo, X_test[columnas_modelo], y_test, scoring="f1",
                                 n_repeats=10, random_state=SEMILLA, n_jobs=-1)
    importancia = [{"columna": c, "caida": float(m), "desvio": float(s)}
                   for c, m, s in zip(columnas_modelo, imp.importances_mean, imp.importances_std)]

    arbol = armar(DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEMILLA),
                  escalar=False)
    f1_arbol = medir(clone(arbol))["f1"]
    arbol.fit(X_train, y_train)
    nombres = [n.split("__")[-1] for n in arbol[:-1].get_feature_names_out()]
    t = arbol[-1].tree_
    cortes = {}
    for nodo in range(t.node_count):
        if t.feature[nodo] >= 0:
            cortes.setdefault(nombres[t.feature[nodo]], float(t.threshold[nodo]))   # el primer corte de cada columna

    return {"comparacion": [{"modelo": k, **v} for k, v in comparacion.items()],
            "particiones": particiones, "ablacion": ablacion,
            "logistica_sin_ubicacion": logistica_sin_ubicacion,
            "curva_umbral": curva, "importancia": importancia,
            "arbol": {"f1": f1_arbol, "cortes": cortes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default="datos/DatasetTsunami.csv", help="CSV con los sismos (salida del pipeline)")
    parser.add_argument("--salida", default="app_data", help="carpeta donde se guardan los archivos de la app")
    parser.add_argument("--rapido", action="store_true", help="no repetir los experimentos (usa los del resumen anterior)")
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(exist_ok=True)

    # 1. Los datos y el corte
    df = pd.read_csv(args.datos)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    datos = df[df["time"] < CORTE].reset_index(drop=True)
    sin_marca = df[df["time"] >= CORTE].reset_index(drop=True)

    # 2. La partición, por grupo
    X, y = datos[CANDIDATAS], datos["tsunami"]
    grupos = grupos_de(datos)
    cv = validacion()
    idx_train, idx_test = next(cv.split(X, y, grupos))
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    g_train = grupos.iloc[idx_train]

    # 3. El modelo elegido, entrenado con todo el entrenamiento
    print("Entrenando el modelo elegido…")
    modelo = modelo_elegido().fit(X_train, y_train)

    # 4. Probabilidades de modelos que no vieron cada sismo
    proba_cv = cross_val_predict(clone(modelo_elegido()), X_train, y_train, groups=g_train, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    proba_test = modelo.predict_proba(X_test)[:, 1]
    proba_sin_marca = modelo.predict_proba(sin_marca[CANDIDATAS])[:, 1] if len(sin_marca) else np.array([])

    # 5. Los experimentos de la Entrega 3
    resumen_previo = salida / "resumen.json"
    if args.rapido and resumen_previo.exists():
        print("Experimentos: se reutilizan los del resumen anterior (--rapido)")
        with open(resumen_previo, encoding="utf-8") as f:
            exp = json.load(f)["experimentos"]
    else:
        print("Repitiendo los experimentos de la Entrega 3 (tarda un par de minutos)…")
        exp = experimentos(datos, X_train, y_train, g_train, cv, modelo, X_test, y_test, proba_cv)

    # 6a. Una tabla con todos los sismos y lo que dijo el modelo de cada uno.
    # Para el gráfico de la validación por grupos se guarda también el grupo de cada sismo y en qué pedazo (1 a 5)
    # cae cada sismo de entrenamiento si se parte al azar o por grupo.
    pedazo_azar = np.zeros(len(X_train), dtype=int)
    for i, (_, val) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA).split(X_train, y_train)):
        pedazo_azar[val] = i + 1
    pedazo_grupo = np.zeros(len(X_train), dtype=int)
    for i, (_, val) in enumerate(cv.split(X_train, y_train, g_train)):
        pedazo_grupo[val] = i + 1
    partes = []
    for tabla, proba, conjunto in [(datos.iloc[idx_train], proba_cv, "entrenamiento"),
                                   (datos.iloc[idx_test], proba_test, "test"),
                                   (sin_marca, proba_sin_marca, "sin marca cargada")]:
        t = tabla[COLUMNAS_ORIGINALES].copy()
        t["conjunto"] = conjunto
        t["grupo"] = grupos.iloc[idx_train if conjunto == "entrenamiento" else idx_test].to_numpy() \
            if conjunto != "sin marca cargada" else ""
        t["pedazo_azar"] = pedazo_azar if conjunto == "entrenamiento" else 0
        t["pedazo_grupo"] = pedazo_grupo if conjunto == "entrenamiento" else 0
        t["probabilidad"] = proba.round(4)
        t["alerta"] = (proba >= UMBRAL).astype(int)
        t["resultado"] = (resultado(t["tsunami"].to_numpy(), t["alerta"].to_numpy())
                          if conjunto != "sin marca cargada" else "sin marca cargada")
        partes.append(t)
    app = pd.concat(partes).sort_values("time").reset_index(drop=True)
    app["region"] = app["place"].str.split(", ").str[-1]
    app["en_el_mar"] = np.where(app["dist_costa_km"] > 0, "mar", "tierra")
    app["anio"] = app["time"].dt.year
    app["mes"] = app["time"].dt.strftime("%Y-%m")
    app["time"] = app["time"].dt.strftime("%Y-%m-%d %H:%M")
    app.to_csv(salida / "sismos_app.csv", index=False)

    # 6b. Los números que muestra la app
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()
    conteo_magtype = X_train["magType"].value_counts()
    tn, fp, fn, tp = confusion_matrix(y_te, (proba_test >= UMBRAL).astype(int)).ravel()
    resumen = {
        "umbral": UMBRAL, "semilla": SEMILLA, "corte": CORTE, "hiperparametros": HIPERPARAMETROS,
        "version_sklearn": sklearn.__version__,
        "datos": {"archivo": Path(args.datos).name,
                  "desde": str(df["time"].min().date()), "hasta": str(df["time"].max().date()),
                  "columnas": int(df.shape[1]), "filas_duplicadas": int(df.duplicated().sum()),
                  "ids_duplicados": int(df["id"].duplicated().sum())},
        "filtros": [
            {"paso": "Dataset del pipeline", "filas": len(df), "con_marca": int(df["tsunami"].sum())},
            {"paso": f"Corte antes de {CORTE}", "filas": len(datos), "con_marca": int(y.sum())},
            {"paso": "Entrenamiento (80 %)", "filas": len(X_train), "con_marca": int(y_tr.sum())},
            {"paso": "Test (20 %)", "filas": len(X_test), "con_marca": int(y_te.sum())},
        ],
        "grupos": {"cantidad": int(grupos.nunique()), "mediana_por_grupo": float(grupos.value_counts().median()),
                   "mayor": grupos.value_counts().index[0], "mayor_sismos": int(grupos.value_counts().iloc[0])},
        "validacion_cruzada": metricas(y_tr, proba_cv, UMBRAL),
        "validacion_cruzada_umbral_05": metricas(y_tr, proba_cv, 0.5),
        "test": {**metricas(y_te, proba_test, UMBRAL),
                 "verdaderos_negativos": int(tn), "falsas_alarmas": int(fp),
                 "falsos_negativos": int(fn), "aciertos_con_marca": int(tp)},
        "piso_test": {"f1": f1_score(y_te, np.zeros_like(y_te), zero_division=0),
                      "pr_auc": float(y_te.mean())},
        # Rangos de lo que vio el modelo: "Probá el modelo" no deja cargar valores fuera de esto
        "rangos": {c: [float(X_train[c].min()), float(X_train[c].max())]
                   for c in ["mag", "depth", "latitude", "longitude"]},
        "magtypes": [m for m in conteo_magtype.index if conteo_magtype[m] >= 20],
        "experimentos": exp,
    }
    with open(salida / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=float)

    # 6c. El modelo, para "Probá el modelo"
    joblib.dump({"pipeline": modelo, "umbral": UMBRAL, "columnas": CANDIDATAS, "semilla": SEMILLA,
                 "datos_hasta": CORTE, "version_sklearn": sklearn.__version__}, salida / "modelo.joblib")

    vc, te = resumen["validacion_cruzada"], resumen["test"]
    print(f"Sismos: {len(df)} | modelados: {len(datos)} | sin marca cargada: {len(sin_marca)}")
    print(f"Validación cruzada (umbral {UMBRAL}): F1 {vc['f1']:.3f} · recall {vc['recall_tsunami']:.3f}")
    print(f"Test (umbral {UMBRAL}): F1 {te['f1']:.3f} · recall {te['recall_tsunami']:.3f} · "
          f"precision {te['precision_tsunami']:.3f} · FN {fn} · FP {fp}")
    print(f"Archivos guardados en {salida}/")


if __name__ == "__main__":
    main()
