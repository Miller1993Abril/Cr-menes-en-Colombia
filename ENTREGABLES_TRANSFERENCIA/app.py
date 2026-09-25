"""Dashboard generico que consume resultados de modelos ya entrenados."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from data_utils import cargar_y_validar


DATOS = cargar_y_validar()
PRED = DATOS.predicciones
METRICAS = DATOS.metricas
IMPORTANCIA = DATOS.importancia

COLORES = {
    "primario": config.COLOR_PRIMARIO,
    "secundario": config.COLOR_SECUNDARIO,
    "acento": config.COLOR_ACENTO,
    "alerta": config.COLOR_ALERTA,
}

app = Dash(__name__, title=config.TITULO)
server = app.server


def tarjeta(titulo: str, identificador: str, nota: str):
    return html.Div(
        [html.P(titulo, className="kpi-title"), html.H2(id=identificador), html.P(nota, className="kpi-note")],
        className="kpi-card",
    )


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div([html.P("PLANTILLA PARA RESULTADOS MODELADOS", className="eyebrow"), html.H1(config.TITULO), html.P(config.SUBTITULO, className="subtitle")]),
                html.Div("DASH + PLOTLY", className="badge"),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Aside(
                    [
                        html.H3("Filtros"),
                        html.Label(config.NOMBRE_GRUPO),
                        dcc.Dropdown(sorted(PRED["grupo"].unique()), multi=True, id="grupo", placeholder="Todos"),
                        html.Label(config.NOMBRE_PERIODO),
                        dcc.Dropdown(sorted(PRED["periodo"].unique()), multi=True, id="periodo", placeholder="Todos"),
                        html.Label("Umbral de clasificacion"),
                        dcc.Slider(0.10, 0.90, 0.05, value=0.50, marks={x: f"{x:.1f}" for x in [0.1, 0.3, 0.5, 0.7, 0.9]}, id="umbral"),
                        html.P("El umbral cambia las predicciones y las metricas mostradas, pero no reentrena el modelo.", className="help"),
                        html.Button("Restablecer", id="restablecer", n_clicks=0),
                    ],
                    className="sidebar",
                ),
                html.Main(
                    [
                        html.Div(
                            [
                                tarjeta(config.NOMBRE_REGISTRO, "kpi-n", "filtrados"),
                                tarjeta(config.NOMBRE_CLASE_POSITIVA, "kpi-real", "proporcion observada"),
                                tarjeta("Prediccion positiva", "kpi-pred", "segun el umbral"),
                                tarjeta("ROC-AUC", "kpi-auc", "en los registros filtrados"),
                            ],
                            className="kpi-grid",
                        ),
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="Resumen",
                                    children=[
                                        html.Div([dcc.Graph(id="grupo-chart"), dcc.Graph(id="prob-chart")], className="grid-2"),
                                        html.Div(id="metricas-strip", className="metric-strip"),
                                        html.Div([dcc.Graph(id="confusion-chart"), dcc.Graph(id="modelos-chart")], className="grid-2"),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Variables importantes",
                                    children=[dcc.Graph(id="importancia-chart")],
                                ),
                                dcc.Tab(
                                    label="Predicciones",
                                    children=[
                                        html.P("Registros ordenados desde la probabilidad mas alta."),
                                        dash_table.DataTable(
                                            id="tabla",
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px", "textAlign": "left"},
                                            style_header={"backgroundColor": COLORES["primario"], "color": "white", "fontWeight": "bold"},
                                            style_data_conditional=[{"if": {"filter_query": "{nivel_riesgo} = Alto"}, "backgroundColor": "#FCE8E8"}],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        html.Footer("Fuente: archivos CSV exportados por el estudiante despues de modelar sus datos."),
                    ],
                    className="content",
                ),
            ],
            className="shell",
        ),
    ],
    className="app",
)


@app.callback(
    Output("grupo", "value"), Output("periodo", "value"), Output("umbral", "value"),
    Input("restablecer", "n_clicks"), prevent_initial_call=True,
)
def restablecer_filtros(_):
    return [], [], 0.50


@app.callback(
    Output("kpi-n", "children"), Output("kpi-real", "children"),
    Output("kpi-pred", "children"), Output("kpi-auc", "children"),
    Output("grupo-chart", "figure"), Output("prob-chart", "figure"),
    Output("metricas-strip", "children"), Output("confusion-chart", "figure"),
    Output("modelos-chart", "figure"), Output("importancia-chart", "figure"),
    Output("tabla", "data"), Output("tabla", "columns"),
    Input("grupo", "value"), Input("periodo", "value"), Input("umbral", "value"),
)
def actualizar(grupos, periodos, umbral):
    d = PRED.copy()
    if grupos:
        d = d[d["grupo"].isin(grupos)]
    if periodos:
        d = d[d["periodo"].isin(periodos)]

    if d.empty:
        vacia = go.Figure().update_layout(template="plotly_white", title="No hay datos con estos filtros")
        return "0", "--", "--", "--", vacia, vacia, [], vacia, vacia, vacia, [], []

    y = d["valor_real"].astype(int).to_numpy()
    prob = d["probabilidad"].to_numpy()
    pred = (prob >= umbral).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    exactitud = accuracy_score(y, pred)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    resumen_grupo = d.groupby("grupo", as_index=False).agg(tasa_real=("valor_real", "mean"), registros=("id", "count")).sort_values("tasa_real")
    fig_grupo = px.bar(resumen_grupo, x="tasa_real", y="grupo", orientation="h", text=resumen_grupo["tasa_real"].map(lambda x: f"{x:.1%}"), title=f"{config.NOMBRE_CLASE_POSITIVA} por {config.NOMBRE_GRUPO.lower()}", color="tasa_real", color_continuous_scale=["#DCEFED", COLORES["acento"]])
    fig_grupo.update_layout(template="plotly_white", coloraxis_showscale=False, xaxis_tickformat=".0%", xaxis_title="Proporcion", yaxis_title="")

    grafica = d.assign(clase=d["valor_real"].map({0: "Clase 0", 1: "Clase 1"}))
    fig_prob = px.histogram(grafica, x="probabilidad", color="clase", nbins=16, barmode="overlay", opacity=0.65, title="Distribucion de probabilidades", color_discrete_map={"Clase 0": COLORES["secundario"], "Clase 1": COLORES["alerta"]})
    fig_prob.add_vline(x=umbral, line_dash="dash", line_color="#F4B942")
    fig_prob.update_layout(template="plotly_white", xaxis_tickformat=".0%")

    valores = [("Exactitud", exactitud), ("Precision", precision), ("Recall", recall), ("F1", f1)]
    strip = [html.Div([html.Span(nombre), html.Strong(f"{valor:.1%}")]) for nombre, valor in valores]

    matriz = confusion_matrix(y, pred, labels=[0, 1])
    fig_cm = go.Figure(go.Heatmap(z=matriz, x=["Predice 0", "Predice 1"], y=["Real 0", "Real 1"], text=matriz, texttemplate="%{text}", colorscale=[[0, "#EAF1F7"], [1, COLORES["secundario"]]], showscale=False))
    fig_cm.update_layout(template="plotly_white", title=f"Matriz de confusion (umbral {umbral:.0%})")

    metricas_largas = METRICAS.melt(id_vars="modelo", value_vars=["exactitud", "precision", "recall", "f1", "roc_auc"], var_name="metrica", value_name="valor")
    fig_modelos = px.bar(metricas_largas, x="modelo", y="valor", color="metrica", barmode="group", title="Comparacion de modelos")
    fig_modelos.update_layout(template="plotly_white", yaxis_tickformat=".0%", xaxis_title="")

    importancia = IMPORTANCIA.sort_values("importancia").tail(15)
    fig_importancia = px.bar(importancia, x="importancia", y="variable", orientation="h", title="Importancia de variables", color_discrete_sequence=[COLORES["acento"]])
    fig_importancia.update_layout(template="plotly_white", xaxis_title="Importancia", yaxis_title="")

    tabla = d.copy()
    tabla["prediccion_umbral"] = pred
    tabla["nivel_riesgo"] = pd.cut(tabla["probabilidad"], bins=[-0.001, 0.33, 0.66, 1.0], labels=["Bajo", "Medio", "Alto"])
    tabla = tabla.sort_values("probabilidad", ascending=False).head(200)
    tabla["probabilidad"] = tabla["probabilidad"].round(3)
    columnas = [{"name": col.replace("_", " ").title(), "id": col} for col in tabla.columns]

    return (f"{len(d):,}", f"{d['valor_real'].mean():.1%}", f"{pred.mean():.1%}", f"{auc:.1%}" if not np.isnan(auc) else "--", fig_grupo, fig_prob, strip, fig_cm, fig_modelos, fig_importancia, tabla.to_dict("records"), columnas)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)

