"""
Dashboard: Crimenes en Colombia
- Tabla resumen estadístico por departamento
- Departamentos con promedio superior al nacional
- Gráfica de distribución normal de promedios
- Matriz de correlación ampliada + interpretación
Autor: Miller Abril
Versión final limpia
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import norm
import dash
from dash import dcc, html, Input, Output, dash_table, callback

# ==============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================
print("📂 Cargando datos...")
df = pd.read_csv("datos_col.csv", encoding="utf-8")
df.columns = df.columns.str.strip()

df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
df["anio"] = df["fecha"].dt.year

MEDIA_NACIONAL = df["cantidad"].mean()
DESVIO_NACIONAL = df["cantidad"].std()
UMBRAL_RIESGO = MEDIA_NACIONAL + DESVIO_NACIONAL
prom_nacional = df['cantidad'].mean().round(3)

print(f"✅ Datos: {df.shape[0]} filas | Media nacional: {prom_nacional}")

# ==============================================
# RESUMEN ESTADÍSTICO POR DEPARTAMENTO
# ==============================================
est_territorio = df.groupby('departamento')['cantidad'].agg([
    ('conteo_registros', 'count'),
    ('total_victimas', 'sum'),
    ('promedio_victimas_por_hecho', 'mean'),
    ('desviacion_estandar', 'std'),
    ('minimo', 'min'),
    ('mediana', 'median'),
    ('maximo', 'max')
]).reset_index()

est_territorio['promedio_victimas_por_hecho'] = est_territorio['promedio_victimas_por_hecho'].round(3)
est_territorio['desviacion_estandar'] = est_territorio['desviacion_estandar'].round(3)
est_territorio = est_territorio.sort_values('total_victimas', ascending=False)

superiores = est_territorio[est_territorio['promedio_victimas_por_hecho'] > prom_nacional]
superiores = superiores[['departamento', 'promedio_victimas_por_hecho']].sort_values('promedio_victimas_por_hecho', ascending=False)

# Gráfica Distribución Normal
promedios = df.groupby('departamento')['cantidad'].mean()
media_p = promedios.mean()
desv_p = promedios.std()

x = np.linspace(promedios.min(), promedios.max(), 200)
curva_normal = norm.pdf(x, media_p, desv_p)

fig_distribucion = go.Figure()
fig_distribucion.add_trace(go.Histogram(
    x=promedios.values,
    histnorm='probability density',
    name='Promedios por departamento',
    marker_color='#3366CC',
    opacity=0.5,
    nbinsx=15
))
fig_distribucion.add_trace(go.Scatter(
    x=x, y=curva_normal, mode='lines',
    name=f'Curva Normal (μ={media_p:.3f} | σ={desv_p:.3f})',
    line=dict(color='red', dash='dash', width=2.5)
))
fig_distribucion.add_vline(
    x=media_p, line_color='green', line_width=2,
    annotation_text=f'Media: {media_p:.3f}', annotation_position='top right'
)
fig_distribucion.update_layout(
    title="Distribución de promedios por departamento vs. Curva Normal",
    xaxis_title="Víctimas por hecho (promedio departamento)",
    yaxis_title="Densidad", bargap=0.2, height=500, template='plotly_white'
)

# ==============================================
# MATRIZ DE CORRELACIÓN AMPLIADA CON PROMEDIOS
# ==============================================
if 'anio' not in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    df['anio'] = df['fecha'].dt.year

prom_dpto = df.groupby('departamento')['cantidad'].mean().reset_index()
prom_dpto.columns = ['departamento', 'prom_victimas']
df_ampliado = df.merge(prom_dpto, on='departamento', how='left')

vars_num = ['codigo', 'cantidad', 'anio', 'prom_victimas']
matriz_corr = df_ampliado[vars_num].corr()
etiquetas = ['Código', 'Víctimas', 'Año', 'Prom. Dpto']

fig_correlacion = px.imshow(
    matriz_corr,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1, aspect='auto'
)
fig_correlacion.update_layout(
    title="Matriz de Correlación: Variables Numéricas",
    height=600,
    xaxis=dict(
        tickmode='array', tickvals=list(range(len(etiquetas))), ticktext=etiquetas
    ),
    yaxis=dict(
        tickmode='array', tickvals=list(range(len(etiquetas))), ticktext=etiquetas
    )
)

# ==============================================
# LISTAS Y MODELO PREDICTIVO
# ==============================================
lista_dptos = sorted(df["departamento"].dropna().unique().tolist())
lista_armas = sorted(df["arma"].dropna().unique().tolist())

X = df[["promedio_dpto", "genero_bin"]].dropna()
y = df.loc[X.index, "es_alto_riesgo"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]
metricas = {
    "Exactitud": accuracy_score(y_test, y_pred),
    "Precisión": precision_score(y_test, y_pred, zero_division=0),
    "Sensibilidad": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "AUC ROC": roc_auc_score(y_test, y_proba)
}

# ==============================================
# ESTILOS Y MENSAJES
# ==============================================
app = dash.Dash(__name__)
server = app.server
COLOR_PRINCIPAL = "#12304A"
COLOR_DESTACADO = "#1976D2"
COLOR_ACENTO = "#FF9800"
COLOR_HOMBRES = "#1976D2"
COLOR_MUJERES = "#FF9800"
COLOR_NO_REPORTA = "#90A4AE"

MENSAJE_PROPOSITO = """
Aplicar métodos estadísticos y modelos predictivos sobre el conjunto de datos Crímenes de Colombia,
dando continuidad al análisis exploratorio mediante Python y Dash.
El propósito es formular y contrastar hipótesis, así como construir modelos de regresión
que permitan identificar patrones y relaciones significativas entre las variables estudiadas.
"""

TEXTO_INTERPRETACION_CORRELACION = """
Se calculó la matriz de correlación de Pearson para evaluar la relación lineal entre las variables cuantitativas del conjunto de datos: código de registro, cantidad de víctimas y año.

El **código** es un identificador secuencial sin relación con la magnitud del hecho → su correlación con la cantidad de víctimas es prácticamente cero, como era esperado.

El **año** presenta una correlación negativa muy baja (-0.13): sugiere una tendencia leve a la reducción de víctimas por hecho con el paso del tiempo, pero el valor es tan cercano a cero que no se considera una relación significativa.

En conjunto, ninguna variable numérica disponible explica la variación en la cantidad de víctimas. Esto indica que las diferencias observadas dependen de factores categóricos: el departamento, el tipo de arma o el grupo de edad, tal como se evidenció en el análisis por territorios.

**Consecuencia para el modelado:** Dado que no existe correlación lineal entre las variables numéricas, una regresión lineal simple con estos predictores tendría muy bajo poder explicativo. Se recomienda incorporar las variables categóricas (mediante codificación) para identificar los factores que realmente inciden en la mayor o menor afectación.
"""

# ==============================================
# DISEÑO DEL DASHBOARD
# ==============================================
app.layout = html.Div([
    # CABECERA
    html.Div([
        html.H1("Crímenes en Colombia", style={"color": "white"}),
        html.P("Análisis exploratorio, estadístico y modelos predictivos",
               style={"color": "#B0BEC5", "fontSize": "18px"}),
        html.Div(MENSAJE_PROPOSITO, style={
            "backgroundColor": "rgba(255,255,255,0.1)", "padding": "15px",
            "borderRadius": "8px", "marginTop": "15px",
            "color": "#E0E0E0", "fontStyle": "italic"
        })
    ], style={"backgroundColor": COLOR_PRINCIPAL, "padding": "25px", "borderRadius": "10px"}),

    # MÉTRICAS DEL MODELO
    html.Div([
        html.H3("📊 Rendimiento del Modelo Predictivo", style={"textAlign": "center"}),
        html.Div([
            html.Div([
                html.H4(nombre),
                html.P(f"{valor:.2%}", style={
                    "fontSize": "24px", "fontWeight": "bold", "color": COLOR_DESTACADO
                })
            ], style={
                "width": "19%", "display": "inline-block", "textAlign": "center",
                "padding": "15px", "border": "1px solid #eee", "borderRadius": "8px"
            })
            for nombre, valor in metricas.items()
        ])
    ], style={"margin": "25px 0"}),

    # CONTRASTE DE ARMAS
    html.Div([
        html.H3("🔫 Contraste de Tipos de Arma Utilizados", style={"marginTop": "30px"}),
        html.Div([
            html.Div([
                html.Label("Seleccionar Departamento(s):"),
                dcc.Dropdown(
                    id="filtro_armas_dpto_multi",
                    options=[{"label": "Todos", "value": "todos"}] + [
                        {"label": d, "value": d} for d in lista_dptos
                    ], value=["todos"], multi=True, style={"width": "100%"}
                )
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div([
                html.Label("Seleccionar Tipo(s) de Arma:"),
                dcc.Dropdown(
                    id="filtro_armas_tipo_multi",
                    options=[{"label": "Todos", "value": "todos"}] + [
                        {"label": a, "value": a} for a in lista_armas
                    ], value=["todos"], multi=True, style={"width": "100%"}
                )
            ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"})
        ], style={"marginBottom": "15px"}),
        dcc.Graph(id="grafico_contraste_armas")
    ], style={"padding": "0 20px"}),

    # GÉNERO POR DEPARTAMENTO
    html.Div([
        html.H3("👨‍👩 Casos por Género — Comparación entre Departamentos", style={"marginTop": "30px"}),
        html.Div([
            html.Label("Seleccionar Departamento(s):"),
            dcc.Dropdown(
                id="filtro_genero_dpto_multi",
                options=[{"label": "Top 10 Nacional", "value": "top10"}] + [
                    {"label": d, "value": d} for d in lista_dptos
                ], value=["top10"], multi=True, style={"width": "70%"}
            )
        ], style={"marginBottom": "15px"}),
        dcc.Graph(id="grafico_genero_dpto_multi")
    ], style={"padding": "0 20px"}),

    # LIENZO DE 4 GRÁFICAS
    html.Div([
        html.H3("📈 Análisis Integral — Comportamiento y Riesgo", style={"textAlign": "center", "marginTop": "40px"}),
        html.Div([
            html.Label("Seleccionar Departamento(s):"),
            dcc.Dropdown(
                id="filtro_lienzo_dpto_multi",
                options=[{"label": "Nacional (Top 10)", "value": "nacional"}] + [
                    {"label": d, "value": d} for d in lista_dptos
                ], value=["nacional"], multi=True, style={"width": "70%"}
            )
        ], style={"marginBottom": "20px", "padding": "0 20px"}),
        dcc.Graph(id="lienzo_4_graficas_multi")
    ]),

    # VALORES ESTADÍSTICOS DE CANTIDAD DE VÍCTIMAS
    html.Div([
        html.H2("📊 Valores Estadísticos de Cantidad de Víctimas",
                style={"marginTop": "40px", "color": COLOR_PRINCIPAL}),
        html.Hr(),

        html.H4("Resumen por Departamento"),
        dash_table.DataTable(
            data=est_territorio.to_dict("records"),
            columns=[
                {"name": "Departamento", "id": "departamento"},
                {"name": "Hechos Registrados", "id": "conteo_registros"},
                {"name": "Total Víctimas", "id": "total_victimas"},
                {"name": "Promedio/Hecho", "id": "promedio_victimas_por_hecho"},
                {"name": "Desviación Estándar", "id": "desviacion_estandar"},
                {"name": "Mínimo", "id": "minimo"},
                {"name": "Mediana", "id": "mediana"},
                {"name": "Máximo", "id": "maximo"}
            ],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": COLOR_PRINCIPAL, "color": "white", "fontWeight": "bold"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
            page_size=15, sort_action="native"
        ),

        html.Div([
            html.H5("Interpretación:"),
            html.Ul([
                html.Li([html.Strong("conteo_registros: "), "cuántos hechos delictivos se registraron ahí"]),
                html.Li([html.Strong("total_victimas: "), "suma de personas afectadas"]),
                html.Li([html.Strong("promedio_victimas_por_hecho: "), "en promedio, cuántas personas resultaron afectadas por cada hecho"]),
                html.Li([html.Strong("desviacion_estandar: "), "variabilidad en el número de víctimas por evento"]),
                html.Li([html.Strong("máximo: "), "el mayor número de víctimas registrado en un solo evento"])
            ], style={"fontSize": "14px", "lineHeight": "1.8"}),
            html.P([
                html.Strong(f"A nivel país: "),
                f"promedio = {prom_nacional} víctimas por hecho"
            ], style={"marginTop": "10px", "fontSize": "15px"})
        ], style={"margin": "20px 0", "padding": "15px", "backgroundColor": "#f5f5f5", "borderRadius": "8px"}),

        html.H4(f"Departamentos con promedio SUPERIOR al nacional ({prom_nacional} víctimas/hecho)"),
        html.P(f"Total: {len(superiores)} departamentos"),
        dash_table.DataTable(
            data=superiores.to_dict("records"),
            columns=[
                {"name": "Departamento", "id": "departamento"},
                {"name": "Promedio de Víctimas por Hecho", "id": "promedio_victimas_por_hecho"}
            ],
            style_table={"overflowX": "auto", "width": "60%"},
            style_header={"backgroundColor": "#B71C1C", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#FFEBEE"},
                {"if": {"column_id": "promedio_victimas_por_hecho"},
                 "color": "#B71C1C", "fontWeight": "bold"}
            ],
            page_size=15, sort_action="native"
        ),

        html.H4("Distribución de promedios por departamento vs. Curva Normal", style={"marginTop": "30px"}),
        dcc.Graph(figure=fig_distribucion),

        html.Div([
            html.H5("Interpretación:"),
            html.P("Las barras azules muestran cuántos departamentos presentan cada valor de promedio de víctimas por hecho. La mayor concentración se encuentra entre 1.05 y 1.15, lo que indica que la mayoría de los territorios tienen comportamientos similares."),
            html.P("La curva discontinua roja representa la distribución normal teórica para comparar el patrón observado:"),
            html.Ul([
                html.Li("El pico se desplaza hacia la izquierda → sesgo positivo: hay pocos departamentos con valores altos que elevan el promedio general."),
                html.Li("La cola derecha se extiende hacia valores mayores → corresponden a los departamentos con mayor afectación por hecho, identificados en la tabla anterior.")
            ])
        ], style={"margin": "20px 0", "padding": "15px", "backgroundColor": "#f0f4f8", "borderRadius": "8px"}),

        # MATRIZ DE CORRELACIÓN + INTERPRETACIÓN
        html.H4("Matriz de Correlación: Variables Numéricas", style={"marginTop": "40px"}),
        dcc.Graph(figure=fig_correlacion),
        html.Div([
            html.H5("Interpretación:"),
            html.P(TEXTO_INTERPRETACION_CORRELACION, style={"lineHeight": "1.8", "textAlign": "justify"})
        ], style={"margin": "20px 0", "padding": "20px", "backgroundColor": "#fefefe", "border": "1px solid #e0e0e0", "borderRadius": "8px"})

    ], style={"padding": "0 20px"}),

    # ANÁLISIS INTERACTIVO
    html.Div([
        html.H3("🔍 Análisis Interactivo", style={"marginTop": "40px"}),
        html.Div([
            html.Label("Seleccionar Departamento(s):"),
            dcc.Dropdown(
                id="filtro_general_multi",
                options=[{"label": "Todos los departamentos", "value": "todos"}] + [
                    {"label": d, "value": d} for d in lista_dptos
                ], value=["todos"], multi=True, style={"width": "70%", "marginBottom": "20px"}
            )
        ]),
        html.Div([
            html.Div([dcc.Graph(id="graf_deptos_multi")], style={"width": "48%", "display": "inline-block"}),
            html.Div([dcc.Graph(id="graf_gen_multi")], style={"width": "48%", "display": "inline-block"}),
        ]),
        html.Div([
            html.Div([dcc.Graph(id="graf_imp_multi")], style={"width": "48%", "display": "inline-block"}),
            html.Div([dcc.Graph(id="graf_riesgo_scatter_multi")], style={"width": "48%", "display": "inline-block"}),
        ])
    ])
], style={"fontFamily": "Arial", "maxWidth": "1300px", "margin": "0 auto", "padding": "20px"})

# ==============================================
# CALLBACKS
# ==============================================
def filtrar_por_lista(df_entrada, columna, valores_seleccionados):
    if "todos" in valores_seleccionados or len(valores_seleccionados) == 0:
        return df_entrada
    return df_entrada[df_entrada[columna].isin(valores_seleccionados)]

@callback(
    Output("grafico_contraste_armas", "figure"),
    [Input("filtro_armas_dpto_multi", "value"),
     Input("filtro_armas_tipo_multi", "value")]
)
def actualizar_contraste_armas(dptos_seleccionados, armas_seleccionadas):
    df_paso = filtrar_por_lista(df, "departamento", dptos_seleccionados)
    df_fil = filtrar_por_lista(df_paso, "arma", armas_seleccionadas)
    cantidad_armas = df_fil["arma"].value_counts().reset_index()
    cantidad_armas.columns = ["arma", "casos"]
    dptos_texto = "Todos los departamentos" if "todos" in dptos_seleccionados else ", ".join(dptos_seleccionados)
    armas_texto = "Todos los tipos" if "todos" in armas_seleccionadas else ", ".join(armas_seleccionadas)
    titulo = f"Contraste de Armas — {dptos_texto} | {armas_texto}"
    fig = px.bar(cantidad_armas, x="casos", y="arma", orientation="h",
                 color="casos", color_continuous_scale="Viridis", title=titulo)
    fig.update_layout(xaxis_title="Casos registrados", yaxis_title="Tipo de Arma", showlegend=False,
                      height=max(600, len(cantidad_armas) * 25), yaxis={"categoryorder": "total ascending"})
    return fig

@callback(Output("grafico_genero_dpto_multi", "figure"),
          Input("filtro_genero_dpto_multi", "value"))
def actualizar_genero_dpto_multi(valores):
    if "top10" in valores:
        dptos_top = df["departamento"].value_counts().nlargest(10).index.tolist()
        dptos_otros = [v for v in valores if v != "top10"]
        dptos = list(dict.fromkeys(dptos_top + dptos_otros))
        titulo = "Top 10 + Seleccionados" if dptos_otros else "Top 10 Departamentos con más registros"
    else:
        dptos = valores
        titulo = f"Departamentos: {', '.join(dptos)}"
    df_fil = df[df["departamento"].isin(dptos)]
    gen_por_dpto = df_fil.groupby(["departamento", "genero"])["cantidad"].sum().reset_index()
    fig = px.bar(gen_por_dpto, x="departamento", y="cantidad", color="genero", barmode="group",
                 color_discrete_map={"MASCULINO": COLOR_HOMBRES, "FEMENINO": COLOR_MUJERES, "NO REPORTA": COLOR_NO_REPORTA},
                 title=titulo, labels={"cantidad": "Cantidad de Casos"})
    fig.update_layout(xaxis_tickangle=-45, height=500, bargap=0.2)
    return fig

@callback(Output("lienzo_4_graficas_multi", "figure"),
          Input("filtro_lienzo_dpto_multi", "value"))
def actualizar_lienzo_multi(valores):
    if "nacional" in valores:
        dptos_top = df["departamento"].value_counts().nlargest(10).index.tolist()
        dptos_otros = [v for v in valores if v != "nacional"]
        dptos = list(dict.fromkeys(dptos_top + dptos_otros))
        titulo_extra = " — Nacional + Seleccionados" if dptos_otros else " — Top 10 Nacional"
    else:
        dptos = valores
        titulo_extra = f" — {', '.join(dptos)}"
    df_fil = df[df["departamento"].isin(dptos)]
    est = df_fil.groupby("departamento")["cantidad"].agg(["mean", "count"]).reset_index()
    est.columns = ["departamento", "promedio", "conteo"]
    est["supera_umbral"] = np.where(est["promedio"] >= UMBRAL_RIESGO, "SI", "NO")
    est = est.sort_values("promedio", ascending=False)
    evolucion = df_fil.groupby(["anio", "genero"])["cantidad"].mean().reset_index()
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Promedio de Víctimas por Hecho{titulo_extra}",
        f"¿Supera Umbral de Riesgo? ({UMBRAL_RIESGO:.2f})",
        "Evolución por Año y Género",
        "Total de Hechos por Género"
    ))
    fig.add_trace(go.Bar(x=est["departamento"], y=est["promedio"], name="Promedio", marker_color="#3366CC"), row=1, col=1)
    fig.add_hline(y=MEDIA_NACIONAL, line_dash="dash", line_color="green",
                  annotation_text=f"Prom: {MEDIA_NACIONAL:.3f}", row=1, col=1)
    fig.add_hline(y=UMBRAL_RIESGO, line_dash="solid", line_color="red",
                  annotation_text=f"Riesgo: {UMBRAL_RIESGO:.2f}", row=1, col=1)
    est_si = est[est["supera_umbral"] == "SI"]
    est_no = est[est["supera_umbral"] == "NO"]
    fig.add_trace(go.Bar(x=est_si["departamento"], y=est_si["promedio"], name="Supera Riesgo", marker_color="#CC3333"), row=1, col=2)
    fig.add_trace(go.Bar(x=est_no["departamento"], y=est_no["promedio"], name="No supera", marker_color="#3366CC"), row=1, col=2)
    for gen, color in zip(["MASCULINO", "FEMENINO", "NO REPORTA"], [COLOR_HOMBRES, COLOR_MUJERES, COLOR_NO_REPORTA]):
        sub = evolucion[evolucion["genero"] == gen]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["anio"], y=sub["cantidad"], name=gen,
                                      mode="lines+markers", line=dict(color=color, width=2.5)), row=2, col=1)
    conteo_gen = df_fil["genero"].value_counts().reset_index()
    conteo_gen.columns = ["genero", "cantidad"]
    fig.add_trace(go.Bar(x=conteo_gen["genero"], y=conteo_gen["cantidad"],
                         marker_color=[COLOR_HOMBRES, COLOR_MUJERES, COLOR_NO_REPORTA]), row=2, col=2)
    fig.update_layout(height=800, showlegend=False, title_text="Análisis Integral de Riesgo y Comportamiento")
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    return fig

@callback(
    [Output("graf_deptos_multi", "figure"), Output("graf_gen_multi", "figure"),
     Output("graf_imp_multi", "figure"), Output("graf_riesgo_scatter_multi", "figure")],
    Input("filtro_general_multi", "value")
)
def actualizar_general_multi(valores):
    df_fil = filtrar_por_lista(df, "departamento", valores)
    top = df_fil.groupby("departamento")["cantidad"].sum().reset_index().sort_values("cantidad", ascending=False).head(15)
    f1 = px.bar(top, x="departamento", y="cantidad", title="Hechos por Departamento",
                color_discrete_sequence=[COLOR_DESTACADO])
    f1.update_layout(xaxis_tickangle=-45)
    gen_agg = df_fil.groupby("genero")["cantidad"].sum().reset_index()
    f2 = px.bar(gen_agg, x="genero", y="cantidad", color="genero",
                color_discrete_map={"MASCULINO": COLOR_HOMBRES, "FEMENINO": COLOR_MUJERES, "NO REPORTA": COLOR_NO_REPORTA},
                title="Distribución por Género", labels={"cantidad": "Total de Casos"})
    f2.update_layout(showlegend=False)
    coef = modelo.coef_[0]
    f3 = px.bar(x=["Promedio Departamental", "Género Masculino"], y=np.abs(coef),
                title="Impacto en el Riesgo", color_discrete_sequence=[COLOR_DESTACADO, COLOR_ACENTO])
    f3.update_layout(showlegend=False)
    f4 = px.scatter(df_fil.dropna(), x="promedio_dpto", y="es_alto_riesgo", color="es_alto_riesgo",
                    labels={"promedio_dpto": "Promedio por Departamento", "es_alto_riesgo": "Alto Riesgo"},
                    title="Nivel de Riesgo vs Promedio", color_continuous_scale=["#CFD8DC", COLOR_DESTACADO])
    return f1, f2, f3, f4

# ==============================================
if __name__ == "__main__":
    print("Iniciando Dashboard...")
    app.run_server(debug=True)