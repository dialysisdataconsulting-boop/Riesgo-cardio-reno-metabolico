import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Sistema DIALYSIS", page_icon="🫀", layout="wide")

@st.cache_resource
def cargar_modelo():
    modelo   = joblib.load("modelo_dialysis.pkl")
    features = joblib.load("features_dialysis.pkl")
    return modelo, features

modelo, features = cargar_modelo()

RANGOS = {
    "resultado_glucosa":             [(0, 100, "🟢"), (100, 126, "🟡"), (126, 9999, "🔴")],
    "valor_hemoglobina_glucosilada": [(0, 5.7, "🟢"), (5.7, 6.5, "🟡"), (6.5, 99, "🔴")],
    "valor_trigliceridos":           [(0, 150, "🟢"), (150, 200, "🟡"), (200, 9999, "🔴")],
    "valor_colesterol_total":        [(0, 200, "🟢"), (200, 240, "🟡"), (240, 9999, "🔴")],
    "valor_colesterol_hdl":          [(40, 9999, "🟢"), (30, 40, "🟡"), (0, 30, "🔴")],
    "valor_colesterol_ldl":          [(0, 100, "🟢"), (100, 160, "🟡"), (160, 9999, "🔴")],
    "valor_acido_urico":             [(0, 6.0, "🟢"), (6.0, 7.0, "🟡"), (7.0, 99, "🔴")],
    "valor_creatina":                [(0, 1.2, "🟢"), (1.2, 1.5, "🟡"), (1.5, 99, "🔴")],
    "masa_corporal":                 [(0, 25, "🟢"), (25, 30, "🟡"), (30, 99, "🔴")],
    "valor_proteinac_reactiva":      [(0, 1.0, "🟢"), (1.0, 3.0, "🟡"), (3.0, 999, "🔴")],
    "resultado_glucosa_promedio":    [(0, 100, "🟢"), (100, 126, "🟡"), (126, 9999, "🔴")],
}

def sem(var, val):
    if var not in RANGOS:
        return ""
    for lo, hi, emoji in RANGOS[var]:
        if lo <= val < hi:
            return emoji
    return ""

def calcular_subscores(d):
    cardio = 0
    cardio += 20 if d["valor_colesterol_ldl"] >= 160 else (10 if d["valor_colesterol_ldl"] >= 130 else 0)
    cardio += 20 if d["valor_colesterol_hdl"] < 35 else (12 if d["valor_colesterol_hdl"] < 40 else (5 if d["valor_colesterol_hdl"] < 50 else 0))
    cardio += 15 if d["actividad_total"] < 150 else (8 if d["actividad_total"] < 500 else 0)
    cardio += 15 if d["masa_corporal"] >= 35 else (10 if d["masa_corporal"] >= 30 else (5 if d["masa_corporal"] >= 25 else 0))
    cardio += 10 if d["valor_trigliceridos"] >= 200 else (5 if d["valor_trigliceridos"] >= 150 else 0)
    cardio += 10 if d["valor_homocisteina"] >= 15 else (5 if d["valor_homocisteina"] >= 12 else 0)
    cardio += 10 if d["sueno_horas"] < 6 else 0
    cardio = min(cardio, 100)

    renal = 0
    renal += 35 if d["valor_creatina"] >= 1.5 else (20 if d["valor_creatina"] >= 1.2 else (5 if d["valor_creatina"] >= 1.0 else 0))
    renal += 25 if d["valor_acido_urico"] >= 7.0 else (15 if d["valor_acido_urico"] >= 6.0 else (5 if d["valor_acido_urico"] >= 5.5 else 0))
    renal += 20 if d["valor_albumina"] < 3.5 else (10 if d["valor_albumina"] < 4.0 else 0)
    renal += 15 if d["valor_proteinac_reactiva"] >= 3.0 else (8 if d["valor_proteinac_reactiva"] >= 1.0 else 0)
    renal += 8  if d["valor_homocisteina"] >= 15 else (4 if d["valor_homocisteina"] >= 12 else 0)
    renal = min(renal, 100)

    metab = 0
    metab += 30 if d["resultado_glucosa"] >= 126 else (18 if d["resultado_glucosa"] >= 100 else 0)
    metab += 25 if d["valor_hemoglobina_glucosilada"] >= 6.5 else (15 if d["valor_hemoglobina_glucosilada"] >= 5.7 else 0)
    metab += 20 if d["masa_corporal"] >= 30 else (10 if d["masa_corporal"] >= 25 else 0)
    metab += 15 if d["valor_trigliceridos"] >= 200 else (8 if d["valor_trigliceridos"] >= 150 else 0)
    metab += 10 if d["valor_insulina"] >= 25 else (5 if d["valor_insulina"] >= 15 else 0)
    metab += 8  if d["valor_colesterol_hdl"] < 40 else (4 if d["valor_colesterol_hdl"] < 50 else 0)
    metab = min(metab, 100)

    return cardio, renal, metab

def interpretar(d):
    factores = []
    if d["masa_corporal"] >= 30:        factores.append(("Obesidad (IMC ≥ 30)", 35))
    elif d["masa_corporal"] >= 25:      factores.append(("Sobrepeso (IMC 25-30)", 20))
    if d["actividad_total"] < 150:      factores.append(("Actividad física insuficiente", 30))
    elif d["actividad_total"] < 500:    factores.append(("Actividad física baja", 15))
    if d["resultado_glucosa"] >= 126:   factores.append(("Glucosa en rango diabético", 30))
    elif d["resultado_glucosa"] >= 100: factores.append(("Glucosa en rango prediabético", 18))
    if d["valor_hemoglobina_glucosilada"] >= 6.5:  factores.append(("HbA1c en rango diabético", 25))
    elif d["valor_hemoglobina_glucosilada"] >= 5.7: factores.append(("HbA1c elevada", 15))
    if d["valor_colesterol_hdl"] < 40:  factores.append(("HDL bajo", 20))
    if d["valor_acido_urico"] >= 7.0:   factores.append(("Hiperuricemia", 25))
    if d["valor_creatina"] >= 1.2:      factores.append(("Creatinina elevada", 20))
    if d["valor_trigliceridos"] >= 200: factores.append(("Hipertrigliceridemia", 15))
    if d["sueno_horas"] < 6:            factores.append(("Sueño insuficiente", 10))
    if d["valor_proteinac_reactiva"] >= 3.0: factores.append(("PCR elevada (inflamación)", 15))
    factores.sort(key=lambda x: x[1], reverse=True)
    return factores[:5]

def gauge(valor, titulo, height=240):
    color = "#E53935" if valor >= 75 else "#FFB300" if valor >= 50 else "#43A047"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50],  "color": "#E8F5E9"},
                {"range": [50, 75], "color": "#FFF8E1"},
                {"range": [75, 100],"color": "#FDEDED"},
            ],
        },
        title={"text": titulo, "font": {"size": 13}}
    ))
    fig.update_layout(height=height, margin=dict(t=50, b=0, l=10, r=10))
    return fig

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.riesgo-alto     {background:#FDEDED;border-left:6px solid #E53935;padding:16px;border-radius:8px;}
.riesgo-moderado {background:#FFF8E1;border-left:6px solid #FFB300;padding:16px;border-radius:8px;}
.riesgo-bajo     {background:#E8F5E9;border-left:6px solid #43A047;padding:16px;border-radius:8px;}
.disclaimer      {background:#F3F4F6;border:1px solid #D1D5DB;border-radius:8px;padding:12px;
                  font-size:12px;color:#6B7280;margin-top:16px;}
</style>
""", unsafe_allow_html=True)

st.title("🫀 Sistema Predictivo DIALYSIS")
st.caption("Riesgo cardio-reno-metabólico · ENSANUT 2022 · XGBoost ROC-AUC 95.9% · Sin tensión arterial")
st.divider()

tab1, tab2 = st.tabs(["👤 Paciente individual", "📋 Carga masiva (CSV)"])

# ── TAB 1: Paciente individual ────────────────────────────────────────────────
with tab1:
    st.subheader("Datos del paciente")
    st.caption("🟢 Normal  🟡 Limítrofe  🔴 Alterado")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demográficos y antropométricos**")
        sexo           = st.selectbox("Sexo", [1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        edad           = st.number_input("Edad (años)", 1, 120, 45)
        peso           = st.number_input("Peso (kg)", 10.0, 300.0, 75.0)
        estatura       = st.number_input("Estatura (m)", 0.5, 2.5, 1.65)
        imc_calc       = round(peso / estatura ** 2, 1)
        masa_corporal  = st.number_input(f"IMC {sem('masa_corporal', imc_calc)}", 10.0, 80.0, imc_calc)
        medida_cintura = st.number_input("Cintura (cm)", 40.0, 200.0, 90.0)
        concentracion_hemoglobina = st.number_input("Hemoglobina (g/dL)", 0.0, 25.0, 14.0)

    with col2:
        st.markdown("**Biomarcadores metabólicos**")
        resultado_glucosa             = st.number_input(f"Glucosa (mg/dL) {sem('resultado_glucosa', 95)}", 0.0, 3000.0, 95.0)
        valor_hemoglobina_glucosilada = st.number_input(f"HbA1c (%) {sem('valor_hemoglobina_glucosilada', 5.5)}", 0.0, 20.0, 5.5)
        valor_insulina                = st.number_input("Insulina (uUI/mL)", 0.0, 500.0, 10.0)
        valor_trigliceridos           = st.number_input(f"Triglicéridos (mg/dL) {sem('valor_trigliceridos', 120)}", 0.0, 2000.0, 120.0)
        valor_colesterol_total        = st.number_input(f"Colesterol total (mg/dL) {sem('valor_colesterol_total', 180)}", 0.0, 600.0, 180.0)
        valor_colesterol_hdl          = st.number_input(f"HDL (mg/dL) {sem('valor_colesterol_hdl', 50)}", 0.0, 200.0, 50.0)
        valor_colesterol_ldl          = st.number_input(f"LDL (mg/dL) {sem('valor_colesterol_ldl', 100)}", 0.0, 400.0, 100.0)
        resultado_glucosa_promedio    = st.number_input(f"Glucosa promedio (mg/dL) {sem('resultado_glucosa_promedio', 95)}", 0.0, 3000.0, 95.0)

    with col3:
        st.markdown("**Biomarcadores renales y otros**")
        valor_acido_urico        = st.number_input(f"Ácido úrico (mg/dL) {sem('valor_acido_urico', 5.0)}", 0.0, 20.0, 5.0)
        valor_creatina           = st.number_input(f"Creatinina (mg/dL) {sem('valor_creatina', 0.9)}", 0.0, 20.0, 0.9)
        valor_albumina           = st.number_input("Albúmina (g/dL)", 0.0, 10.0, 4.0)
        valor_homocisteina       = st.number_input("Homocisteína (umol/L)", 0.0, 100.0, 10.0)
        valor_proteinac_reactiva = st.number_input(f"PCR (mg/L) {sem('valor_proteinac_reactiva', 2.0)}", 0.0, 200.0, 2.0)
        valor_ferritina          = st.number_input("Ferritina (ng/mL)", 0.0, 2000.0, 80.0)
        valor_folato             = st.number_input("Folato (ng/mL)", 0.0, 50.0, 8.0)
        valor_transferrina       = st.number_input("Transferrina (mg/dL)", 0.0, 600.0, 250.0)
        valor_vitamina_bdoce     = st.number_input("Vitamina B12 (pg/mL)", 0.0, 2000.0, 400.0)
        valor_vitamina_d         = st.number_input("Vitamina D (ng/mL)", 0.0, 100.0, 25.0)

    st.markdown("**Factores ambientales y conductuales**")
    colA, colB, colC = st.columns(3)
    with colA:
        temperatura_ambiente = st.number_input("Temperatura ambiente (°C)", -10.0, 50.0, 22.0)
    with colB:
        sueno_horas = st.number_input("Horas de sueño", 0, 24, 7)
    with colC:
        actividad_total = st.number_input("Actividad total (MET-min/sem)", 0, 10000, 500)

    if st.button("🔍 Calcular riesgo", type="primary", use_container_width=True):
        datos = {
            "sexo": sexo, "edad": edad,
            "concentracion_hemoglobina": concentracion_hemoglobina,
            "temperatura_ambiente": temperatura_ambiente,
            "valor_acido_urico": valor_acido_urico,
            "valor_albumina": valor_albumina,
            "valor_colesterol_hdl": valor_colesterol_hdl,
            "valor_colesterol_ldl": valor_colesterol_ldl,
            "valor_colesterol_total": valor_colesterol_total,
            "valor_creatina": valor_creatina,
            "resultado_glucosa": resultado_glucosa,
            "valor_insulina": valor_insulina,
            "valor_trigliceridos": valor_trigliceridos,
            "resultado_glucosa_promedio": resultado_glucosa_promedio,
            "valor_hemoglobina_glucosilada": valor_hemoglobina_glucosilada,
            "valor_ferritina": valor_ferritina,
            "valor_folato": valor_folato,
            "valor_homocisteina": valor_homocisteina,
            "valor_proteinac_reactiva": valor_proteinac_reactiva,
            "valor_transferrina": valor_transferrina,
            "valor_vitamina_bdoce": valor_vitamina_bdoce,
            "valor_vitamina_d": valor_vitamina_d,
            "peso": peso, "estatura": estatura,
            "masa_corporal": masa_corporal,
            "medida_cintura": medida_cintura,
            "sueno_horas": sueno_horas,
            "actividad_total": actividad_total,
        }

        df_p = pd.DataFrame([datos])[features]
        prob = modelo.predict_proba(df_p)[0][1]
        cardio, renal, metab = calcular_subscores(datos)
        factores = interpretar(datos)

        st.divider()

        if prob >= 0.75:    nivel, css, emoji = "ALTO",     "riesgo-alto",     "🔴"
        elif prob >= 0.50:  nivel, css, emoji = "MODERADO", "riesgo-moderado", "🟡"
        else:               nivel, css, emoji = "BAJO",     "riesgo-bajo",     "🟢"

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="{css}">
                <h2>{emoji} Riesgo {nivel}</h2>
                <h1 style="font-size:52px">{prob*100:.1f}%</h1>
                <p>Probabilidad global cardio-reno-metabólico</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.plotly_chart(gauge(prob * 100, "Riesgo global DIALYSIS", 270), use_container_width=True)

        st.divider()
        st.subheader("📊 Desglose por dimensión clínica")
        sc1, sc2, sc3 = st.columns(3)
        sc1.plotly_chart(gauge(cardio, "❤️ Riesgo Cardiovascular"), use_container_width=True)
        sc2.plotly_chart(gauge(renal,  "🫘 Riesgo Renal"),          use_container_width=True)
        sc3.plotly_chart(gauge(metab,  "⚗️ Riesgo Metabólico"),     use_container_width=True)

        st.divider()
        st.subheader("🧠 Principales impulsores de riesgo")
        if factores:
            total_w = sum(f[1] for f in factores)
            fig3 = go.Figure(go.Bar(
                y=[f[0] for f in factores],
                x=[round(f[1] / total_w * 100, 1) for f in factores],
                orientation="h",
                marker_color=["#E53935", "#EF6C00", "#FFB300", "#7986CB", "#26A69A"],
                text=[f"{round(f[1]/total_w*100,1)}%" for f in factores],
                textposition="outside"
            ))
            fig3.update_layout(
                title="Contribución estimada por factor de riesgo",
                xaxis_title="Contribución (%)",
                height=280, margin=dict(t=50, b=20, l=20, r=60),
                xaxis=dict(range=[0, 60])
            )
            st.plotly_chart(fig3, use_container_width=True)
            resumen = " + ".join([f[0] for f in factores[:3]])
            st.info(f"💡 Principal impulsor de riesgo: **{resumen}**")
        else:
            st.success("✅ No se identificaron factores de riesgo clínicos significativos.")

        st.markdown("""
        <div class="disclaimer">
        ⚕️ <strong>Aviso importante:</strong> Herramienta de apoyo predictivo basada en ENSANUT 2022 (n=4,363).
        No sustituye la valoración médica clínica. Resultados deben ser interpretados por personal de salud
        calificado. Modelo XGBoost · ROC-AUC 95.9% (sin tensión arterial) · Validación k-fold k=5.
        </div>""", unsafe_allow_html=True)

# ── TAB 2: Carga masiva ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Carga un archivo CSV con múltiples pacientes")
    st.info("El CSV debe contener las mismas columnas que el dataset de entrenamiento (sin tension_arterial).")
    archivo = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])
    if archivo:
        df_nuevos = pd.read_csv(archivo)
        st.success(f"✅ {len(df_nuevos)} registros cargados")
        st.dataframe(df_nuevos.head(), use_container_width=True)
        cols_faltantes = [c for c in features if c not in df_nuevos.columns]
        if cols_faltantes:
            st.error(f"Columnas faltantes: {cols_faltantes}")
        else:
            if st.button("🔍 Predecir todos los pacientes", type="primary"):
                probs = modelo.predict_proba(df_nuevos[features])[:, 1]
                df_nuevos["probabilidad_riesgo_%"] = (probs * 100).round(1)
                df_nuevos["nivel_riesgo"] = pd.cut(
                    probs, bins=[0, 0.50, 0.75, 1.0],
                    labels=["🟢 Bajo", "🟡 Moderado", "🔴 Alto"],
                    include_lowest=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Alto",     (df_nuevos["nivel_riesgo"] == "🔴 Alto").sum())
                c2.metric("🟡 Moderado", (df_nuevos["nivel_riesgo"] == "🟡 Moderado").sum())
                c3.metric("🟢 Bajo",     (df_nuevos["nivel_riesgo"] == "🟢 Bajo").sum())
                st.dataframe(
                    df_nuevos[["probabilidad_riesgo_%", "nivel_riesgo"] + features]
                    .sort_values("probabilidad_riesgo_%", ascending=False),
                    use_container_width=True)
                csv_out = df_nuevos.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados CSV", csv_out,
                    "resultados_dialysis.csv", "text/csv", use_container_width=True)
                st.markdown("""
                <div class="disclaimer">
                ⚕️ Herramienta de apoyo predictivo · ENSANUT 2022 · No sustituye valoración médica.
                </div>""", unsafe_allow_html=True)

st.divider()
st.caption("Sistema DIALYSIS · XGBoost · ROC-AUC 95.9% · ENSANUT 2022 · Uso exclusivo de apoyo clínico")
