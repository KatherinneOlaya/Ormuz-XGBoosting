import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
st.set_page_config(
    page_title=" Brent Predictor Terminal",
    page_icon="🛢️",
    layout="wide"
)

# =====================================================================
# CSS 
# =====================================================================
st.markdown("""
<style>

.block-container{
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #111827;
    padding: 1rem;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================================
# CARGA DEL MODELO
# =====================================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_brent_xgb_v3.joblib')
    features = joblib.load('features_brent_v3.joblib')
    return modelo, features

try:
    modelo_xgb, lista_features = cargar_modelo()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    st.error(f"❌ Error cargando modelo: {e} ❌❌❌")

# =====================================================================
# SIDEBAR
# =====================================================================
if modelo_cargado:

    with st.sidebar:

        st.header("📊 Datos de Operación de Hoy")

        st.markdown("""
        Ingresa los valores actuales para proyectar el precio de mañana.
        """)

        brent_std_7 = st.number_input(
            "Volatilidad del Crudo (7 días)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1
        )

        volatilidad = st.number_input(
            "Incertidumbre de Mercado",
            min_value=0.0,
            max_value=5.0,
            value=1.2,
            step=0.1
        )

        brent_lag3 = st.number_input(
            "Precio Brent hace 3 días",
            min_value=50.0,
            max_value=150.0,
            value=87.5
        )

        vol_exp_ksa = st.number_input(
            "Exportaciones KSA",
            min_value=0.0,
            max_value=20.0,
            value=7.2,
            step=0.1
        )

        flete_change = st.number_input(
            "Cambio del Flete (%)",
            min_value=-20.0,
            max_value=20.0,
            value=2.5,
            step=0.5
        )

        brent_lag1 = st.number_input(
            "Precio Brent de Ayer",
            min_value=50.0,
            max_value=150.0,
            value=88.0
        )

        ejecutar = st.button("✈️ Calcular Predicción")

# =====================================================================
# MAIN UI
# =====================================================================

    st.markdown("""
    <div class='main-title'>
    🛢️ Machine Learning aplicado:
     Averiguando el Precio del Brent ✨🛢️ 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='subtitle'>
    Sistema de predicción operativa basado en XGBoost para decisiones estratégicas de supply chain y trading energético. (Proyecto amateur :D)
    https://github.com/KatherinneOlaya
    </div>
    """, unsafe_allow_html=True)

    if ejecutar:

        datos_usuario = {
            'brent_std_7': brent_std_7,
            'volatilidad': volatilidad,
            'brent_lag3': brent_lag3,
            'vol_exp_ksa': vol_exp_ksa,
            'flete_change': flete_change,
            'brent_lag1': brent_lag1
        }

        df_input = pd.DataFrame([datos_usuario])

        for col in lista_features:
            if col not in df_input.columns:
                df_input[col] = 0.0

        df_input = df_input[lista_features]

        prediccion = modelo_xgb.predict(df_input)[0]

        margen_error = 1.01
        delta = prediccion - brent_lag1

        # ============================================================
        # KPIs
        # ============================================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Precio Proyectado",
                f"${prediccion:.2f}",
                f"{delta:.2f} USD"
            )

        with c2:
            st.metric(
                "Margen Esperado",
                f"± ${margen_error:.2f}"
            )

        with c3:
            tendencia = "ALCISTA" if delta > 0 else "BAJISTA"
            st.metric(
                "Tendencia",
                tendencia
            )

        st.divider()

        # ============================================================
        # RECOMENDACIÓN
        # ============================================================

        st.subheader("💡 Recomendación Estratégica")

        if prediccion > brent_lag1 + 0.5:

            st.success("""
            🔼 TENDENCIA ALCISTA DETECTADA

            • Retener inventario temporalmente  
            • Asegurar contratos de transporte hoy  
            • Posible incremento del valor spot mañana
            """)

        elif prediccion < brent_lag1 - 0.5:

            st.warning("""
            🔽 TENDENCIA BAJISTA DETECTADA

            • Acelerar despachos  
            • Ejecutar ventas antes de caída esperada  
            • Evitar acumulación de inventario
            """)

        else:

            st.info("""
            🔄 MERCADO ESTABLE

            • Mantener operación estándar  
            • No se esperan cambios bruscos
            """)

else:

    st.info("""
    💡 Coloca los archivos .joblib en la misma carpeta para activar el sistema.
    """)
