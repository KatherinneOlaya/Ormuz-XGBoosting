import streamlit as st
import joblib
import pandas as pd
import numpy as np



# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Aramco Brent Predictor Terminal",
    page_icon="🔮",
    layout="wide"
)

# =====================================================================
# CARGA DEL MODELO (CEREBRO DE LA IA)
# =====================================================================
@st.cache_resource  # Esto evita que el modelo se recargue cada vez que mueves un botón
def cargar_modelo():
    modelo = joblib.load('modelo_brent_xgb_v3.joblib')
    features = joblib.load('features_brent_v3.joblib')
    return modelo, features

try:
    modelo_xgb, lista_features = cargar_modelo()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    st.error(f"❌ No se pudo cargar el archivo del modelo. Asegúrate de que 'modelo_brent_xgb_v3.joblib' esté en la misma carpeta. Error: {e}")

# =====================================================================
# INTERFAZ GRÁFICA
# =====================================================================
st.title("🔮 Terminal de Inteligencia Predictiva: Precio del Brent")
st.markdown("---")

if modelo_cargado:
    st.sidebar.header("📊 Datos de Operación de Hoy")
    st.sidebar.markdown("Ingresa los valores actuales para proyectar el precio de mañana.")

    # 1. Creamos los controles interactivos en la barra lateral basados en tus variables más importantes
    brent_std_7 = st.sidebar.number_input("Volatilidad del Crudo (7 días) [brent_std_7]", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    volatilidad = st.sidebar.number_input("Incertidumbre de Mercado (Volatilidad Hoy)", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
    brent_lag3 = st.sidebar.number_input("Precio del Brent hace 3 días ($) [brent_lag3]", min_value=50.0, max_value=150.0, value=87.5)
    vol_exp_ksa = st.sidebar.number_input("Exportaciones de KSA (Millones de Barriles) [vol_exp_ksa]", min_value=0.0, max_value=20.0, value=7.2, step=0.1)
    flete_change = st.sidebar.number_input("Cambio Porcentual del Flete (%) [flete_change]", min_value=-20.0, max_value=20.0, value=2.5, step=0.5)
    brent_lag1 = st.sidebar.number_input("Precio del Brent de Ayer ($) [brent_lag1]", min_value=50.0, max_value=150.0, value=88.0)

    # 2. BOTÓN DE PREDICCIÓN
    if st.sidebar.button("🔮 Calcular Predicción para Mañana"):
        
        # 3. Construimos el diccionario con los datos ingresados por el usuario
        # OJO: Ponemos exactamente los mismos nombres de variables que X_train
        datos_usuario = {
            'brent_std_7': brent_std_7,
            'volatilidad': volatilidad,
            'brent_lag3': brent_lag3,
            'vol_exp_ksa': vol_exp_ksa,
            'flete_change': flete_change,
            'brent_lag1': brent_lag1
        }
        
        # Convertimos a DataFrame para que XGBoost lo entienda
        df_input = pd.DataFrame([datos_usuario])
        
        # SECRETO PROFESIONAL: Reordenamos las columnas para asegurarnos de que tengan el mismo orden exacto del entrenamiento
        # Si hay variables que faltan en la interfaz (como ormuz_lag_1 o day_of_week que quedaron en 0%), las rellenamos con 0
        for col in lista_features:
            if col not in df_input.columns:
                df_input[col] = 0.0
                
        df_input = df_input[lista_features] # Forzar orden estricto
        
        # 4. EJECUTAR PREDICCIÓN
        prediccion = modelo_xgb.predict(df_input)[0]
        
        # 5. MOSTRAR RESULTADOS EJECUTIVOS
        st.subheader("🎯 Resultado de la Simulación")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Precio Proyectado del Brent (Mañana)", value=f"${prediccion:.2f} USD")
        with col2:
            margen_error = 1.01 # Tu MAE del modelo V3
            st.metric(label="Margen de Variación Esperado (MAE)", value=f"± ${margen_error:.2f} USD")
            
        # 6. RECOMENDACIÓN OPERATIVA AUTOMÁTICA (Lógica de Negocio)
        st.markdown("### 💡 Sugerencia Estratégica para Supply Chain:")
        if prediccion > brent_lag1 + 0.5:
            st.success("🔼 **TENDENCIA ALCISTA DETECTADA:** Se sugiere retener inventario en los tanques hoy. El precio subirá mañana, permitiendo vender los barriles a un mayor valor o asegurar fletes antes del aumento.")
        elif prediccion < brent_lag1 - 0.5:
            st.warning("🔽 **TENDENCIA BAJISTA DETECTADA:** Se sugiere acelerar las órdenes de venta y despacho hoy mismo para asegurar el precio actual antes de la caída de mañana.")
        else:
            st.info("🔄 **MERCADO ESTABLE:** El precio se mantendrá dentro de los rangos normales. Mantener la estrategia operativa estándar sin compras de pánico.")
            
else:
    st.info("💡 Por favor, coloca los archivos descargados (.joblib) en la misma carpeta que este script para activar la terminal.")
    
