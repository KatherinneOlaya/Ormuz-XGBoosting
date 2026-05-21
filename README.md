# 🛢️ Ormuz-XGBoosting: Inteligencia Predictiva para el Crudo Brent ✨
<img width="1501" height="831" alt="image" src="https://github.com/user-attachments/assets/fe1cc78a-50a4-49e1-8868-863bca3fe009" />

### 🛢️ ¿Por qué este proyecto? (Enfoque de Mercado) ✨

Este es un desarrollo con fines educativos y de experimentación analítica (amateur), nacido de la curiosidad por entender cómo los eventos del mundo real impactan los mercados financieros. 

Para predecir el precio del barril a 24 horas, el modelo no se limita a mirar el precio de ayer; en su lugar, integra variables fundamentales de la oferta, la demanda y el riesgo geopolítico global, fundamentado en tres pilares:

   $$y_{t+1}=f(y_t,y_{t-1},x_t)$$

1. **El Gigante de la Oferta:** Monitoreamos las exportaciones de Arabia Saudita (`vol_exp_ksa`), entendiendo que cualquier movimiento del mayor exportador de crudo del mundo mueve la balanza comercial.
2. **El Cuello de Botella Logístico:** Evaluamos las congestiones en el Estrecho de Ormuz, el punto de tránsito marítimo más crítico del petróleo global, donde el tráfico y los fletes marcan el pulso del suministro físico.
3. **El Termómetro del Miedo:** Analizamos la volatilidad en la bolsa de valores y la desviación estándar del Brent, capturando el sentimiento de incertidumbre, especulación y pánico de los inversores en tiempo real.


---

## 📖 Contexto del Proyecto ✨

El mercado energético global sufre de una alta volatilidad debido a factores geopolíticos y shocks en la cadena de suministro física. Para este análisis, el sistema fue entrenado y validado utilizando un dataset de simulación de alta fidelidad que simula 45 días de datos de alta frecuencia, modelados a partir de los patrones de comportamiento observados desde el inicio de la crisis en el Estrecho de Ormuz.

Este enfoque de modelación de escenarios permitió estresar el algoritmo XGBoost bajo condiciones críticas de mercado, evaluando su capacidad de respuesta ante picos de volatilidad controlados, fluctuaciones reales de la oferta de Arabia Saudita y cuellos de botella logísticos sin los sesgos de ruidos externos ajenos a la crisis.

A través de un proceso iterativo de optimización de hiperparámetros y *Feature Engineering*, el algoritmo **XGBoost (Versión 3)** logró descifrar los patrones de fondo del mercado, pasando de un desfase severo a un **ajuste óptimo impecable** con un margen de error absoluto medio (**MAE**) de tan solo **$1.01 USD**. Teniendo en cuenta que el modelo logró adaptarse con un datos limitados, que son lo que se afrontan en un contexto de crisis geopolítica.

---

## 🔬 El Cerebro de la IA: Motores del Precio ✨

Gracias a las herramientas de interpretabilidad del modelo (*Feature Importance*), descubrimos que la IA no adivina al azar, sino que opera bajo una lógica financiera y física perfecta:

* 🛢️ **Volatilidad y Riesgo de Mercado (49.4%):** Las variables `brent_std_7` y `volatilidad` dominan las decisiones del algoritmo. El modelo detecta el "miedo" o nerviosismo a corto plazo para anticipar movimientos drásticos.
* 🛢️ **Efecto Memoria Cíclica (11.6%):** La variable `brent_lag3` (precio de hace 3 días) resultó ser más influyente que el precio de ayer, demostrando que el modelo busca tendencias macro y descarta el ruido diario.
* 🛢️ **El Factor Físico de Suministro (6.6%):** El volumen de exportaciones de Arabia Saudita (`vol_exp_ksa`) actúa como el estabilizador de la oferta dentro de las ecuaciones internas del algoritmo.

---

## 🛠️ Evolución y MLOps (De Laboratorio a Producción) ✨

El proyecto atravesó tres fases estrictas de maduración para erradicar el *Overfitting*:
1. **V1 (Original):** Sobreajustado al pasado (MAE: $1.99 USD / $R^2$: -309%).
2. **V2 (Regularizado):** Excesivamente penalizado, cayendo en *Underfitting* (MAE: $1.62 USD).
3. **V3 (Optimizado):** **Ajuste Perfecto.** Equilibrio exacto capaz de predecir rebotes y picos de crisis en tiempo real (MAE: $1.01 USD).

---

## 🚀 Guía de Instalación y Ejecución Local ✨

Para evitar transferir archivos pesados e incompatibles entre sistemas, este proyecto no incluye la carpeta del entorno virtual. En su lugar, puedes reconstruir el ecosistema exacto de ejecución siguiendo estos pasos en tu terminal:

### 1. Clonar el repositorio e ingresar a la carpeta
```bash
git clone (https://github.com/KatherinneOlaya/Ormuz-XGBoosting)
cd Ormuz-XGBoosting
````

### 2. Crear y activar tu entorno virtual limpio 
```bash
python -m venv aramco_env
aramco_env\Scripts\activate
````

### 3. Instalar las dependecias.
```bash
pip install -r requirements.txt
````

## Uso de Terminal de Streamlit ✨
Aquí para levantar la terminal es necesario tener el entorno activado y ejecutar el siguiente comando:
```bash
streamlit run app.py
````

### ¿Qué hace la app? 🦋
1. **Entrada de Datos:** Permite ingresar los indicadores de volatilidad actual, precios retrasados y volumen de exportación de KSA desde una barra lateral intuitiva.

2. **Predicción Instantánea:** XGBoost calcula la proyección para el día de mañana junto a su margen de variación estimado.

3. **Sugerencia Estratégica Automatizada:** Traduce la predicción numérica en una instrucción de negocios directa para el equipo de compras ("Retener Inventario / Venta Inmediata / Mercado Estable").
