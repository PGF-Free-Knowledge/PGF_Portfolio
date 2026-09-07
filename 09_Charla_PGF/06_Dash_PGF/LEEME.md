# Dashboard — De medir a predecir

Dashboard interactivo (Streamlit) para la charla *"Desarrollo de modelos y
arquitecturas inteligentes para apoyar la gestión, predicción, explicación
y optimización del desempeño energético en infraestructuras críticas"*.

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run dashboard_energia.py
```

Mantén `synthetic_datacenter_hourly_3days.csv` y
`synthetic_tower_hourly_3days.csv` en la misma carpeta que el script — el
dashboard los carga automáticamente.

## Datasets disponibles (selector en la barra lateral)

1. **Data Center — dataset piloto real** (`synthetic_datacenter_hourly_3days.csv`).
   Es el dataset con el que desarrollaste el pipeline de ML/SHAP en
   `Procesoinicialpgfenergia2.ipynb`. Métricas reproducibles: Ridge R²=0,979
   (MAE 10,18 kW) — Random Forest R²=0,911 (MAE 19,41 kW), el mismo usado
   para XAI/SHAP. Estas cifras coinciden con las de la diapositiva 17 del PPTX.

2. **Torre de telecomunicaciones — dataset piloto real**
   (`synthetic_tower_hourly_3days.csv`). ⚠️ En este dataset,
   `total_consumption_kW` es exactamente `radio_load_kW + generator_kW`
   (identidad algebraica) — por eso Ridge da R²≈1. Se deja disponible como
   caso de estudio, pero no es un resultado de aprendizaje real; el
   dashboard muestra esta advertencia en pantalla.

3. **Planta industrial — datos sintéticos (demo)**. Generador in-memory
   (no viene de ningún archivo real), útil para mostrar el simulador de
   escenarios con una variable controlable tipo "setpoint".

También puedes cargar tu propio CSV desde la barra lateral (mismas columnas
que el dataset elegido).

## Pantallas

1. **Planta** — estado general (KPIs del último registro + serie completa).
2. **Desempeño** — línea base energética (LBEn) y CUSUM, calculados sobre
   la variable de carga principal de cada dataset.
3. **Predicción ML** — Ridge y Random Forest entrenados en vivo (70/30,
   igual que el notebook), con métricas y gráfico real vs. predicho.
4. **XAI / SHAP** — importancia global (bar, beeswarm) y explicación local
   (waterfall) sobre el Random Forest.
5. **Escenarios** — simulador what-if con sliders adaptados a cada dataset
   (carga TI/HVAC/temperatura en Data Center; carga de radio/batería/
   generador/temperatura en la Torre; carga/temperatura/ocupación/setpoint
   en el demo).
6. **Decisión** — comparación de estrategia reactiva (tipo ISO 50001) vs.
   predictiva (IA):
   - En los datasets reales: se compara sobre el **mismo** conjunto de
     validación, aplicando un supuesto de corrección fijo y explícito
     (1% reactivo / 3% predictivo) — determinista y reproducible.
   - En el dataset demo: se mantiene la búsqueda de setpoint óptimo por
     grilla horaria.

   Nota: el notebook original tenía una comparación ISO-vs-IA que mezclaba
   subconjuntos de datos distintos y usaba aleatoriedad sin semilla fija —
   ese cálculo no se reutilizó tal cual; esta versión corrige ambos
   problemas para que el número mostrado sea confiable.

## Notas para la demo en vivo

- El flujo de pestañas sigue tu hilo conductor:
  Planta → Datos → Desempeño → ML → Predicción → XAI/SHAP → Escenarios → Decisión.
- Las pestañas avisan explícitamente cuándo un número es ilustrativo (modo
  demo, o los supuestos de corrección en Decisión) para no presentar cifras
  sintéticas como reales.
- Recomendado: dejar el dashboard abierto en una pestaña del navegador y
  VS Code en otra, alternando según tu guion (PowerPoint → Dashboard →
  Python/VS Code → Dashboard → PowerPoint). El PPTX (`charla_energia.pptx`)
  ya incluye los llamados "▶ DEMO 1/2/3" en las diapositivas 17, 19 y 20.
