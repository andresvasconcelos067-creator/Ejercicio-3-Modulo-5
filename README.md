# Ejercicio-3-Modulo-5

# Sentiment Qwen2-0.5B — Streamlit (Zero-shot vs Few-shot)

Aplicación simple en Streamlit que usa **Qwen/Qwen2-0.5B** (🤗 transformers) para clasificar sentimientos en reseñas de películas, comparando **Zero-shot** y **Few-shot**.

## Requisitos
- Python 3.10+
- pip / conda
- (Opcional) GPU con CUDA

## Instalación rápida
```bash
# 1) Crear entorno
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 2) Instalar dependencias
pip install streamlit transformers torch --extra-index-url https://download.pytorch.org/whl/cu121  # si tienes CUDA 12.1
# Si vas en CPU:
pip install streamlit transformers torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Ejecutar localmente
```bash
streamlit run streamlit_app.py
```

## Uso
- Escribe una reseña en el área de texto y ejecuta en **Zero-shot** o **Few-shot**.
- En **Few-shot**, puedes editar los ejemplos (3–5) antes de inferir.
- **Carga masiva (opcional):** sube `.txt` (una reseña por línea) o `.csv` con columna `review`. Descarga resultados como CSV.

## Estructura
- `streamlit_app.py` — App principal con UI Streamlit y lógica de inferencia.

## Notas de reproducibilidad
- El modelo se descarga automáticamente de Hugging Face: `Qwen/Qwen2-0.5B`.
- Se cachea el modelo durante la sesión con `st.cache_resource`.
- Parámetros de inferencia `max_new_tokens` y `temperature` configurables en la UI.

## Criterios de evaluación (checklist)
- [x] **Zero-shot:** carga Qwen2-0.5B, prompt directo, procesa reseña sin ejemplos y muestra etiqueta Positivo/Negativo/Neutro.
- [x] **Few-shot:** 3–5 ejemplos incluidos en el prompt, procesa reseña y compara con Zero-shot.
- [x] **Modelo:** Qwen2-0.5B vía `transformers`.
- [x] **Interfaz:** Streamlit simple (inputs, sliders, resultados, comparativa).
- [x] **Archivos:** carga opcional `.txt` / `.csv` para clasificación masiva + descarga CSV.
- [x] **Errores:** manejo básico de errores y validaciones (columna `review`, excepciones de carga).
- [x] **Ejecución local:** instrucciones en README.

## Licencia
MIT
