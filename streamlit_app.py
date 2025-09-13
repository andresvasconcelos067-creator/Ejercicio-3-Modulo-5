
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Qwen2-0.5B Sentiment (Zero-shot vs Few-shot)", page_icon="🎬", layout="centered")
st.title("🎬 Sentiment classifier with Qwen2-0.5B — Zero‑shot vs Few‑shot")
st.caption("Modelo: Qwen/Qwen2-0.5B · Librería: 🤗 transformers · Interfaz: Streamlit")

@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "Qwen/Qwen2-0.5B"
    tok = AutoTokenizer.from_pretrained(model_id)
    # Cargar en CPU por compatibilidad; si hay GPU disponible, se usará dtype apropiado
    kwargs = {}
    if torch.cuda.is_available():
        kwargs.update(dict(torch_dtype=torch.float16, device_map="auto"))
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tok, mdl

tokenizer, model = load_model()

st.subheader("Parámetros de inferencia")
max_new_tokens = st.slider("max_new_tokens", 4, 64, 8, step=1)
temperature = st.slider("temperature", 0.0, 1.5, 0.2, step=0.1)

def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

def postprocess_to_label(text: str) -> str:
    # Busca la última ocurrencia para evitar contaminar con el prompt
    lower = text.strip().lower()
    for lbl in ["positivo", "negativo", "neutro"]:
        if lbl in lower[-80:]:
            return lbl.capitalize()
    # fallback por si el modelo responde en inglés
    for eng, esp in [("positive","Positivo"),("negative","Negativo"),("neutral","Neutro")]:
        if eng in lower[-80:]:
            return esp
    # último recurso: heurística simple
    return "Neutro"

st.subheader("📍 Zero‑shot")
st.write("Clasificación directa sin ejemplos de contexto.")
zs_prompt_tmpl = (
    "Clasifica el **sentimiento** de la siguiente reseña de película en una sola palabra, "
    "elegida exactamente de entre: [Positivo, Negativo, Neutro].\n\n"
    "Reseña: \"{review}\"\n"
    "Respuesta (una sola palabra): "
)
review_input = st.text_area("Escribe una reseña", placeholder="La fotografía fue hermosa pero la historia me aburrió.", height=100)
zs_btn = st.button("Clasificar (Zero‑shot)")
zs_result = None
if zs_btn and review_input.strip():
    zs_text = generate(zs_prompt_tmpl.format(review=review_input.strip()))
    zs_result = postprocess_to_label(zs_text)
    st.success(f"Zero‑shot: **{zs_result}**")
    with st.expander("Ver salida completa del modelo (zero‑shot)"):
        st.code(zs_text)

st.subheader("🧩 Few‑shot (3–5 ejemplos)")
st.write("Incluye ejemplos de reseñas etiquetadas como **contexto** para guiar al modelo.")
default_examples = [
    ("Me encantó el desarrollo de los personajes y el final me hizo llorar.", "Positivo"),
    ("Las actuaciones fueron malas y el guion es un desastre.", "Negativo"),
    ("Tiene momentos buenos y otros flojos; en general es pasable.", "Neutro"),
    ("La banda sonora es impresionante, pero el ritmo es irregular.", "Neutro"),
    ("Una obra maestra visual con una historia poderosa.", "Positivo"),
]
with st.expander("Ejemplos (puedes editarlos)"):
    examples = []
    for i, (txt, lbl) in enumerate(default_examples, start=1):
        c1, c2 = st.columns([3,1])
        with c1:
            txt = st.text_input(f"Ejemplo {i} — Reseña", value=txt, key=f"ex_txt_{i}")
        with c2:
            lbl = st.selectbox(f"Etiqueta {i}", ["Positivo","Negativo","Neutro"], index=["Positivo","Negativo","Neutro"].index(lbl), key=f"ex_lbl_{i}")
        examples.append((txt, lbl))

fs_prompt_head = (
    "Eres un asistente que clasifica **sentimientos** en reseñas de películas. "
    "Responde siempre con **una sola palabra** de esta lista exacta: Positivo, Negativo, Neutro.\n\n"
    "### Ejemplos\n"
)
def build_fewshot_prompt(examples, review):
    blocks = []
    for txt, lbl in examples:
        blocks.append(f"Reseña: \"{txt}\"\nSentimiento: {lbl}\n")
    blocks_text = "\n".join(blocks)
    return f"{fs_prompt_head}{blocks_text}\n### Tarea\nReseña: \"{review}\"\nSentimiento: "

fs_btn = st.button("Clasificar (Few‑shot)")
if fs_btn and review_input.strip():
    fs_prompt = build_fewshot_prompt(examples, review_input.strip())
    fs_text = generate(fs_prompt)
    fs_result = postprocess_to_label(fs_text)
    st.info(f"Few‑shot: **{fs_result}**")
    if zs_result is not None:
        st.write("**Comparación:**")
        st.write(f"- Zero‑shot → {zs_result}")
        st.write(f"- Few‑shot  → {fs_result}")
    with st.expander("Ver salida completa del modelo (few‑shot)"):
        st.code(fs_text)

st.subheader("📁 Carga masiva (opcional)")
st.write("Sube un archivo **.txt** (una reseña por línea) o **.csv** (columna `review`).")
uploaded = st.file_uploader("Selecciona archivo .txt o .csv", type=["txt","csv"])
mode = st.radio("Modo de clasificación para carga masiva", ["Zero‑shot","Few‑shot"], horizontal=True)
if uploaded is not None:
    import pandas as pd, io
    try:
        if uploaded.name.lower().endswith(".txt"):
            reviews = [line.strip() for line in io.StringIO(uploaded.getvalue().decode("utf-8")).read().splitlines() if line.strip()]
            df = pd.DataFrame({"review": reviews})
        else:
            df = pd.read_csv(uploaded)
            if "review" not in df.columns:
                st.error("El .csv debe contener una columna llamada 'review'.")
                st.stop()
        preds = []
        for r in df["review"].tolist():
            if mode == "Zero‑shot":
                txt = generate(zs_prompt_tmpl.format(review=r))
            else:
                txt = generate(build_fewshot_prompt(examples, r))
            preds.append(postprocess_to_label(txt))
        df["pred"] = preds
        st.success(f"Clasificadas {len(df)} reseñas.")
        st.dataframe(df.head(50))
        # descarga
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados (.csv)", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Ocurrió un error procesando el archivo: {e}")

st.divider()
st.write("**Notas:**")
st.write("- Si tienes GPU, se usará automáticamente; de lo contrario, la inferencia será en CPU.")
st.write("- El modelo es pequeño (0.5B) y puede cometer errores; few‑shot suele mejorar la estabilidad.")

import os
os.system("python -m streamlit run streamlit_app.py")