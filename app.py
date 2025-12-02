import streamlit as st
from PIL import Image, ImageChops
import io
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tempfile
import os

# -------------------------
# Utilidades
# -------------------------
def read_image_bytes(uploaded_file):
    data = uploaded_file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

# ELA (Error Level Analysis) - versión simple y segura en Windows
def ela_image(pil_img, quality=90, scale=15):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_name = tmp.name
    tmp.close()
    pil_img.save(tmp_name, "JPEG", quality=quality)
    with Image.open(tmp_name) as recompressed:
        recompressed = recompressed.convert("RGB")
        diff = ImageChops.difference(pil_img, recompressed)
        arr = np.asarray(diff, dtype=np.float32) * scale
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        ela = Image.fromarray(arr)
    try:
        os.unlink(tmp_name)
    except Exception:
        pass
    return ela

# Degrade: re-compress + blur to amplify diferencias
def degraded_image(pil_img, quality=60, blur_radius=2):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_name = tmp.name
    tmp.close()
    pil_img.save(tmp_name, "JPEG", quality=quality)
    with Image.open(tmp_name) as recompressed:
        recompressed = recompressed.convert("RGB")
        degraded = recompressed.filter(Image.Filter.GaussianBlur(radius=blur_radius))
    try:
        os.unlink(tmp_name)
    except Exception:
        pass
    return degraded

# -------------------------
# Extracción de parches y embeddings
# -------------------------
def extract_patches(img_pil, patch_size=64, stride=32):
    """Devuelve lista de patches y su grid shape (rows, cols)."""
    img = img_pil.convert("RGB")
    w, h = img.size
    # pad to multiple of stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    pad_h = (stride - (h - patch_size) % stride) % stride
    if pad_w or pad_h:
        new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0,0,0))
        new_img.paste(img, (0,0))
        img = new_img
        w, h = img.size

    patches = []
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        row = []
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))
    rows = (h - patch_size) // stride + 1
    cols = (w - patch_size) // stride + 1
    return patches, positions, (rows, cols), (w, h)

def batch_embeddings(patches, model, device, batch_size=32, transform=None):
    """Recibe lista de PIL patches y devuelve embeddings numpy (N, D)."""
    if transform is None:
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            tensor_batch = torch.stack([transform(p) for p in batch]).to(device)
            feats = model(tensor_batch)  # expected shape (B, D)
            feats = feats.detach().cpu().numpy()
            embs.append(feats)
    embs = np.vstack(embs)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = embs / norms
    return embs

# -------------------------
# Algoritmo de detección (kNN anomaly on patch embeddings)
# -------------------------
def patch_anomaly_scores(embeddings, k=6):
    """
    Para cada embedding, calcula la distancia media a sus k vecinos más cercanos.
    Distancias altas -> anomalía (posible edición).
    """
    nbrs = NearestNeighbors(n_neighbors=min(k+1, embeddings.shape[0]), algorithm='auto', metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    # distances incluye el elemento mismo (distancia 0) en la columna 0
    # descartamos distancia 0 y promediamos las siguientes k
    if distances.shape[1] > 1:
        mean_dist = distances[:, 1:].mean(axis=1)
    else:
        mean_dist = distances[:, 0]
    # Normalizar 0..1
    minv, maxv = mean_dist.min(), mean_dist.max()
    if maxv - minv < 1e-6:
        scores = np.zeros_like(mean_dist)
    else:
        scores = (mean_dist - minv) / (maxv - minv)
    return scores

# -------------------------
# Construcción de heatmap y overlay
# -------------------------
def build_heatmap(scores, grid_shape, image_size, patch_size=64, stride=32, sigma=8):
    rows, cols = grid_shape
    score_map = np.zeros((rows, cols), dtype=float)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            score_map[r, c] = scores[idx]
            idx += 1
    # upsample to image_size
    heat_small = score_map
    heat = np.kron(heat_small, np.ones((stride, stride)))  # nearest upsample
    heat = heat[:image_size[1], :image_size[0]]  # crop to original
    # smooth
    heat = gaussian_filter(heat, sigma=sigma)
    # normalize
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return heat

def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5, cmap='jet'):
    arr = np.asarray(img_pil).astype(np.uint8)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(arr)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, extent=(0, arr.shape[1], arr.shape[0], 0))
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# -------------------------
# Streamlit UI + Pipeline
# -------------------------
st.set_page_config(page_title="Detector Deep Learning de Edición", layout="wide")
st.title("🔬 Detector De Imágenes Editadas")
st.markdown("Sube una imagen, el modelo extrae embeddings por parches y computa un mapa de anomalías (sin necesidad de dataset de entrenamiento).")

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Sube imagen (jpg/png)", type=["jpg","jpeg","png"])
    st.markdown("**Parámetros**")
    patch_size = st.selectbox("Tamaño de parche", options=[32,48,64,96], index=2)
    stride = st.selectbox("Stride (paso entre parches)", options=[16,24,32,48], index=2)
    k = st.slider("k vecinos (kNN)", 1, 12, 6)
    use_gpu = st.checkbox("Usar GPU si está disponible", value=False)

with col2:
    st.markdown("**Visualizaciones forenses**")
    show_ela = st.checkbox("Generar ELA (Error Level Analysis)", value=True)
    show_degrade = st.checkbox("Generar imagen degradada y sustracción", value=True)
    alpha = st.slider("Opacidad overlay heatmap", 0.1, 0.9, 0.45)

if uploaded is not None:
    st.info("Preparando imagen...")
    original = read_image_bytes(uploaded)
    st.image(original, caption="Original", use_column_width=True)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    st.write(f"Usando device: {device}")

    # modelo backbone: ResNet50 preentrenado (cortamos al avgpool)
    st.info("Cargando modelo backbone preentrenado (esto puede tardar unos segundos la primera vez)...")
    resnet = models.resnet50(pretrained=True)
    # quitamos la última capa fully connected y usamos el feature vector de avgpool
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.to(device)
    feature_extractor.eval()

    # 1) crear patches
    patches, positions, grid_shape, image_size = extract_patches(original, patch_size=patch_size, stride=stride)
    st.write(f"Parches: {len(patches)}  Grid: {grid_shape}  Tamaño imagen: {image_size}")

    # 2) embeddings
    st.info("Extrayendo embeddings de parches...")
    # wrapper to return flattened vector from feature_extractor
    def forward_get_feats(x):
        with torch.no_grad():
            f = feature_extractor(x)    # shape (B, 2048, 1, 1)
            f = f.reshape(f.size(0), -1)
            return f

    # create transform that resizes patch to 224 and returns tensor
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # batch embeddings (we'll use the forward_get_feats by temporary monkeypatch)
    # build DataLoader-like batching
    embs = []
    batch_size = 32
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        tensor_batch = torch.stack([transform(p) for p in batch]).to(device)
        with torch.no_grad():
            feats = forward_get_feats(tensor_batch)
            feats = feats.cpu().numpy()
            embs.append(feats)
    embs = np.vstack(embs)
    # normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    st.info("Calculando puntuaciones de anomalía (kNN)...")
    scores = patch_anomaly_scores(embs, k=k)

    st.info("Construyendo heatmap...")
    heat = build_heatmap(scores, grid_shape, image_size, patch_size=patch_size, stride=stride, sigma=max(3, patch_size//8))

    # overlay
    overlay = overlay_heatmap_on_image(original, heat, alpha=alpha)
    st.image(overlay, caption="Overlay heatmap (zonas sospechosas)", use_column_width=True)

    # mostrar mapa solo heatmap
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('off')
    ax.imshow(heat, cmap='inferno')
    st.pyplot(fig)

    # Mostrar ELA y degradado
    if show_ela:
        st.info("Generando ELA...")
        ela = ela_image(original, quality=90, scale=14)
        st.image(ela, caption="ELA (Error Level Analysis)", use_column_width=True)

    if show_degrade:
        st.info("Generando imagen degradada y sustracción...")
        # recompress and subtract to highlight differences
        degraded = original.copy()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmpname = tmp.name
        tmp.close()
        degraded.save(tmpname, "JPEG", quality=60)
        with Image.open(tmpname) as dec:
            dec = dec.convert("RGB")
            diff = ImageChops.difference(original, dec)
        try:
            os.unlink(tmpname)
        except:
            pass
        st.image(dec, caption="Imagen degradada (recompress 60%)", use_column_width=True)
        st.image(diff, caption="Sustracción Original - Degradada", use_column_width=True)

    # Interpretación simple
    mean_score = float(np.mean(scores))
    st.markdown("---")
    st.subheader("Interpretación rápida")
    st.write(f"Puntuación media de anomalía en parches: **{mean_score:.3f}** (0..1)")
    if mean_score > 0.35:
        st.error("🔴 La imagen muestra *evidencia significativa* de edición en zonas concretas.")
    elif mean_score > 0.18:
        st.warning("🟡 Hay indicios de edición en zonas puntuales — revisar ELA y la overlay heatmap.")
    else:
        st.success("🟢 No se detectan anomalías fuertes — la imagen parece consistente.")

    st.markdown("**Notas sobre la técnica:** este método es no supervisado y detecta *anomalías* en el espacio de características. Es capaz de señalar zonas donde el contenido no encaja con el resto (clonado, pegado, retocado), pero no es infalible: artefactos por compresión, ruido o técnicas de retoque sutiles pueden afectar el resultado.")

