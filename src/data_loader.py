"""
Carregamento e preparação do dataset EuroSAT RGB.

O EuroSAT RGB tem 27.000 imagens 64x64 (3 canais) do satélite Sentinel-2,
divididas em 10 classes de uso e cobertura do solo.

IMPORTANTE — sobre o download:
O `tensorflow-datasets` baixa o EuroSAT de um servidor da DFKI
(`madm.dfki.de`) que historicamente fica fora do ar ou retorna HTTP 403. Para
nao depender dele, este modulo busca o dataset de fontes estaveis, em ordem:

  1) Zip oficial no Zenodo (arquivo permanente);
  2) Mirror no Hugging Face Hub (`datasets.load_dataset`), ja pre-instalado
     no Google Colab — usado como fallback automatico.

O download acontece uma unica vez e fica em cache em `data/`.

Decisao de arquitetura: a normalizacao (Rescaling 1/255) NAO e feita aqui.
Ela vive como primeira camada dos modelos (ver models.py). Por isso este
modulo entrega as imagens como uint8 [0, 255].
"""

import os
import urllib.request
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

# Ordem canonica das classes (igual aos nomes das pastas no zip oficial).
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

IMG_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)

DATA_DIR = "data"

# Fontes do zip RGB, tentadas em ordem. Zenodo e o arquivo oficial permanente.
_ZIP_URLS = [
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip?download=1",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",  # legado (ultimo recurso)
]

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _normalize(name):
    """Normaliza nome de classe para casar variacoes de mirror.

    'Annual Crop'/'AnnualCrop' -> 'annualcrop'; 'Industrial Buildings' -> 'industrial'.
    """
    n = name.lower().replace(" ", "").replace("_", "")
    n = n.replace("buildings", "")
    return n


_CANON_LOOKUP = {_normalize(c): i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Estrategia 1: download do zip + leitura da estrutura de pastas
# ---------------------------------------------------------------------------
def _looks_like_class_root(path):
    if not os.path.isdir(path):
        return False
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    norm = {_normalize(d) for d in subdirs}
    return len(norm & set(_CANON_LOOKUP)) >= 5


def _find_class_root(base):
    if _looks_like_class_root(base):
        return base
    for dirpath, _dirnames, _files in os.walk(base):
        if _looks_like_class_root(dirpath):
            return dirpath
    raise FileNotFoundError(
        "Estrutura de classes do EuroSAT nao encontrada apos a extracao."
    )


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _load_from_zip(verbose=True):
    from PIL import Image

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "EuroSAT_RGB.zip")
    extract_dir = os.path.join(DATA_DIR, "eurosat_rgb_extracted")

    if not os.path.isdir(extract_dir):
        if not os.path.exists(zip_path):
            last_err = None
            for url in _ZIP_URLS:
                try:
                    if verbose:
                        print(f"Baixando EuroSAT RGB de: {url}")
                    _download(url, zip_path)
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if verbose:
                        print(f"  falhou ({type(e).__name__}: {e})")
            if last_err is not None:
                raise last_err
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    root = _find_class_root(extract_dir)
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )

    images, labels = [], []
    for cls in classes:
        label_idx = _CANON_LOOKUP.get(_normalize(cls))
        if label_idx is None:
            continue
        cls_dir = os.path.join(root, cls)
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(_IMG_EXTS):
                img = Image.open(os.path.join(cls_dir, fn)).convert("RGB")
                if img.size != (IMG_SIZE, IMG_SIZE):
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.asarray(img, dtype="uint8"))
                labels.append(label_idx)

    if not images:
        raise RuntimeError("Nenhuma imagem lida da estrutura de pastas do EuroSAT.")
    return np.stack(images), np.asarray(labels, dtype="int64")


# ---------------------------------------------------------------------------
# Estrategia 2: Hugging Face Datasets (fallback — pre-instalado no Colab)
# ---------------------------------------------------------------------------
def _load_from_huggingface(verbose=True):
    from datasets import load_dataset

    if verbose:
        print("Baixando EuroSAT RGB via Hugging Face (blanchon/EuroSAT_RGB)...")
    ds = load_dataset("blanchon/EuroSAT_RGB", split="train")

    label_names = ds.features["label"].names
    remap = {idx: _CANON_LOOKUP.get(_normalize(nm), idx)
             for idx, nm in enumerate(label_names)}

    images = np.stack(
        [
            np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="uint8")
            for img in ds["image"]
        ]
    )
    labels = np.asarray([remap[l] for l in ds["label"]], dtype="int64")
    return images, labels


def load_raw_data(verbose=True):
    """Carrega o EuroSAT RGB inteiro em memoria, tentando fontes estaveis.

    Returns:
        images: np.ndarray (~27000, 64, 64, 3) uint8
        labels: np.ndarray (~27000,) int64  -- indices conforme CLASS_NAMES
    """
    errors = []
    for loader in (_load_from_zip, _load_from_huggingface):
        try:
            images, labels = loader(verbose=verbose)
            if verbose:
                print(f"Dataset carregado: {images.shape[0]} imagens.")
            return images, labels
        except Exception as e:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {type(e).__name__}: {e}")
            if verbose:
                print(f"Fonte indisponivel -> {errors[-1]}")

    raise RuntimeError(
        "Nao foi possivel obter o EuroSAT RGB por nenhuma fonte.\n"
        + "\n".join(errors)
        + "\n\nNo Colab, garanta acesso a internet. Em ultimo caso, instale o "
        "Hugging Face Datasets com:  !pip install -q datasets"
    )


# ---------------------------------------------------------------------------
# Divisao e pipelines tf.data (inalterado)
# ---------------------------------------------------------------------------
def make_splits(images, labels, test_size=0.15, val_size=0.15, seed=42):
    """Divide os dados em treino/validacao/teste de forma estratificada (70/15/15)."""
    x_temp, x_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_relative = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=seed
    )
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def make_tf_datasets(splits, batch_size=64, shuffle_train=True, seed=42):
    """Converte os splits em tf.data.Dataset (batched + prefetch)."""
    import tensorflow as tf

    def to_ds(x, y, training):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training and shuffle_train:
            ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return (
        to_ds(*splits["train"], training=True),
        to_ds(*splits["val"], training=False),
        to_ds(*splits["test"], training=False),
    )