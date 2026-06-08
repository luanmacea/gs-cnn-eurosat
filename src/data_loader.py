"""
Carregamento e preparação do dataset EuroSAT RGB.

O EuroSAT RGB tem 27.000 imagens 64x64 (3 canais) do satélite Sentinel-2,
divididas em 10 classes de uso e cobertura do solo. O TensorFlow Datasets
expõe apenas o split 'train', então a divisão treino/validação/teste é feita
aqui, de forma estratificada, para manter o balanceamento entre as classes.

Decisão de arquitetura: a normalização (Rescaling 1/255) NÃO é feita aqui.
Ela vive como primeira camada dos modelos (ver models.py), garantindo que
treino e inferência usem exatamente o mesmo pré-processamento (sem train/serve
skew). Por isso este módulo entrega as imagens como uint8 [0, 255].
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

# Ordem oficial das classes no TFDS (eurosat/rgb)
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


def load_raw_data():
    """Baixa (1ª vez) e carrega o EuroSAT RGB inteiro em memória.

    Returns:
        images: np.ndarray (27000, 64, 64, 3) uint8
        labels: np.ndarray (27000,) int64
    """
    ds = tfds.load("eurosat/rgb", split="train", as_supervised=True, batch_size=-1)
    images, labels = tfds.as_numpy(ds)
    return images, labels.astype("int64")


def make_splits(images, labels, test_size=0.15, val_size=0.15, seed=42):
    """Divide os dados em treino/validação/teste de forma estratificada.

    Por padrão: 70% treino, 15% validação, 15% teste.

    Returns:
        dict com chaves 'train', 'val', 'test', cada uma um par (x, y).
    """
    # 1) separa o teste do restante
    x_temp, x_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    # 2) separa a validação do que sobrou (proporção relativa ao restante)
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
    """Converte os splits em tf.data.Dataset (batched + prefetch).

    Labels permanecem como inteiros -> usar sparse_categorical_crossentropy.
    Imagens permanecem uint8 [0,255] -> o modelo normaliza internamente.
    """

    def to_ds(x, y, training):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training and shuffle_train:
            ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = to_ds(*splits["train"], training=True)
    val_ds = to_ds(*splits["val"], training=False)
    test_ds = to_ds(*splits["test"], training=False)
    return train_ds, val_ds, test_ds
