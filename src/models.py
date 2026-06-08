"""
Definição das duas arquiteturas de CNN treinadas do zero.

A comparação entre os dois modelos é o centro técnico do projeto. Eles foram
desenhados para contar uma história clara:

- CNN_A (Baseline): rede simples, sem nenhuma regularização. Serve para
  estabelecer o piso de desempenho e, deliberadamente, evidenciar overfitting.

- CNN_B (Refinada): rede mais profunda com data augmentation, BatchNorm,
  Dropout progressivo e GlobalAveragePooling. Cada uma dessas escolhas ataca
  diretamente uma limitação observada no baseline.

Ambos os modelos incluem uma camada Rescaling(1/255) como primeira operação,
de modo que recebem imagens uint8 [0,255] e normalizam internamente. Isso
elimina divergência entre o pré-processamento de treino e o de inferência.
"""

from tensorflow.keras import layers, models


def build_cnn_a(input_shape=(64, 64, 3), num_classes=10):
    """CNN baseline: 3 blocos Conv+Pool, sem regularização.

    Espera-se que esta rede atinja boa acurácia de treino mas mostre um gap
    relevante para a validação (overfitting), por não ter mecanismos de
    regularização nem aumento de dados.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="CNN_A_Baseline",
    )
    return model


def build_cnn_b(input_shape=(64, 64, 3), num_classes=10):
    """CNN refinada: mais profunda, com regularização e aumento de dados.

    Diferenças em relação à CNN_A e o porquê de cada uma:
      - Data augmentation (flip/rotation/zoom): expõe o modelo a variações e
        reduz overfitting. As camadas de augmentation só atuam no treino.
      - padding='same': mantém a dimensão espacial nas convoluções, permitindo
        empilhar mais blocos sobre imagens pequenas (64x64).
      - BatchNormalization: estabiliza o treino e acelera a convergência.
      - Dropout progressivo (0.25 -> 0.5): regularização crescente.
      - GlobalAveragePooling no lugar de Flatten: reduz drasticamente os
        parâmetros da camada densa final e atua como regularizador estrutural.
      - Dois Conv seguidos antes do pooling: aumenta o campo receptivo sem
        perda agressiva de resolução.
    """
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            data_augmentation,
            # Bloco 1
            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Bloco 2
            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Bloco 3
            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            # Classificador
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="CNN_B_Refined",
    )
    return model
