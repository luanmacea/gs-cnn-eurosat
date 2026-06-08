"""
Treino reutilizável e persistência de histórico.

Mantém a lógica de compilação/treino fora dos notebooks para garantir que
CNN_A e CNN_B sejam treinadas exatamente com a mesma rotina (mudando apenas
os hiperparâmetros), o que torna a comparação justa e reprodutível.
"""

import json
import os

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


def compile_and_train(
    model,
    train_ds,
    val_ds,
    *,
    epochs=30,
    lr=1e-3,
    checkpoint_path="models/model.keras",
    patience=5,
    use_reduce_lr=False,
):
    """Compila e treina o modelo, salvando o melhor checkpoint.

    Args:
        epochs: nº máximo de épocas (EarlyStopping pode parar antes).
        lr: learning rate inicial do Adam.
        checkpoint_path: onde salvar o melhor modelo (por val_accuracy).
        patience: paciência do EarlyStopping (em épocas).
        use_reduce_lr: se True, reduz o lr ao estagnar a val_loss.

    Returns:
        history do Keras (objeto com .history).
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]
    if use_reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def save_history(history, path):
    """Salva o histórico de treino em JSON (para reuso na comparação)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hist = history.history if hasattr(history, "history") else history
    serializable = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_history(path):
    """Carrega um histórico de treino salvo em JSON."""
    with open(path) as f:
        return json.load(f)
