"""
Avaliação e visualização de resultados.

Concentra tudo que alimenta o critério de "Comparação entre modelos e análise
técnica" (20 pts) e parte de "Avaliação dos modelos": predições, relatório de
classificação, matrizes de confusão, sobreposição de curvas e inspeção visual
de erros.

Todas as funções de plot aceitam um save_path opcional. Quando fornecido, a
figura é salva em reports/figures/ (com dpi adequado para o relatório) e também
exibida no notebook.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sns.set_theme(style="whitegrid")


def _ensure_dir(path):
    if path:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def get_predictions(model, ds):
    """Roda o modelo sobre um tf.data.Dataset e devolve rótulos e probabilidades.

    Returns:
        y_true: np.ndarray (N,)
        y_pred: np.ndarray (N,)  -- argmax das probabilidades
        y_proba: np.ndarray (N, num_classes)
    """
    y_true, y_proba = [], []
    for x, y in ds:
        y_proba.append(model.predict(x, verbose=0))
        y_true.append(np.asarray(y))
    y_true = np.concatenate(y_true)
    y_proba = np.concatenate(y_proba)
    y_pred = y_proba.argmax(axis=1)
    return y_true, y_pred, y_proba


def text_report(y_true, y_pred, class_names):
    """Relatório de classificação (precision/recall/f1 por classe) em texto."""
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )


def report_dataframe(y_true, y_pred, class_names):
    """Mesmo relatório, como DataFrame, útil para exibir tabela no notebook."""
    import pandas as pd

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    return pd.DataFrame(report).transpose()


def plot_training_curves(history, title="", save_path=None):
    """Curvas de loss e acurácia (treino vs. validação) de um único modelo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="treino")
    axes[0].plot(history["val_loss"], label="validação")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="treino")
    axes[1].plot(history["val_accuracy"], label="validação")
    axes[1].set_title(f"{title} — Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("acurácia")
    axes[1].legend()

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, class_names, title="", save_path=None, normalize=True
):
    """Matriz de confusão (normalizada por linha = recall por classe)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    norm_tag = "(normalizada) " if normalize else ""
    plt.title(f"Matriz de Confusão {norm_tag}— {title}")
    plt.ylabel("Classe real")
    plt.xlabel("Classe predita")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def compare_training_curves(hist_a, hist_b, labels=("CNN-A", "CNN-B"), save_path=None):
    """Sobrepõe as curvas de validação dos dois modelos para comparação direta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist_a["val_loss"], label=f"{labels[0]} val")
    axes[0].plot(hist_b["val_loss"], label=f"{labels[1]} val")
    axes[0].set_title("Loss de validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(hist_a["val_accuracy"], label=f"{labels[0]} val")
    axes[1].plot(hist_b["val_accuracy"], label=f"{labels[1]} val")
    axes[1].set_title("Acurácia de validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("acurácia")
    axes[1].legend()

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_misclassified(images, y_true, y_pred, y_proba, class_names, n=15, save_path=None):
    """Mostra imagens classificadas incorretamente (real vs. predito + confiança)."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("Nenhuma imagem classificada incorretamente no conjunto avaliado.")
        return

    selection = wrong[:n]
    cols = 5
    rows = int(np.ceil(len(selection) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for i, k in enumerate(selection):
        ax = axes[i]
        ax.imshow(images[k])
        ax.axis("off")
        confidence = y_proba[k, y_pred[k]]
        ax.set_title(
            f"real: {class_names[y_true[k]]}\n"
            f"pred: {class_names[y_pred[k]]} ({confidence:.2f})",
            fontsize=8,
            color="crimson",
        )

    fig.suptitle(f"Exemplos de erros (n={len(selection)})", y=1.0)
    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
