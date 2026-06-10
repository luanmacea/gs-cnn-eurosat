# 🛰️ Classificação de Uso e Cobertura do Solo com CNNs (EuroSAT RGB)

**Global Solution — Visão Computacional aplicada à Indústria Espacial**

Projeto de classificação de imagens de satélite usando **redes neurais
convolucionais treinadas do zero** (sem modelos pré-treinados). O sistema
classifica imagens do satélite **Sentinel-2** em 10 categorias de uso e
cobertura do solo (LULC) e funciona como o componente de visão computacional de
uma plataforma que agrega, em um único lugar, dados meteorológicos e de
sensoriamento remoto que produtores rurais e analistas já consomem de
diferentes fontes.

---

## 👥 Integrantes

* Davi Passanha de Sousa Guerra - RM551605
* Cauã Gonçalves de Jesus - RM97648
* Luan Silveira Macea - RM98290
* Rui Amorim Siqueira - RM98436
* Luigi Ferrara Sinno - RM98047

## 🎥 Vídeo de apresentação

> **PREENCHER** — link do YouTube (até 3 min): `https://youtu.be/...`

---

## 🎯 O problema e a conexão com a Indústria Espacial

Produtores e analistas que dependem de dados meteorológicos hoje precisam
consultar várias plataformas e satélites separadamente. A plataforma proposta
centraliza essas fontes. Para que a análise seja útil, é preciso saber **o que
existe em cada região observada** — e é aí que entra a visão computacional.

A CNN recebe um recorte de imagem de satélite e classifica seu uso do solo
(cultura anual, floresta, área urbana, água etc.). Essa informação é então
cruzada com as séries meteorológicas da mesma coordenada, dando contexto
agronômico aos dados. A conexão com a **Indústria Espacial** é direta: toda a
matéria-prima vem de **sensoriamento remoto orbital** (Sentinel-2), e a CNN é o
elo que transforma pixels de satélite em informação acionável.

**Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

---

## 🗂️ Estrutura do repositório

```
gs-cnn-eurosat/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda_dataset.ipynb          # exploração e divisão do dataset
│   ├── 02_treino_cnn_a.ipynb         # treino do baseline
│   ├── 03_treino_cnn_b.ipynb         # treino do modelo refinado
│   └── 04_avaliacao_comparativa.ipynb# métricas, matrizes e análise de erros
├── src/
│   ├── data_loader.py                # download, split estratificado, tf.data
│   ├── models.py                     # arquiteturas CNN_A e CNN_B
│   ├── train.py                      # rotina de treino + histórico
│   └── evaluate.py                   # predições, relatórios e gráficos
├── app/
│   └── streamlit_app.py              # demonstração funcional
├── models/                           # pesos salvos (cnn_b_best.keras)
└── reports/
    ├── relatorio_tecnico.md          # relatório técnico completo
    └── figures/                      # gráficos exportados
```

---

## ⚙️ Como executar

O fluxo recomendado é o **Google Colab com GPU** (treino rápido e sem
configuração local). Também funciona localmente.

### Opção A — Google Colab (recomendado)

1. Suba este repositório para o seu GitHub.
2. Abra cada notebook no Colab (`Arquivo → Abrir notebook → GitHub`, ou
   `https://colab.research.google.com/github/luanmacea/gs-cnn-eurosat`).
3. Ative a GPU: **Ambiente de execução → Alterar tipo de ambiente → T4 GPU**.
4. Na **primeira célula** de cada notebook, descomente o bloco do Colab para
   clonar o repositório:
   ```python
   !git clone https://github.com/luanmacea/gs-cnn-eurosat.git
   %cd gs-cnn-eurosat
   ```
5. Rode os notebooks **na ordem** (01 → 02 → 03 → 04).

> ⚠️ **NÃO rode `pip install -r requirements.txt` no Colab.** O Colab já traz
> TensorFlow, Keras, numpy, scikit-learn, matplotlib, seaborn e
> tensorflow-datasets em versões compatíveis. Reinstalar as versões fixas
> rebaixa o `numpy`/`ml_dtypes` e quebra o JAX do Colab (erro *"JAX requires
> ml_dtypes version 0.5 or newer"*). O `requirements.txt` é só para uso local.
>
> Se reiniciar o runtime no meio do caminho, rode novamente a célula de clone
> (`git clone` + `%cd`) antes das demais — a célula de bootstrap já lida com a
> detecção da raiz do projeto automaticamente.

### Opção B — Local

```bash
# Python 3.10–3.12
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                 # abra os notebooks da pasta notebooks/
```

> Sem GPU o treino é mais lento (dezenas de minutos por modelo), mas funciona.

---

## ▶️ Ordem de execução e o que cada etapa gera

Execute os notebooks nesta ordem. As saídas (pesos e gráficos) são geradas
automaticamente.

| Ordem | Notebook | O que faz | Gera |
|------|----------|-----------|------|
| 1 | `01_eda_dataset.ipynb` | Baixa o EuroSAT, explora classes e amostras, define a divisão 70/15/15 | (visualizações) |
| 2 | `02_treino_cnn_a.ipynb` | Treina o baseline (~5 min na GPU) | `models/cnn_a_best.keras`, `reports/cnn_a_history.json`, `reports/figures/cnn_a_curves.png` |
| 3 | `03_treino_cnn_b.ipynb` | Treina o modelo refinado (~8–10 min na GPU) | `models/cnn_b_best.keras`, `reports/cnn_b_history.json`, `reports/figures/cnn_b_curves.png` |
| 4 | `04_avaliacao_comparativa.ipynb` | Compara os dois modelos | matrizes de confusão, curvas comparadas e grade de erros em `reports/figures/` |

> O download do dataset (~90 MB) acontece **uma vez**, na primeira execução do
> notebook 01, via `tensorflow-datasets`.

---

## 🖥️ Demonstração funcional (Streamlit)

Após treinar a CNN-B (gerando `models/cnn_b_best.keras`):

```bash
pip install -r requirements.txt          # se ainda não instalou
streamlit run app/streamlit_app.py
```

A aplicação abre no navegador (`http://localhost:8501`). Faça upload de uma
imagem de satélite — pode usar uma amostra do próprio conjunto de teste do
EuroSAT — e veja a classe predita, a confiança e a distribuição de
probabilidades entre as 10 classes.

---

## 🧠 As duas arquiteturas (resumo)

| | **CNN-A — Baseline** | **CNN-B — Refinada** |
|---|---|---|
| Blocos convolucionais | 3 (Conv → Pool) | 3 blocos com Conv duplo + Pool |
| Regularização | nenhuma | BatchNorm + Dropout (0.25 → 0.5) |
| Aumento de dados | não | flip + rotação + zoom |
| Camada final | `Flatten → Dense(128)` | `GlobalAveragePooling → Dense(256)` |
| Objetivo | estabelecer o piso, mostrar overfitting | generalizar e atingir ≥ 88% |

A comparação detalhada e a justificativa técnica de cada decisão estão em
[`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md).

---

## 📊 Resultados

> **PREENCHER** com os valores reais após rodar os notebooks 02–04.

| Modelo | Parâmetros | Acc. validação | Acc. teste |
|---|---|---|---|
| CNN-A (Baseline) | ~684 k | _preencher_ | 82,14% |
| CNN-B (Refinada) | ~176 k | _preencher_ | 91,28% |

Meta de referência: **acurácia ≥ 88%** no conjunto de teste — **atingida pela CNN-B (91,28%)**.

---

## 🔁 Reprodutibilidade

- Divisão treino/validação/teste fixada com `seed=42`.
- Versões de bibliotecas fixadas em `requirements.txt`.
- Lógica de dados, modelos, treino e avaliação isolada em `src/` e reutilizada
  pelos notebooks, evitando divergência entre experimentos.

## 🛠️ Stack

Python · TensorFlow/Keras · tensorflow-datasets · scikit-learn · NumPy ·
Pandas · Matplotlib · Seaborn · Streamlit.
