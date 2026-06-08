# Relatório Técnico — Classificação de Uso do Solo com CNNs (EuroSAT RGB)

**Global Solution — Visão Computacional aplicada à Indústria Espacial**

> Este relatório segue, seção a seção, os sete critérios de avaliação do
> enunciado. Os campos marcados com **[PREENCHER]** dependem dos resultados
> reais da execução — substitua-os pelos números que o notebook
> `04_avaliacao_comparativa.ipynb` produzir.

---

## 1. Definição do problema e conexão com a Indústria Espacial *(10 pts)*

A solução integrada da Global Solution propõe uma plataforma que centraliza
dados meteorológicos e de sensoriamento remoto que produtores rurais e analistas
já consomem de múltiplas fontes (diferentes satélites e provedores). Para que
esses dados gerem decisão, é necessário caracterizar **o que existe em cada
região observada**.

O problema de visão computacional resolvido aqui é, portanto, a **classificação
de uso e cobertura do solo (LULC)** a partir de recortes de imagens de satélite.
Dada uma imagem, o modelo determina sua classe (cultura anual, floresta, área
urbana, corpo d'água etc.). Essa classe é o atributo que conecta a imagem às
séries meteorológicas da mesma coordenada, dando contexto agronômico ao dado.

**Conexão com a Indústria Espacial:** toda a matéria-prima é proveniente de
**sensoriamento remoto orbital** — as imagens vêm do satélite Sentinel-2
(programa Copernicus). A CNN é o elo que transforma observação espacial em
informação acionável dentro da plataforma.

---

## 2. Dataset, preparação e pré-processamento *(15 pts)*

**Origem.** Dataset **EuroSAT (RGB)**, derivado de imagens multiespectrais do
Sentinel-2 e amplamente utilizado como benchmark de classificação de cobertura
do solo. Foi selecionado por estar diretamente alinhado ao tema (imagens de
satélite de áreas agrícolas, urbanas, vegetação e água) e por permitir
treinamento do zero com custo computacional viável.

**Composição.** 27.000 imagens RGB de 64×64 pixels, distribuídas em **10
classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake. O conjunto é levemente
desbalanceado (aproximadamente 2.000–3.000 imagens por classe — ver gráfico de
distribuição no notebook 01).

**Divisão.** Estratificada em **70% treino / 15% validação / 15% teste**
(`seed=42`), preservando a proporção de classes em cada partição
(`sklearn.train_test_split` com `stratify`). Os tamanhos resultantes e a
verificação de balanceamento por split estão no notebook 01.

**Pré-processamento.** As imagens são mantidas como `uint8 [0,255]` no pipeline
de dados; a **normalização (Rescaling 1/255) é a primeira camada de cada
modelo**. Essa decisão de arquitetura garante que treino e inferência apliquem
exatamente o mesmo pré-processamento, eliminando *train/serve skew* — a mesma
imagem enviada na demonstração (Streamlit) passa pela normalização idêntica à
do treino. O aumento de dados (augmentation) da CNN-B também vive no grafo do
modelo e atua apenas em treino.

---

## 3. Treinamento das CNNs do zero *(20 pts)*

Foram implementadas e treinadas **duas arquiteturas próprias**, sem qualquer
modelo pré-treinado. Ambas começam com `Rescaling(1/255)` e terminam em
`Dense(10, softmax)`. A rotina de treino é compartilhada (`src/train.py`),
mudando apenas hiperparâmetros, o que torna a comparação justa.

### CNN-A — Baseline

```
Rescaling(1/255)
Conv2D(32, 3x3, relu) → MaxPool
Conv2D(64, 3x3, relu) → MaxPool
Conv2D(128, 3x3, relu) → MaxPool
Flatten → Dense(128, relu) → Dense(10, softmax)
```

- Sem regularização e sem aumento de dados — proposital.
- Otimizador Adam (lr=1e-3), perda `sparse_categorical_crossentropy`.
- `EarlyStopping` (patience=5, restaura melhores pesos), até 30 épocas.
- **Parâmetros: ~684 mil** (a maior parte na densa após o `Flatten`).

### CNN-B — Refinada

```
Rescaling(1/255)
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
[Conv(32) → BN → ReLU] x2 → MaxPool → Dropout(0.25)
[Conv(64) → BN → ReLU] x2 → MaxPool → Dropout(0.25)
 Conv(128) → BN → ReLU     → MaxPool → Dropout(0.3)
GlobalAveragePooling → Dense(256, relu) → Dropout(0.5) → Dense(10, softmax)
```

- Otimizador Adam (lr=1e-3) com `ReduceLROnPlateau` (fator 0.5, patience=3).
- `EarlyStopping` (patience=10), até 50 épocas.
- **Parâmetros: ~176 mil** — *menos* que o baseline, graças ao
  `GlobalAveragePooling` no lugar do `Flatten`.

A evolução de acurácia e perda ao longo das épocas está registrada nas figuras
`cnn_a_curves.png` e `cnn_b_curves.png`.

---

## 4. Avaliação dos modelos *(parte dos pts de análise)*

A avaliação combina métricas quantitativas e análise qualitativa, sempre sobre o
**conjunto de teste** (nunca visto no treino).

- **Métricas quantitativas:** acurácia, precision, recall e F1 (por classe e
  macro), via `classification_report`.
- **Análise qualitativa:** grade de imagens mal classificadas pela CNN-B, com
  classe real, classe predita e confiança (`erros_cnn_b.png`).

**Resultados [PREENCHER com os valores reais]:**

- Acurácia de teste — CNN-A: **[PREENCHER]%**
- Acurácia de teste — CNN-B: **[PREENCHER]%**
- Classes com melhor F1: **[PREENCHER]**
- Classes com pior F1: **[PREENCHER]**

---

## 5. Comparação entre arquiteturas e análise técnica *(20 pts)*

| Modelo | Parâmetros | Acc. validação (melhor) | Acc. teste |
|---|---|---|---|
| CNN-A (Baseline) | ~684 k | **[PREENCHER]** | **[PREENCHER]** |
| CNN-B (Refinada) | ~176 k | **[PREENCHER]** | **[PREENCHER]** |

**Análise [ajustar conforme os resultados reais]:**

- **Generalização.** A CNN-A tende a apresentar um gap relevante entre treino e
  validação (overfitting), visível nas curvas. A CNN-B reduz esse gap — a
  combinação de **data augmentation**, **BatchNorm** e **Dropout** força o
  modelo a aprender padrões mais robustos em vez de memorizar o treino.
- **Efeito de cada técnica.**
  - *Augmentation* aumenta a diversidade efetiva do treino (rotações e flips são
    transformações válidas para imagens aéreas, que não têm orientação canônica).
  - *BatchNorm* estabiliza e acelera a convergência.
  - *Dropout progressivo* regulariza, com intensidade crescente em direção à
    saída.
  - *GlobalAveragePooling* reduz drasticamente os parâmetros e atua como
    regularizador estrutural — por isso a CNN-B é mais leve **e** generaliza
    melhor.
- **Matriz de confusão.** As confusões mais prováveis ocorrem entre classes de
  vegetação com textura/cor semelhantes — tipicamente *AnnualCrop*,
  *PermanentCrop*, *HerbaceousVegetation* e *Pasture*. **[Descrever o que a
  matriz real mostrou: quais pares se confundem mais.]**
- **Conclusão.** O melhor modelo é a **CNN-B**, que atinge maior acurácia de
  teste com menor número de parâmetros — melhor desempenho e melhor eficiência.

---

## 6. Demonstração funcional *(10 pts)*

Aplicação em **Streamlit** (`app/streamlit_app.py`) que carrega o melhor modelo
(`models/cnn_b_best.keras`) e classifica imagens novas enviadas pelo usuário,
exibindo a classe predita, a confiança e o gráfico de probabilidades das 10
classes. A interface contextualiza o resultado dentro da plataforma de dados
meteorológicos. Execução: `streamlit run app/streamlit_app.py`.

---

## 7. Documentação, organização e clareza *(10 pts)*

- Código organizado em `src/` (dados, modelos, treino, avaliação) e reutilizado
  pelos notebooks, garantindo reprodutibilidade e leitura limpa.
- Notebooks numerados na ordem de execução, com texto explicativo entre as
  etapas.
- Gráficos exportados em `reports/figures/`.
- `requirements.txt` com versões fixadas; divisão de dados com `seed` fixo.
- `README.md` com instruções de execução (Colab e local) e mapa do repositório.

---

## Limitações e melhorias futuras

- **EuroSAT RGB** descarta as bandas multiespectrais do Sentinel-2 (incluindo o
  infravermelho próximo, base do NDVI). Uma evolução natural é usar a versão
  multiespectral para capturar assinaturas espectrais da vegetação.
- Imagens de 64×64 limitam o detalhamento; recortes maiores ou super-resolução
  poderiam ajudar em classes de fronteira.
- **[Se a meta de 88% não for atingida:** discutir aqui as causas prováveis
  (capacidade do modelo, épocas, regularização) e os próximos ajustes.**]**
