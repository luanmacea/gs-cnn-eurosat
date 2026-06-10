# Relatório Técnico — Classificação de Uso do Solo com CNNs (EuroSAT RGB)

**Global Solution — Visão Computacional aplicada à Indústria Espacial**

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

**Obtenção dos dados.** O download é feito do arquivo oficial do EuroSAT no
**Zenodo**, com fallback automático para o mirror no **Hugging Face**. Optou-se
por essa abordagem porque o servidor legado usado por padrão pelo
`tensorflow-datasets` (DFKI) fica indisponível com frequência (HTTP 403).

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
[Conv(64)  → BN → ReLU] x2 → MaxPool → Dropout(0.25)
[Conv(128) → BN → ReLU] x2 → MaxPool → Dropout(0.30)
[Conv(256) → BN → ReLU] x2 → MaxPool → Dropout(0.40)
GlobalAveragePooling → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)
```

- Otimizador Adam (lr=1e-3) com `ReduceLROnPlateau` (fator 0.5, patience=3).
- `EarlyStopping` (patience=12, restaura melhores pesos), até 60 épocas.
- **Parâmetros: ~1,2 milhão.** A CNN-B troca capacidade extra (filtros mais
  largos: 64 → 128 → 256, com convoluções duplas) por poder de generalização.
  Uma versão mais estreita (~176 mil parâmetros) foi testada e estacionava em
  ~82% de acurácia — mesmo patamar do baseline —, evidenciando que a capacidade
  era o gargalo. O `GlobalAveragePooling` mantém a densa final enxuta apesar do
  aumento de filtros.

A evolução de acurácia e perda ao longo das épocas está registrada nas figuras
`cnn_a_curves.png` e `cnn_b_curves.png`.

---

## 4. Avaliação dos modelos

A avaliação combina métricas quantitativas e análise qualitativa, sempre sobre o
**conjunto de teste** (nunca visto no treino).

- **Métricas quantitativas:** acurácia, precision, recall e F1 (por classe e
  macro), via `classification_report`.
- **Análise qualitativa:** grade de imagens mal classificadas pela CNN-B, com
  classe real, classe predita e confiança (`erros_cnn_b.png`).

**Resultados:**

- Acurácia de teste — CNN-A: **82,47%**
- Acurácia de teste — CNN-B: **90,21%**
- **Desempenho por classe:** as classes visualmente mais distintas — corpos
  d'água (SeaLake), floresta (Forest) e áreas construídas (Residential,
  Industrial) — apresentam os maiores F1. As maiores quedas concentram-se nas
  classes de vegetação e cultivo (PermanentCrop, HerbaceousVegetation,
  AnnualCrop, Pasture), que compartilham textura e cor. Os valores exatos por
  classe estão no `classification_report` impresso no notebook 04.

---

## 5. Comparação entre arquiteturas e análise técnica *(20 pts)*

| Modelo | Parâmetros | Acc. validação (melhor) | Acc. teste |
|---|---|---|---|
| CNN-A (Baseline) | ~684 k | 82,88% | 82,47% |
| CNN-B (Refinada) | ~1,2 M | 90,70% | 90,21% |

**Análise:**

- **Generalização.** A CNN-A apresenta um gap relevante entre treino e validação
  (overfitting), visível nas curvas: a acurácia de treino sobe bem acima da de
  validação. A CNN-B reduz esse gap — treino e validação caminham próximos — graças
  à combinação de **data augmentation**, **BatchNorm** e **Dropout**, que força o
  modelo a aprender padrões mais robustos em vez de memorizar o treino. O ganho
  líquido foi de **+7,7 pontos** de acurácia no teste (82,47% → 90,21%).
- **Efeito de cada técnica.**
  - *Augmentation* aumenta a diversidade efetiva do treino (rotações e flips são
    transformações válidas para imagens aéreas, que não têm orientação canônica).
  - *BatchNorm* estabiliza e acelera a convergência, e permitiu treinar uma rede
    mais profunda sem instabilidade.
  - *Dropout progressivo* (0.25 → 0.5) regulariza, com intensidade crescente em
    direção à saída.
  - *GlobalAveragePooling* atua como regularizador estrutural e evita uma densa
    final gigante (como a do `Flatten` no baseline), mantendo o crescimento de
    parâmetros sob controle mesmo com filtros mais largos.
- **Capacidade vs. regularização.** Uma primeira versão da CNN-B, estreita
  (~176 mil parâmetros), regularizava bem mas estacionava em ~82% — mesmo nível
  do baseline —, indicando que o gargalo era a capacidade do modelo. Alargar os
  filtros (64/128/256) com a regularização mantida elevou o teto para ~90%.
- **Matriz de confusão.** Os erros se concentram entre as classes de vegetação e
  cultivo com textura e cor semelhantes — tipicamente entre *AnnualCrop*,
  *PermanentCrop*, *HerbaceousVegetation* e *Pasture* —, o que é esperado pela
  proximidade visual dessas coberturas. Classes bem distintas (SeaLake, Forest,
  Residential) quase não se confundem. A matriz completa está em
  `reports/figures/cm_cnn_b.png`.
- **Conclusão.** O melhor modelo é a **CNN-B**: a combinação de mais capacidade
  (filtros mais largos) com regularização (augmentation, BatchNorm e Dropout)
  entrega acurácia de teste superior e fecha o gap de generalização que afundava
  o baseline. O baseline, sem regularização, satura por overfitting; uma CNN-B
  estreita satura por falta de capacidade; a versão final equilibra os dois.

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
  multiespectral para capturar assinaturas espectrais da vegetação — justamente
  as classes que mais se confundem no RGB.
- Imagens de 64×64 limitam o detalhamento; recortes maiores ou super-resolução
  poderiam ajudar nas classes de fronteira (cultivos e vegetação).
- A acurácia de ~90% no teste atende à meta de referência (≥88%), mas há espaço
  para ganho com arquiteturas residuais (ResNet) treinadas do zero ou com mais
  épocas e *learning rate schedule* mais elaborado.