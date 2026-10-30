# Classificação de cartas com CNN: naipes e comparação com MobileNetV2

Projeto de visão computacional que usa TensorFlow/Keras para classificar imagens de cartas de baralho. O dataset original tem **53 pastas/classes** — combinações como `ace of spades` e `king of hearts`, além de `joker` —, mas o notebook remapeia esses rótulos para dois problemas mais compactos:

1. **14 categorias de valor:** ace, 2–10, jack, queen, king e joker;
2. **5 categorias de símbolo:** hearts, diamonds, spades, clubs e joker.

O resultado final concentra-se no segundo problema: reconhecer o símbolo da carta com uma CNN customizada e compará-la com uma MobileNetV2 pré-treinada.

## Dataset e pipeline de rótulos

O `Cards Image Dataset` é carregado com `image_dataset_from_directory`, mantendo a divisão em diretórios:

```text
Cards Image csv/
├── train/
├── valid/
└── test/
```

Na execução registrada, o carregador encontrou:

| Divisão | Imagens | Classes originais |
|---|---:|---:|
| `train` | 7.624 | 53 |
| `valid` | 265 | 53 |
| `test` | 265 | 53 |

As classes originais são convertidas para índices de valor e de símbolo por `tf.lookup.StaticHashTable`. Portanto, várias pastas diferentes podem compartilhar o mesmo rótulo final de naipe. As imagens são normalizadas com `Rescaling(1./255)` e o pipeline usa `tf.data.AUTOTUNE` para prefetch.

## Etapas do experimento

### 1. Baselines

Uma CNN leve com duas camadas convolucionais (`16` e `32` filtros), pooling, uma camada densa de 64 neurônios e dropout de 0,3 é treinada por 10 épocas para os problemas de valores e símbolos. Essa etapa estabelece uma referência de custo e desempenho.

### 2. Tuning com Optuna

O `TPESampler` executa **15 trials** usando aproximadamente 15% do treino — 1.144 imagens — para acelerar a busca em CPU. O espaço de busca inclui:

| Hiperparâmetro | Espaço |
|---|---|
| `filters_1` | 16–64 |
| `filters_2` | 32–128 |
| `dense_units` | 64–256 |
| `dropout_rate` | 0,1–0,5 |
| `learning_rate` | 0,001–0,01, escala logarítmica |
| `batch_size` | 32 ou 64 |
| épocas máximas | 3–5 |

O melhor trial registrado encontrou 38 filtros na primeira convolução, 115 na segunda, 115 neurônios densos, dropout de aproximadamente 0,331, learning rate de aproximadamente 0,00384 e batch size 32.

### 3. Treinamento final

Depois do tuning, a rede customizada é reconstruída e treinada com o conjunto completo de treino/validação. A configuração efetivamente codificada no notebook é:

```text
entrada: 224 × 224 × 3
Conv2D: 38 filtros → MaxPooling2D
Conv2D: 115 filtros → MaxPooling2D
Flatten → Dense(115, relu)
Dropout(0,331)
Dense(5, softmax)
Adam com learning rate 0,0014136
até 5 épocas, EarlyStopping com paciência 2
```

O tuning usa imagens de 128×128 para ser mais barato; o treinamento definitivo usa 224×224. Essa diferença é intencional no notebook e deve ser considerada ao reproduzir os números.

### 4. Comparação com MobileNetV2

O projeto também adiciona uma MobileNetV2 com pesos da ImageNet, congela o extrator convolucional e treina apenas um novo classificador para as cinco classes. O download dos pesos exige conexão com a internet na primeira execução.

## Resultados registrados

| Modelo | Acurácia no teste | Loss no teste |
|---|---:|---:|
| CNN customizada otimizada | **95,85%** | **0,1478** |
| MobileNetV2 com transfer learning | **86,04%** | **0,3961** |

O notebook também produz curvas de treino/validação e uma matriz de confusão para investigar quais símbolos são mais confundidos.

## Como executar

Os diretórios de imagens e os arquivos `.h5` não são versionados por causa do tamanho. Baixe o Cards Image Dataset de uma fonte autorizada e coloque `train`, `valid` e `test` dentro de `Cards Image csv/`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install tensorflow optuna numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook CNN.ipynb
```

No Windows, a célula inicial que usa o comando Unix `unzip` pode não funcionar no `cmd`/PowerShell. Nesse caso, extraia o arquivo manualmente ou use `zipfile`/7-Zip e execute o notebook a partir da pasta que já contém `Cards Image csv/train`, `valid` e `test`.

## Estrutura esperada

```text
trabalhoCNN/
├── CNN.ipynb
├── Cards Image csv/
│   ├── cards.csv
│   ├── train/       # não versionado
│   ├── valid/       # não versionado
│   ├── test/        # não versionado
│   └── *.h5         # não versionados
└── README.md
```

## Limitações

- A busca de hiperparâmetros usa apenas uma fração do treino e poucas épocas; ela é uma aproximação para reduzir o custo computacional.
- O desempenho depende da divisão e da qualidade das imagens, e não foi avaliado em cartas de outro dataset.
- O remapeamento depende dos nomes das pastas (`hearts`, `diamonds`, `spades`, `clubs` e `joker`).
- Não há script de inferência ou exportação do modelo final; o fluxo principal está no notebook.
- Acurácia alta nesse conjunto não garante robustez a iluminação, rotação, cartas reais ou cenários fora da distribuição.
