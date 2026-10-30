# Classificação de naipes com CNN

Projeto de visão computacional que desenvolve e otimiza uma rede neural convolucional para classificar imagens de cartas de baralho.

## Objetivo

Construir um classificador visual capaz de reconhecer cinco classes de naipes — Copas, Ouros, Espadas, Paus e Joker — usando imagens do Cards Image Dataset.

## Pipeline

1. Organização dos dados em treino, validação e teste.
2. Redimensionamento e normalização das imagens com `tf.data`.
3. Treinamento de uma CNN customizada com camadas `Conv2D` e `Dense`.
4. Busca de hiperparâmetros com Optuna e TPESampler.
5. Avaliação com acurácia, perda e matriz de confusão.
6. Comparação com MobileNetV2 usando transfer learning.

O notebook registra 95,85% de acurácia no teste para o modelo customizado otimizado. Esse resultado é experimental e depende da divisão e da qualidade do dataset.

## Tecnologias

- Python
- TensorFlow e Keras
- Optuna
- NumPy e Pandas
- scikit-learn
- Matplotlib e Seaborn

## Como executar

```bash
pip install tensorflow optuna numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook CNN.ipynb
```

O dataset de imagens e os modelos `.h5` não são versionados por causa do tamanho. Baixe o Cards Image Dataset de uma fonte autorizada e mantenha a estrutura `train`, `valid` e `test` dentro de `Cards Image csv/`.

## Observação

O notebook também contém uma etapa exploratória para números das cartas, mas o modelo final documentado concentra-se na classificação dos naipes.
