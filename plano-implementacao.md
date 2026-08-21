# Plano de Implementação — Desafio 1 enxuto

## Objetivo

Entregar um único notebook versionado na raiz, `desafio1.ipynb`, que implementa uma MLP em PyTorch para classificação de diabetes e regressão de preços de casas. A implementação deve ser curta, reproduzível, comentada em cada bloco e diagnosticar o treinamento de ponta a ponta.

## Tecnologias

Usar somente Python/Jupyter, PyTorch, TensorBoard, pandas, NumPy, scikit-learn e Matplotlib. Essas escolhas seguem o levantamento em `docs/tecnologias.md` e cobrem dados tabulares, MLP, métricas, visualização e observabilidade. Não usar PyTorch Lightning, Optuna, KaggleHub, Seaborn ou bibliotecas adicionais.

## Dados e preparação

| Problema | Arquivo | Alvo | Baseline |
|---|---|---|---|
| Classificação | `~/Downloads/diabetes_prediction/diabetes_prediction_dataset.csv` | `diabetes` | `LogisticRegression` |
| Regressão | `~/Downloads/house_price_regression/house_price_regression_dataset.csv` | `House_Price` | `LinearRegression` |

- Fixar a semente em `42` e usar GPU quando disponível.
- Separar os dados em 70% treino, 15% validação e 15% teste. Estratificar o Diabetes.
- Ajustar todos os transformadores somente no treino.
- Diabetes: `ColumnTransformer` com `StandardScaler` para atributos numéricos e `OneHotEncoder` para categóricos.
- House Price: `StandardScaler` para atributos e alvo; reverter a escala do alvo para MAE, RMSE e R².
- Converter dados transformados em `TensorDataset` e `DataLoader`: batch 256 para Diabetes e 64 para House Price.

## Modelo e treino

- Criar uma única classe `MLP(nn.Module)` com duas camadas ocultas `[32, 16]`.
- Cada bloco oculto segue `Linear → BatchNorm opcional → ativação → Dropout opcional`.
- Classificação: uma saída, `BCEWithLogitsLoss` com `pos_weight` calculado do treino e métricas de acurácia, precisão, recall e F1.
- Regressão: uma saída linear, `MSELoss` e métricas MAE, RMSE e R² em escala real.
- Usar AdamW, limite de 40 épocas e `ReduceLROnPlateau` com `factor=0.5`, `patience=3`, `threshold=1e-3` absoluto e `min_lr=1e-5`.
- Aplicar early stopping com paciência de 8 épocas e melhoria mínima de `1e-4`, preservando separadamente o checkpoint de qualquer melhor `val_loss`.
- Implementar loops explícitos de treino e validação, `model.train()`, `model.eval()` e `torch.no_grad()`.
- A cada época, registrar loss, métrica, learning rate após o passo do scheduler, normas média/máxima dos gradientes e contador do early stopping no TensorBoard.
- Reiniciar visualmente a sessão TensorBoard de cada experimento com `purge_step=0`, evitando misturar épocas de execuções anteriores.
- Salvar o melhor modelo por `val_loss` em `checkpoints/` com `state_dict`, configuração, época, métricas e threshold de classificação.

## Experimentos controlados

Executar os mesmos cinco experimentos para ambos os datasets, preservando split, seed, arquitetura e duração. Alterar somente o fator indicado.

| Nome | Alteração |
|---|---|
| `baseline` | ReLU, inicialização padrão, sem BatchNorm e sem Dropout. |
| `leaky_relu` | LeakyReLU no lugar de ReLU. |
| `kaiming` | Inicialização He/Kaiming no lugar da padrão. |
| `batchnorm` | BatchNorm no lugar de nenhuma normalização. |
| `dropout` | Dropout 0,2, mantendo BatchNorm. |

Selecionar o melhor experimento por F1 de validação no Diabetes e por RMSE de validação no House Price. O teste será consultado somente após essa seleção.

## Diagnóstico e entrega

- Exibir curvas de loss, métricas, gap treino-validação, learning rate e gradient norms; nomear o comportamento observado como convergência, underfitting, overfitting ou instabilidade.
- Relatar um problema realmente observado, a hipótese anterior, a correção e o efeito mensurado. O diagnóstico não será antecipado antes da execução.
- Escolher o threshold do Diabetes pela validação, persistir no checkpoint e aplicá-lo sem alteração no teste.
- Recarregar o melhor `state_dict`, reproduzir as previsões com `torch.allclose` e comparar métricas.
- Interpretar a matriz de confusão do Diabetes e mostrar gráfico real versus previsto e resíduos do House Price.
- Executar o notebook a partir de kernel limpo. Confirmar eventos TensorBoard, ausência de vazamento de dados e criação de checkpoints.

### Contexto histórico consultado

O arquivo local `docs/RELATORIO_FINAL.md` será usado somente como contexto, nunca como fonte para selecionar hiperparâmetros ou reportar métricas desta versão. Ele confirma que o split 70/15/15, o escalonamento ajustado apenas no treino e a normalização do alvo são essenciais. Também antecipa dois pontos a verificar nos novos resultados: o dataset de imóveis possui somente cerca de 1.000 linhas e uma regressão linear já é um baseline muito forte; portanto, uma MLP pode não superá-lo e esse resultado deve ser interpretado como evidência de variância/adequação do modelo, não escondido. A nova versão evita a grade extensa do relatório e mantém cinco comparações controladas por dataset.

## Versionamento

Versionar apenas os arquivos essenciais da raiz, incluindo `desafio1.ipynb` e este plano. A pasta `docs/` é histórico local e fica ignorada, assim como `runs/`, `checkpoints/` e arquivos de dados.
