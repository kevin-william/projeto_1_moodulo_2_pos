# Plano de implementação — Desafio 1 enxuto

## Objetivo

Manter um único notebook executável de cima para baixo para classificação de Diabetes e regressão de preços de imóveis. A solução usa PyTorch sem abstrações de alto nível, aplica pré-processamento somente no treino, compara experimentos controlados e avalia o teste apenas após a seleção pela validação.

## Convenções obrigatórias de código

- Todo identificador criado no projeto é escrito em português, por extenso e com significado de domínio.
- APIs externas preservam seus nomes oficiais, como `forward`, `state_dict`, `DataLoader`, `batch_size` e `num_workers`.
- Constantes de configuração ficam em português e em letras maiúsculas; números relevantes não permanecem espalhados pelo código.
- Cada célula tem uma responsabilidade principal e contém comentários curtos explicando o que faz e como faz.
- Treino, validação, métricas, checkpoint, registro e controle de parada são funções separadas e coesas.
- O notebook usa semente 42, `num_workers=0` e caminhos construídos com `pathlib.Path`.

## Pipeline dos dados

1. Ler os dois arquivos CSV existentes em `Downloads`.
2. Remover registros duplicados e ausentes antes da divisão.
3. Separar treino, validação e teste na proporção 70/15/15; a classificação usa estratificação.
4. Ajustar `StandardScaler`, `OneHotEncoder` e `ColumnTransformer` exclusivamente no treino.
5. Criar `TensorDataset` e `DataLoader` com lote 256 para Diabetes e 64 para imóveis.
6. Comparar com `LogisticRegression` e `LinearRegression` como referências não triviais.

## Modelo e experimentos

- Uma única classe `PerceptronMulticamadas` define explicitamente camadas, ativações e `forward`.
- Arquitetura fixa `[32, 16]`, apropriada ao objetivo didático e aos dados tabulares.
- Classificação: `BCEWithLogitsLoss`, peso da classe positiva calculado no treino e métricas de acurácia, precisão, sensibilidade e média harmônica.
- Regressão: `MSELoss` e métricas MAE, RMSE e R² na escala original.
- Cinco configurações controladas por problema: referência, LeakyReLU, Kaiming/He, BatchNorm e Dropout 0,2.
- Seleção pela validação: maior F1 para Diabetes e menor RMSE para imóveis.

## Treinamento e diagnóstico

- AdamW com taxa inicial `1e-3`.
- `ReduceLROnPlateau` com fator 0,5, paciência 3, limiar absoluto `1e-3` e taxa mínima `1e-5`.
- Até 40 épocas com parada antecipada de paciência 8 e melhoria mínima `1e-4`.
- Checkpoint sempre baseado na menor perda de validação, independentemente do limiar da parada antecipada.
- Registro por época de perdas, métricas, taxa de aprendizado posterior ao scheduler e normas média e máxima dos gradientes.
- TensorBoard em `runs/` e checkpoints em `checkpoints/`; ambas as pastas permanecem no `.gitignore`.

## Resultado final

- Escolher o limiar de classificação exclusivamente na validação e salvá-lo no checkpoint.
- Recarregar os melhores checkpoints e confirmar a reprodução das previsões.
- Avaliar o teste uma única vez e apresentar matriz de confusão, reais versus previstos e resíduos.
- Nomear o fenômeno observado nas curvas e relatar ao menos um problema real com sintoma, hipótese anterior, correção e efeito medido.
- Usar o relatório histórico como evidência: manter a regressão linear como referência forte para o pequeno conjunto de imóveis e evitar transferir hiperparâmetros antigos sem nova validação controlada.

## Verificação

- Compilar todas as células.
- Executar o notebook do kernel limpo, do início ao fim.
- Confirmar ajuste dos transformadores somente no treino.
- Confirmar CUDA quando disponível e funcionamento em CPU como alternativa.
- Confirmar redução de taxa, parada antecipada, checkpoints e reprodução de previsões.
- Versionar somente os arquivos essenciais; ignorar `.venv/`, `docs/`, `runs/`, `checkpoints/`, dados e caches.
