# AGENTS — Guia do Projeto MLP

Este documento é a fonte de verdade para qualquer agente (humano ou IA) que
vá trabalhar neste repositório. Todas as convenções aqui são **obrigatórias**
salvo indicação contrária.

---

## 1. Visão Geral

Prova de conceito de Deep Learning implementando um **Perceptron
Multicamadas** aplicado a dois problemas tabulares:

- **Classificação binária** — Diabetes.
- **Regressão** — Preço de imóveis (House Price), avaliada com **MSE**, **MAE**,
  **RMSE** e **R²** (coeficiente de determinação).

Foco em **estabilidade do treinamento** (não apenas em métricas finais):
diagnóstico de gradientes, comparação entre técnicas de normalização, efeito
de regularização e monitoramento em TensorBoard.

Toda a lógica de código fica exclusivamente no notebook
`projeto-1-modulo2.ipynb`. O plano detalhado está em
`plano-implementacao.md`.

---

## 2. Stack

| Camada | Ferramenta |
|---|---|
| Linguagem | Python 3.11+ |
| Framework de DL | PyTorch |
| Dados tabulares | pandas, numpy |
| Pré-processamento / métricas | scikit-learn |
| Visualização de treino | TensorBoard (`torch.utils.tensorboard`) |
| Plots no notebook | matplotlib |
| Notebook | Jupyter (`.ipynb`, nbformat 4) |

Plataforma alvo: **Windows** (cuidado com `num_workers > 0` no `DataLoader`).

---

## 3. Estrutura do Repositório

```
.
├── AGENTS.md                  # este arquivo
├── plano-implementacao.md     # plano detalhado (referência)
├── projeto-1-modulo2.ipynb    # ÚNICO arquivo com lógica de código
└── .gitignore
```

Pastas **não versionadas** (criadas em runtime):

- `runs/` — logs do TensorBoard (um subdiretório por experimento).
- `*.pt`, `*.pth` — checkpoints locais.

Os datasets **não vivem no repositório**. São lidos dos caminhos absolutos:

- `C:\Users\lord_\Downloads\diabetes_prediction\diabetes_prediction_dataset.csv`
- `C:\Users\lord_\Downloads\house_price_regression\house_price_regression_dataset.csv`

---

## 4. Convenção de Nomes (Código)

**Regra absoluta:** tudo o que for escrito por nós — variáveis, funções,
classes, métodos, parâmetros, constantes, módulos — deve estar em
**português**, **sem siglas** e **sem abreviações**. Os nomes precisam ser
**cheios de significado** e **contribuir para o entendimento do fluxo** do
programa.

### 4.1. Idioma e abreviações

Tudo em português, sem exceção para código próprio. Exemplos canônicos:

| Em vez de | Escrever |
|---|---|
| `lr` | `taxa_aprendizado` |
| `model` | `modelo` |
| `train_loader` | `carregador_treino` |
| `val_loader` | `carregador_validacao` |
| `test_loader` | `carregador_teste` |
| `X`, `y` | `caracteristicas`, `rotulos` |
| `batch_size` | `tamanho_lote` |
| `epochs` | `epocas` |
| `criterion` | `criterio` |
| `optimizer` | `otimizador` |
| `scheduler` | `agendador_taxa_aprendizado` |
| `loss` | `perda` |
| `acc` | `acuracia` |
| `MLP` | `PerceptronMulticamadas` |
| `state_dict` (variável nossa) | `dicionario_estado` |
| `pred` | `previsao` |
| `correct` | `acertos` |
| `dropout` (variável) | `taxa_descarte` |
| `BN` / `LN` | `normalizacao_lote` / `normalizacao_camada` |
| `MSE` (variável) | `erro_quadratico_medio` |
| `MAE` (variável) | `erro_absoluto_medio` |
| `RMSE` (variável) | `raiz_erro_quadratico_medio` |
| `R²` (variável) | `coeficiente_determinacao` |
| `num_workers` (variável) | `numero_processos` |
| `drop_last` (variável) | `descartar_ultimo_lote` |
| `device` | `dispositivo` |
| `xb`, `yb` | `lote_caracteristicas`, `lote_rotulos` |

### 4.2. Semântica dos nomes

Cada identificador deve responder sozinho às perguntas **"o que é isto?"** ou
**"o que isto faz?"**. Evitar nomes genéricos que sirvam para qualquer coisa.

- ❌ Evitar: `dados`, `info`, `aux`, `temp`, `resultado`, `obj`, `item`,
  `coisa`, `x`, `y`, `a`, `b`.
- ✅ Preferir nomes que comuniquem **papel/responsabilidade**:
  `carregador_treino` > `treino_loader` > `dl` > `loader`.
- Coleções: usar **plural** ou sufixo do conteúdo —
  `epocas`, `historico_epocas`, `resultados_validacao`.
- Verbos para funções/métodos quando houver ação clara:
  `executar_epoca_treino`, `calcular_gradientes`, `salvar_checkpoint`.
- Substantivos para funções que retornam um valor bem definido:
  `inicializar_historico`, `obter_perda_media`.

Se um nome precisa de **comentário ao lado** para ser entendido, é sinal de
que o nome deveria ser mais explícito.

### 4.3. Métodos pequenos e coesos (responsabilidade única)

Cada função/método deve ter **uma responsabilidade clara e verificável**:

- Uma função que cabe na cabeça sem reler é uma função boa.
- Trecho acima de ~30 linhas misturando "treinar", "avaliar", "logar" e
  "imprimir" é sinal para **quebrar em helpers nomeados**.
- Helpers usados por uma única função podem ser co-localizados; helpers
  reutilizáveis ganham módulo próprio (no nosso caso, ficam na Célula 3 de
  Utils).
- O **nome** do método deve descrever a ação: `executar_epoca_treino`,
  `executar_epoca_validacao`, `registrar_epoca_no_historico`,
  `registrar_log_do_terminal`.

### 4.4. Números mágicos viram constantes nomeadas

Toda constante numérica com **significado** deve virar constante nomeada em
**português**, no topo do bloco onde é usada (ou na célula de Utils se for
compartilhada):

```python
PROPORCAO_TESTE = 0.15
PROPORCAO_VALIDACAO = 0.15
EPOCAS_PADRAO = 50
INTERVALO_LOG = 10
LIMIAR_DECISAO_BINARIA = 0.5
PACIENCIA_PARADA_ANTECIPADA = 5
```

Constantes vêm **antes** dos blocos que as usam e em `UPPER_SNAKE` em
português (sem `MAX_`, `MIN_`, `DEFAULT_` em inglês — usar
`EPOCAS_PADRAO`, `TAXA_ENTRADA_MAXIMA` etc.).

### 4.5. Simplicidade e comentários como fallback

Regra de ouro: **o código mais simples que funciona vence**.

- Evitar one-liners espertos sem necessidade.
- Evitar abstrações prematuras (classe base genérica para "qualquer
  treinamento").
- Quando uma construção for **inevitavelmente complexa** (ex.: split
  estratificado + scaler sem vazamento + generator do DataLoader), usar
  **comentários em português explicando o porquê**, em vez de tentar
  comprimir tudo em uma linha.
- Preferir **código legível** a **código curto**. Linha densa que precisa
  ser decifrada é dívida técnica.

### 4.6. Exceções (APIs externas)

Identificadores que pertencem a **bibliotecas externas** permanecem como
na API original, sem tradução:

- PyTorch: `nn.Linear`, `nn.Dropout`, `model.state_dict()`, `forward()`,
  `optimizer.step()`, `DataLoader`, `TensorDataset`, `BatchNorm1d`,
  `LayerNorm`, `ReLU`, `MSELoss`, `BCEWithLogitsLoss`.
- scikit-learn: `StandardScaler`, `train_test_split`, `Pipeline`,
  `LabelEncoder`.
- TensorBoard: `SummaryWriter`, `add_scalar`, `add_histogram`.

Em comentários e docstrings, pode-se explicar o equivalente em português,
mas **nunca renomear** uma chamada de API.

---

## 5. Convenções do Notebook

- **Uma célula por responsabilidade** (não empilhar imports + treino + plot).
- Comentários sucintos em português explicando o **porquê** de blocos não
  triviais; o **o quê** deve estar claro pelo próprio código.
- Markdown explicativo entre seções lógicas.
- Reprocessamento: o notebook deve rodar de ponta a ponta sem erro, do início
  ao fim, sem estados ocultos entre células.
- **Reprodutibilidade**: seeds devem ser fixadas no início
  (`torch.manual_seed`, `numpy.random.seed`, `random.seed`,
  `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `DataLoader(generator=...)`).
- **Sem vazamento**: qualquer `fit` (scaler, encoder, normalizador) acontece
  **somente** no conjunto de treino. Validação e teste só recebem `transform`.
- **Split**: sempre `train_test_split` com `stratify=rotulos` em classificação.
- **Windows-safe**: `numero_processos=0` por padrão no `DataLoader`.

---

## 6. Política de Commits

### 6.1. Atomicidade

Cada commit representa **uma única mudança lógica coerente**:

- ✅ `adicionar: célula de preprocessamento do dataset de diabetes`
- ❌ `adicionar: preprocess + treino + plot + checkpoint + tensorboard`

Se a mudança envolve múltiplos arquivos por natureza (ex.: `.gitignore` +
`AGENTS.md` no bootstrap), agrupar com mensagem clara: `bootstrap:
configurar repositório`.

### 6.2. Mensagens

Formato recomendado (Conventional Commits adaptado):

```
<tipo>: <descrição curta em português, imperativo, ≤ 72 caracteres>

<opcional: corpo explicando o porquê, sem repetir o quê>
```

Tipos comuns:

| Tipo | Uso |
|---|---|
| `bootstrap` | Configuração inicial do repositório |
| `adicionar` | Novo conteúdo (célula, seção, arquivo) |
| `alterar` | Mudança em conteúdo existente |
| `corrigir` | Correção de bug ou erro |
| `documentar` | Apenas markdown / docs |
| `refatorar` | Reorganização sem mudar comportamento |
| `reverter` | Rollback |

Exemplos válidos:

- `bootstrap: configurar repositório (gitignore e AGENTS)`
- `documentar: adicionar plano de implementação`
- `adicionar: célula com função de treino genérica`
- `corrigir: evitar vazamento no StandardScaler (fit só no treino)`

### 6.3. Verificação pré-commit

Antes de cada commit:

1. `git status` — revisar o que vai entrar.
2. `git diff` — reler as alterações linha a linha.
3. Garantir que **nenhum dataset**, **checkpoint** ou **log do TensorBoard**
   está sendo versionado (estão no `.gitignore`).
4. Se houver mudanças em código do notebook, rodar as células afetadas.

---

## 7. Como Executar

```powershell
# Ativar ambiente virtual (recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio pandas scikit-learn matplotlib tensorboard jupyter

# Abrir o notebook
jupyter notebook projeto-1-modulo2.ipynb

# Em outro terminal, abrir o TensorBoard (após rodar pelo menos um experimento)
tensorboard --logdir runs
```

---

## 8. Como Continuar

Iterativo. Para cada bloco do plano:

1. O usuário descreve o **comportamento** desejado.
2. O agente implementa a célula correspondente do notebook.
3. Juntos validamos antes de seguir para o próximo bloco.

Mudanças no `AGENTS.md` ou `plano-implementacao.md` também passam por commit
atômico documentado.
