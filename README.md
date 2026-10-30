# TCC — Experimentos de classificação facial com TensorFlow

Notebook experimental de classificação de imagens de rostos com redes neurais convolucionais, incluindo uma arquitetura inspirada na **LeNet** em TensorFlow/Keras.

> Este repositório preserva experimentos de um TCC, não uma aplicação pronta para produção. O notebook reúne abordagens diferentes e não está validado para execução integral, de cima para baixo.

## Conteúdo

- [`TCC_TensorFlow.ipynb`](TCC_TensorFlow.ipynb): código, comentários e experimentos, em versão higienizada para publicação.
- `.gitignore`: exclusões preventivas para dados locais, modelos e credenciais.

O conjunto de imagens e os modelos treinados **não estão incluídos**. As saídas incorporadas ao arquivo original foram removidas; este README não apresenta métricas como resultados reproduzidos.

## Experimento principal

O bloco que começa com `LARGURA, ALTURA = 500, 500` implementa:

1. Leitura de arquivos JPG, JPEG e PNG em pastas separadas de treino, teste e validação.
2. Conversão BGR → RGB, redimensionamento para 500 × 500 e normalização em `[0, 1]`.
3. Codificação de classes com `LabelEncoder`, ajustado somente no treino, seguida de one-hot encoding.
4. Rede com convoluções de 6 e 16 filtros 5 × 5, average pooling 2 × 2, flatten, camadas densas de 120 e 84 unidades, dropout de 0,3 e saída softmax.
5. Treinamento com Adam, `categorical_crossentropy`, lote de 8 imagens e 15 épocas; checkpoint por menor perda de validação.
6. Predição no teste, relatório de classificação e matriz de confusão.

O experimento foi escrito para seis classes; nesse bloco, a quantidade de saídas é calculada a partir das classes encontradas no treino. Há também carregamento em 250 × 250, outras CNNs e tentativas de exportação de modelos.

## Dados locais

Use apenas imagens para as quais você tenha autorização de uso. Os nomes abaixo são identificadores genéricos, não dados de participantes:

```text
data/
└── faces/
    ├── treino/
    │   ├── pessoa_01/
    │   └── pessoa_02/
    ├── teste/
    │   ├── pessoa_01/
    │   └── pessoa_02/
    └── validacao/
        ├── pessoa_01/
        └── pessoa_02/
models/
```

Repita a estrutura para todas as classes. As classes de teste e validação precisam existir no treino. Separe os conjuntos com cuidado para evitar que imagens duplicadas ou muito semelhantes contaminem a avaliação.

## Uso no Google Colab

1. Abra o arquivo `.ipynb` no Colab pela opção de abrir um notebook do GitHub ou enviando o arquivo.
2. Disponibilize seu conjunto autorizado **no armazenamento local da sessão**. Não há montagem automática nem conexão com Google Drive nesta versão.
3. Na célula de configuração, ajuste `DATASET_ROOT` (padrão `./data/faces`) e `MODELS_ROOT` (padrão `./models`). O armazenamento temporário do Colab pode ser perdido ao encerrar a sessão.
4. Confira as dependências do ambiente. O bloco principal usa TensorFlow (`tensorflow.keras`), NumPy, OpenCV (`cv2`), Matplotlib, Seaborn e scikit-learn. Outros experimentos importam pandas, Pillow, requests, Keras, `tf_keras` e TensorFlow.js.
5. Para explorar o bloco principal, execute a configuração local e depois a célula completa de treinamento em 500 × 500. Ela contém suas próprias importações. Leia o código antes de executar; o treinamento pode exigir bastante memória e se beneficiar de GPU.
6. Mantenha `SHOW_PRIVATE_IMAGES = False` para não exibir fotografias nas células de visualização. Só habilite a opção em ambiente privado e com autorização.

**Não use “Executar tudo” como procedimento garantido.** Não há versões de dependências fixadas nem ambiente reproduzível validado. As instruções acima orientam a exploração do código, não representam um teste de execução bem-sucedido.

## Limitações conhecidas

- Células antigas usam variáveis diferentes das criadas nos blocos mais recentes (`X_imagens_*`, `Y_categorias_*`, `train_ds` e outras), podendo falhar em uma sessão nova.
- Há mistura de `keras`, `tf_keras` e `tensorflow.keras`, além de chamadas antigas como `predict_classes`, `fit_generator` e `layers.experimental.preprocessing`.
- Alguns trechos antigos aplicam binarização, limiares e perda binária; não devem ser confundidos com o fluxo multiclasse softmax do bloco principal.
- A resolução varia entre experimentos; imagens, rótulos e modelos precisam ser compatíveis.
- O bloco principal salva o melhor checkpoint, mas calcula as predições usando o modelo ao final do treinamento, sem recarregar automaticamente o melhor checkpoint.
- Existem índices fixos de imagens, variáveis não definidas e um trecho de exportação SavedModel incompleto, incluindo erro de sintaxe. As tentativas de exportação não são um fluxo validado de implantação.
- Não há garantia de generalização para pessoas ou condições fora do conjunto de treino, nem avaliação reproduzida de equidade, robustez ou segurança.

## Privacidade da versão pública

Foram removidos os resultados de execução, imagens incorporadas, contadores de execução e metadados específicos da sessão. Nomes de participantes, exemplos de arquivos pessoais, caminhos do Drive e o endereço de uma fotografia pessoal foram substituídos por referências locais genéricas. A montagem do Google Drive foi retirada e a exibição de fotografias ficou desativada por padrão.

Essa proteção não impede que uma execução futura gere novos resultados, nomes de classes ou imagens. Revise e limpe as saídas antes de compartilhar novamente. Não publique datasets, modelos ou informações de participantes sem autorização. A `.gitignore` ajuda a evitar inclusões acidentais, mas não substitui essa revisão.
