# Análise Visual de Equipamentos de Proteção Individual (EPIs) com Visão Computacional e IA Generativa

Projeto acadêmico integrado entre as disciplinas de **Processamento de Imagens e Sinais** e **Inteligência Artificial / IA Generativa**.

---

## Status do Projeto

Esta versão corresponde à entrega de **Processamento de Imagens** (19/10/2026).

- **Neste repositório:** o serviço de processamento de imagens (`POST /process`), com testes, e a avaliação A/B das técnicas de realce com o modelo treinado (`best.pt`, um YOLOv8n), feita no split de validação. Os resultados estão na seção **Serviço de processamento de imagens**.
- **Fora deste repositório:** o serviço do modelo (`POST /detect`), do Vini. O contrato entre os dois serviços está em [`docs/contrato-api.md`](docs/contrato-api.md).
- **Ainda não implementado:** a integração com IA Generativa e a avaliação final no split de teste. Neste repositório, a Etapa 5 (estado por EPI) existe, por enquanto, só no script que gera as figuras da apresentação.

---

## Integrantes

- Abner Wallace da Costa Rodrigues — RA 24111862
- Elidio Giacon Neto — RA 24111663
- Guilherme Martins Messa — RA 24111940
- Maria Antonia Rodrigues Vieira — RA 24111894
- Victor Silva Murakami — RA 24111697
- Vinicius Henrique Dalaqua da Silva — RA 24111678

---

## Descrição do Projeto

### Problema

A verificação do uso correto de Equipamentos de Proteção Individual em ambientes de trabalho é normalmente feita por inspeção humana, o que é trabalhoso, sujeito a falhas de atenção e difícil de registrar de forma padronizada.

### Proposta

Desenvolver um protótipo acadêmico capaz de:

1. analisar uma imagem contendo um trabalhador;
2. identificar, por visão computacional, a presença ou a ausência explícita de determinados EPIs;
3. converter esse resultado em um **dado estruturado**;
4. enviar esse dado estruturado a um modelo de **IA Generativa**, que produzirá um relatório textual de orientação.

### Objetivo

Demonstrar, de ponta a ponta, a integração entre uma etapa de processamento/análise de imagens e uma etapa de geração de linguagem natural, evidenciando a separação clara de responsabilidades entre as duas técnicas.

### Escopo do MVP

- uma imagem de entrada;
- um trabalhador principal por imagem;
- três EPIs: **capacete**, **colete** e **óculos de proteção**;
- saída estruturada + relatório textual.

### Relevância

O tema é relevante do ponto de vista acadêmico porque exige um pipeline completo e realista: dataset anotado, detecção de objetos, interpretação de resultados e geração de texto condicionada a dados estruturados.

**Importante:** trata-se de um **protótipo acadêmico**, não de um sistema industrial ou certificado de segurança do trabalho. Os resultados automáticos possuem limitações e não devem ser usados como base para decisões reais de segurança.

---

## Escopo

**Dentro do escopo inicial:**

- um trabalhador principal por imagem (sem associação de EPIs a múltiplas pessoas);
- classes positivas: `helmet`, `vest`, `goggles`;
- classes negativas explícitas: `no-helmet`, `no-vest`, `no-goggles`.

**Sobre as classes negativas:** o dataset escolhido possui anotações explícitas de ausência. Elas serão preservadas justamente para **evitar a inferência incorreta de que "não detectado" é o mesmo que "ausente"**. Uma detecção ausente pode significar apenas falha do modelo, oclusão ou baixa confiança.

**Fora do escopo inicial:**

- `boots` / `no-boots`;
- `gloves` / `no-gloves`;
- múltiplas pessoas por imagem;
- vídeo ou processamento em tempo real;
- interface gráfica.

**Ajuste futuro possível:** caso a classe `goggles` / `no-goggles` apresente qualidade claramente insuficiente durante os experimentos, o MVP poderá ser reduzido para capacete + colete. Essa redução **não** é a decisão atual, apenas um plano de contingência.

---

## Dataset

### Dataset principal

- **Nome:** PPE detection — dam
- **Plataforma:** Roboflow Universe
- **Link:** https://universe.roboflow.com/dam-zapvl/ppe-detection-qlq3d-i0ezw
- **Licença:** CC BY 4.0

**Características:**

- aproximadamente **5.140 imagens**;
- tarefa de **Object Detection**, com anotações em **bounding boxes**;
- divisão aproximada de 70% treino / 20% validação / 10% teste;
- exportação disponível em formatos compatíveis com YOLO.

**Classes disponíveis no dataset:**

`helmet`, `vest`, `goggles`, `boots`, `gloves`, `no-helmet`, `no-vest`, `no-goggles`, `no-boots`, `no-gloves`

**Classes que serão utilizadas:**

`helmet`, `no-helmet`, `vest`, `no-vest`, `goggles`, `no-goggles`

### Estratégia de uso dos dados

1. **Inspeção inicial:** será feita uma inspeção visual por amostragem para validar a qualidade das imagens e, principalmente, a consistência das annotations — com atenção especial a `goggles` / `no-goggles`.
2. **Filtragem:** as classes fora do escopo poderão ser removidas por um script simples de filtragem. Nenhum modelo adicional será criado apenas para selecionar amostras.
3. **Splits:** a divisão original de train / validation / test será preservada, evitando vazamento entre conjuntos.
4. **Volume:** não será fixado um limite artificial de imagens. O protótipo poderá usar um subconjunto representativo para validar o pipeline, e o treinamento final poderá usar todas as imagens relevantes após a filtragem, se fizer sentido.
5. **Diversidade:** será mantida diversidade de poses, iluminação e distância da câmera, evitando selecionar apenas imagens fáceis.

### Dataset alternativo

> **Observação:** como alternativa, caso o dataset principal apresente problemas relevantes de qualidade, foi identificado o dataset **PPE — caps Workspace** (Roboflow Universe, aproximadamente 16.223 imagens, CC BY 4.0): https://universe.roboflow.com/caps-workspace-sapoe/ppe-cpxsz-bkrmw
>
> Este dataset **ainda não foi escolhido para uso** e consta apenas como plano de contingência.

---

## Pipeline de Processamento de Imagens

```mermaid
flowchart TD
    A[Imagem de entrada] --> B["Carregamento da imagem (nosso serviço, POST /process)"]
    B --> C["Preparação: resize / letterbox / normalização (Ultralytics, serviço do modelo)"]
    C --> D[Inferência com YOLOv8n fine-tuned]
    D --> E[Bounding boxes + classe + confidence]
    E --> F[Filtragem e interpretação das detecções]
    F --> G[Resultado estruturado dos EPIs]
```

**Etapa 1 — Carregamento da imagem.** A imagem é lida e convertida para um array numérico (matriz de pixels), formato exigido pelas etapas seguintes. No protótipo, isso acontece no nosso serviço (`POST /process`): a foto chega por upload, o formato é validado pelo conteúdo, a orientação EXIF é aplicada e o serviço devolve um PNG sem perdas, do mesmo tamanho da foto orientada.

**Etapa 2 — Preparação / resize / letterbox.** A imagem é redimensionada para a resolução de entrada esperada pela rede. O *letterbox* aplica o redimensionamento preservando a proporção original e preenchendo as bordas, o que evita distorcer objetos como capacetes e coletes. Também ocorre a normalização dos valores de pixel. Essa etapa fica dentro do Ultralytics, no serviço do modelo; o nosso serviço não redimensiona a imagem.

**Etapa 3 — Inferência com YOLOv8n.** A imagem preparada é submetida ao modelo de detecção, que produz um conjunto de candidatos de detecção sobre a imagem inteira em uma única passagem.

**Etapa 4 — Bounding boxes, classe e confidence.** Cada detecção resultante contém as coordenadas da caixa delimitadora, a classe prevista e um valor de confiança. Detecções redundantes sobre o mesmo objeto são eliminadas por *Non-Maximum Suppression* (NMS).

**Etapa 5 — Filtragem e interpretação das detecções.** É a etapa de lógica própria do projeto: aplicação de limiar mínimo de confiança, descarte de classes fora do escopo e consolidação das detecções em um estado por EPI. Aqui será tratada explicitamente a distinção entre três situações: EPI identificado, ausência explicitamente identificada (classe negativa detectada) e resultado indeterminado (nenhuma das duas). Se o modelo detectar na mesma imagem a classe positiva e a negativa do mesmo EPI (por exemplo, `helmet` e `no-helmet`), o estado desse EPI também é **indeterminado**. Neste repositório, a regra está em `scripts/make_segmentation_figures.py`, com testes, e é usada para gerar as figuras da apresentação; ela ainda não faz parte do nosso serviço.

**Etapa 6 — Resultado estruturado.** A saída final da etapa de visão é um objeto estruturado, por exemplo:

```text
capacete: identificado
colete:   identificado
oculos:   ausencia explicitamente identificada
```

> **Nota sobre o framework:** boa parte do preprocessing, o NMS e a execução da inferência são gerenciados internamente pelo **Ultralytics**. Essas funcionalidades **não serão reimplementadas manualmente** sem necessidade; o esforço do projeto se concentra na preparação dos dados, no fine-tuning e na etapa de interpretação (etapas 5 e 6).

---

## Metodologia de Visão Computacional

- **Tipo de problema:** Object Detection.
- **Modelo:** YOLOv8n (variante *nano*), com **pesos pré-treinados** (`yolov8n.pt`).
- **Framework:** Ultralytics.
- **Estratégia de treinamento:** Transfer Learning / fine-tuning a partir dos pesos pré-treinados — a rede **não** será treinada do zero.

**Por que Object Detection.** O dataset escolhido já é anotado com *bounding boxes*. Converter o problema para classificação de imagem descartaria a informação de localização, sem simplificar o projeto de forma relevante. Além disso, a detecção permite tratar cada EPI como um objeto independente na cena.

**Por que YOLOv8n com transfer learning.** Os pesos pré-treinados já codificam características visuais genéricas, o que reduz drasticamente a quantidade de dados e de tempo de treinamento necessários. A variante *nano* é a mais leve da família, adequada ao hardware disponível e ao prazo do projeto acadêmico. O fine-tuning foi feito no YOLOv8n; manter a família evita retreinar.

> **Mudança em relação ao planejamento:** a primeira versão deste README previa o YOLO11n. O único treino feito foi com o YOLOv8n, e retreinar não cabia no prazo. A família, o framework (Ultralytics) e as etapas do pipeline são os mesmos.

---

## Serviço de processamento de imagens (este repositório)

O serviço recebe a foto, prepara-a para o modelo e devolve um PNG. O contrato completo (parâmetros, cabeçalhos e erros) está em [`docs/contrato-api.md`](docs/contrato-api.md).

**Fluxo combinado entre os serviços.** O app do Vini envia a foto para o nosso `POST /process`. Respondemos com um PNG sem perdas, do mesmo tamanho da imagem orientada, e o app repassa esse PNG ao `POST /detect` do Vini. O letterbox, a normalização e o NMS continuam dentro do Ultralytics, no serviço do Vini.

**O que o serviço faz:**

- identifica o formato (JPEG ou PNG) pelos primeiros bytes e recusa outros formatos, arquivos vazios ou corrompidos e arquivos acima de 10 MiB ou de 40 milhões de pixels;
- aplica a orientação EXIF (foto de celular "deitada" fica em pé) e devolve o PNG sem EXIF;
- aplica uma técnica de realce opcional; o padrão é `none`, isto é, nenhuma;
- mede a nitidez por FFT (fração da energia do espectro acima de 0,25 ciclo/pixel, com a imagem redimensionada até o lado maior ter 640 px) e a devolve no cabeçalho `X-Sharpness-HF-Ratio`. Ainda não há limiar calibrado: o valor é só informativo.

**Como rodar** (Python 3.14; a porta padrão do serviço é a 8001):

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python -m uvicorn --app-dir src epi_imgproc.api:app --port 8001
```

### Avaliação A/B das técnicas de pré-processamento

**Pergunta:** algum tratamento de imagem melhora o modelo? Avaliamos a linha de base e 7 técnicas nas 1026 imagens do split de validação (versão 1 do dataset no Roboflow). Todas passaram pelo mesmo pipeline do serviço, com o mesmo `best.pt`. O split de teste não foi usado.

**Regra, registrada antes da rodada:** uma técnica só vira padrão se o mAP50 das 6 classes do README subir e o recall de nenhuma delas cair, em relação à linha de base.

| Técnica | mAP50 (6 classes) | recall helmet | recall no-helmet | recall vest | recall no-vest | recall goggles | recall no-goggles | Veredito |
|---|---|---|---|---|---|---|---|---|
| nenhuma (`none`, linha de base) | 0,6804 | 0,9486 | 0,2913 | 0,9489 | 0,7798 | 0,6814 | 0,5677 | n/a |
| equalização de histograma | 0,6727 | 0,9495 | 0,3107 | 0,9463 | 0,7752 | 0,6195 | 0,4968 | reprovada |
| CLAHE | 0,6770 | 0,9486 | 0,3107 | 0,9419 | 0,7752 | 0,6460 | 0,5871 | reprovada |
| correção gama (0,8) | 0,6789 | 0,9520 | 0,3204 | 0,9445 | 0,7936 | 0,6903 | 0,5419 | reprovada |
| suavização gaussiana | 0,6743 | 0,9334 | 0,2913 | 0,9419 | 0,7844 | 0,6106 | 0,5613 | reprovada |
| unsharp mask | 0,6768 | 0,9478 | 0,3107 | 0,9541 | 0,7569 | 0,6726 | 0,5484 | reprovada |
| passa-baixa FFT | 0,6809 | 0,9469 | 0,2913 | 0,9445 | 0,7890 | 0,6372 | 0,5613 | reprovada |
| ênfase de altas frequências FFT | 0,6785 | 0,9495 | 0,3010 | 0,9489 | 0,7706 | 0,6903 | 0,5806 | reprovada |

Como cada métrica foi medida:

- **mAP50:** validação padrão do Ultralytics (`model.val`), com conf 0,001, IoU do NMS 0,7, `imgsz` 640, `rect` e `max_det` 300. A coluna é a média do AP50 das 6 classes do README.
- **Recall:** contagem exata, nas predições dessa mesma validação. Para cada classe, conta as predições com confiança ≥ 0,25 que o Ultralytics casou com uma caixa anotada da mesma classe com IoU ≥ 0,5 e divide pelo número de caixas anotadas da classe. A primeira rodada (16/10) lia o recall de uma curva interpolada do Ultralytics. A contagem exata (17/10) mudou 21 dos 48 valores de recall e nenhum veredito; o mAP50 ficou igual em todos os dígitos.

**Resultado:** nenhuma das 7 técnicas, com os parâmetros testados, passou na regra nesta validação. Por isso o serviço entrega, por padrão, a imagem sem tratamento (`technique=none`). O passa-baixa FFT foi o único a subir o mAP50, em menos de 0,001 (0,6809 contra 0,6804), mas o recall de 4 classes caiu. Os parâmetros de cada técnica estão em [`docs/contrato-api.md`](docs/contrato-api.md).

Detalhes da avaliação:

- O script é `scripts/run_ab_eval.py`. As 8 rodadas avaliaram as 1026 imagens, conforme a contagem do próprio Ultralytics, e a linha de base é idêntica pixel a pixel ao que o Ultralytics leria do arquivo original.
- Na cópia dos rótulos usada na avaliação, 16 anotações em polígono foram convertidas em caixas (a mesma regra do Ultralytics). Sem isso, o Ultralytics descartaria 6 imagens.
- A versão do dataset usada no treino ainda não foi confirmada (o checkpoint aponta `PPE-detection-1`). Se imagens desta validação estiveram no treino, os valores absolutos estão otimistas; por isso as técnicas são comparadas só entre si.
- Há uma única divisão de validação, e as diferenças de mAP50 em relação à linha de base ficam entre −0,0078 e +0,00045. Diferenças dessa ordem podem ser ruído.

### Decisões tomadas depois do planejamento

| Decisão | Motivo |
| --- | --- |
| Modelo YOLOv8n no lugar do YOLO11n previsto | O único treino feito foi com o YOLOv8n (ver **Metodologia de Visão Computacional**). |
| Dois serviços REST: processamento de imagens (este repositório, com FastAPI) e modelo (`POST /detect`, do Vini) | Divisão do trabalho no grupo; o contrato entre os dois serviços está em `docs/contrato-api.md`. |
| Porta padrão 8001 para o nosso serviço | Permite rodar os dois serviços na mesma máquina. |
| Conflito na Etapa 5 (classe positiva e negativa do mesmo EPI na mesma imagem) vira "indeterminado" | Um trabalhador principal não pode usar e não usar o mesmo EPI ao mesmo tempo. |

---

## Integração com Inteligência Artificial

O projeto combina **duas IAs com responsabilidades distintas e não sobrepostas**.

### Pipeline completo

```mermaid
flowchart TD
    A[Imagem de entrada] --> B[Processamento da imagem]
    B --> C[YOLOv8n fine-tuned]
    C --> D[Deteccoes: boxes + classe + confidence]
    D --> E[Resultado estruturado dos EPIs]
    E --> F[Montagem do prompt estruturado]
    F --> G[Modelo de IA Generativa]
    G --> H[Relatorio / orientacao textual]
```

### Responsabilidades

| Componente | Responsabilidade |
| --- | --- |
| **Visão Computacional (YOLOv8n)** | Identificar os EPIs na imagem e produzir um resultado estruturado e verificável. |
| **IA Generativa (LLM)** | Interpretar o resultado estruturado recebido e transformá-lo em uma resposta textual compreensível e contextualizada. |

A IA Generativa **não é responsável por detectar os EPIs** e não recebe a tarefa de analisar a imagem. Ela atua exclusivamente sobre o dado já estruturado pela etapa de visão.

### Exemplo conceitual

Entrada fornecida ao modelo generativo:

```text
Capacete: identificado
Colete: identificado
Óculos: ausência explicitamente identificada
```

Saída esperada: uma síntese textual do resultado da análise, acompanhada de orientação geral sobre o item cuja ausência foi identificada.

### Definição da ferramenta generativa

A API / modelo de IA Generativa **ainda não foi definida**. A escolha será feita posteriormente, durante a etapa de exploração da disciplina de IA Generativa, considerando disponibilidade, custo e adequação ao formato de prompt estruturado descrito acima. Por esse motivo, nenhum SDK de LLM consta no `requirements.txt` desta etapa.

---

## Limitações e cuidados

- **`goggles` é uma classe visualmente mais difícil:** óculos de proteção são objetos pequenos, frequentemente transparentes e sujeitos a oclusão, o que tende a reduzir a qualidade da detecção.
- **Classes fracas na validação:** com o `best.pt`, o recall é de 0,29 em `no-helmet`, 0,57 em `no-goggles` e 0,68 em `goggles` (seção **Serviço de processamento de imagens**). A maioria das ausências de capacete anotadas não é detectada, e, quando nenhuma das duas classes aparece, o estado fica indeterminado.
- **Qualidade das annotations:** o dataset é de origem pública e as anotações precisam ser inspecionadas antes do treinamento final.
- **Ausência de detecção ≠ ausência real:** a não detecção de uma classe positiva não será tratada como comprovação de que o EPI não está sendo usado.
- **Protótipo acadêmico:** o sistema não é uma solução certificada de segurança do trabalho e não substitui inspeção humana.
- **IA Generativa pode produzir texto incorreto:** por isso ela receberá contexto estruturado e restrito, e sua saída deverá ser avaliada criticamente, sem assumir que o texto gerado é automaticamente correto.

---

## Tecnologias

- Python 3.14
- Ultralytics / YOLOv8n (PyTorch)
- OpenCV, NumPy e Pillow
- FastAPI e uvicorn (serviço REST)
- pytest (testes)

As versões usadas estão fixadas no [`requirements.txt`](requirements.txt).

---

## Próximas etapas

1. Inspeção do dataset e da qualidade das annotations.
2. Preparação e filtragem das classes dentro do escopo.
3. Protótipo do pipeline com um subconjunto representativo.
4. Fine-tuning do YOLOv8n (feito: `best.pt`).
5. Avaliação dos resultados (feita na validação para as técnicas de pré-processamento; falta a avaliação final no split de teste).
6. Definição da ferramenta generativa e integração.
7. Demonstração final.

---

## Checklist da entrega

| Requisito do professor | Onde foi atendido |
| --- | --- |
| 1. Nome completo dos integrantes | Seção **Integrantes** |
| 2. Descrição detalhada do projeto, com o pipeline de processamento de imagens e explicação de cada etapa | Seções **Descrição do Projeto**, **Escopo**, **Pipeline de Processamento de Imagens** e **Metodologia de Visão Computacional** |
| 3. Descrição detalhada da integração com a disciplina de IA, com o pipeline completo | Seção **Integração com Inteligência Artificial** |
| 4. Arquivo `requirements.txt` | Arquivo [`requirements.txt`](requirements.txt) na raiz do repositório |
| 5. Link(s) do(s) dataset(s) utilizado(s) | Seção **Dataset** |
