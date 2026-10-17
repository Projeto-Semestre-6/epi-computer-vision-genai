# Contrato da API do serviço de processamento de imagens

Versão 0.1.0, 16/10/2026, atualizado em 17/10/2026. Público: Vini (serviço do modelo) e quem integra o app.

## Fluxo

1. O app do Vini envia a foto para o nosso `POST /process`.
2. Respondemos com um PNG do **mesmo tamanho** da imagem orientada, sem redimensionar.
3. O app repassa esse PNG ao `POST /detect` do Vini, com um nome de arquivo terminado em `.png`, porque o `/detect` valida a extensão.
4. As caixas que o modelo devolve valem diretamente sobre o PNG e sobre a foto orientada: as coordenadas são as mesmas.

O letterbox, a normalização e o NMS continuam dentro do Ultralytics, no serviço do Vini. Não os repetimos aqui.

## `POST /process`

Requisição `multipart/form-data`:

| Parte | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `file` | arquivo | sim | JPEG ou PNG, até 10 MiB (10.485.760 bytes) e até 40 milhões de pixels. |
| `technique` | parâmetro de query | não | Técnica a aplicar. Padrão: `none`. |

O formato é identificado pelos primeiros bytes do arquivo. O nome do arquivo e o `Content-Type` são ignorados.

### Resposta 200

- Corpo: PNG sem perdas, 8 bits, 3 canais (RGB), **sem EXIF nem outros metadados**.
- Tamanho: igual ao da imagem depois de aplicada a orientação EXIF. Nada é redimensionado.

Cabeçalhos:

| Cabeçalho | Exemplo | Significado |
|---|---|---|
| `X-Image-Size` | `4000x3000` | Largura × altura do PNG devolvido. |
| `X-EXIF-Orientation` | `6` | Valor da tag Orientation aplicado (1 = nenhuma mudança). |
| `X-Source-Format` | `jpeg` | Formato detectado na entrada. |
| `X-Technique` | `none` | Técnica aplicada. |
| `X-Sharpness-HF-Ratio` | `0.081234` | Métrica de nitidez da entrada (seção abaixo). |

### Política de orientação EXIF

Celulares gravam a foto "deitada" e indicam o giro na tag EXIF Orientation. Nós lemos a tag, giramos ou espelhamos os pixels e devolvemos o PNG já em pé e sem EXIF. Consequências:

- Nas orientações 5 a 8, largura e altura trocam em relação ao raster gravado no JPEG. "Mesmo tamanho" significa o tamanho da foto como ela aparece na tela.
- Ninguém depois de nós precisa aplicar EXIF de novo, e não há risco de girar duas vezes: o PNG não carrega a tag.
- É o mesmo comportamento do OpenCV 5.0 (`IMREAD_COLOR`) e, portanto, do Ultralytics ao ler arquivos. Os testes conferem isso pixel a pixel nas 8 orientações.
- Tag ausente, ilegível ou fora de 1 a 8 é tratada como 1.

### Outras conversões

- Transparência (canal alfa) é descartada sem composição.
- Imagem em tons de cinza vira 3 canais iguais.
- PNG de 16 bits vira 8 bits.

São as mesmas conversões que o Ultralytics faz ao ler um arquivo.

### Métrica de nitidez (`X-Sharpness-HF-Ratio`)

É a fração da energia do espectro de Fourier acima de 0,25 ciclo/pixel, em que 0,5 é o limite de Nyquist. Segue o material de FFT da disciplina: altas frequências são bordas e detalhes, e uma imagem borrada concentra a energia no centro do espectro.

- É medida na imagem de entrada já orientada, **antes** da técnica, em tons de cinza.
- É medida na escala que o modelo enxerga: a imagem é redimensionada (reduzida ou ampliada, como faz o Ultralytics) até o lado maior ter 640 px, o `imgsz` do `best.pt`. Sem isso, toda foto de alta resolução pareceria borrada. A interpolação não é a mesma do Ultralytics: usamos `INTER_AREA` na redução e `INTER_LINEAR` na ampliação, e o Ultralytics usa `INTER_LINEAR` nos dois sentidos.
- Antes da FFT, a média é subtraída e aplicamos uma janela de Hann, para que as bordas da imagem não criem altas frequências falsas. Um lado com 4 px ou menos fica sem janela nesse eixo. Com tão poucas amostras, a janela de Hann zera a imagem (2 px) ou põe metade ou mais da energia em 0,25 ciclo/pixel ou acima (3 e 4 px), e a métrica deixaria de depender do conteúdo.
- Valor entre 0 e 1. **Menor significa mais borrada.**
- **Ainda não há limiar calibrado.** Por enquanto o valor é só informativo e não deve bloquear nada. O limiar será calibrado com a distribuição da métrica no split de validação, que o script de avaliação grava em `sharpness_val.csv`.

### Técnicas disponíveis

Nenhuma técnica é aplicada por padrão. Uma técnica só vira padrão se passar na avaliação A/B com o `best.pt` no split de validação: o mAP50 das 6 classes do README tem de aumentar sem que o recall de nenhuma delas caia.

| `technique` | Domínio | Parâmetros |
|---|---|---|
| `none` | n/a | n/a |
| `histogram_equalization` | espacial | equalização global da luminância (L do CIELAB) |
| `clahe` | espacial | `clip_limit=2.0`, grade 8×8, na luminância |
| `gamma` | espacial | `gamma=0.8` (clareia sombras), por canal |
| `gaussian_blur` | espacial | `sigma=1.0` px na escala do modelo |
| `unsharp_mask` | espacial | `sigma=1.0`, `amount=1.0`, na luminância |
| `fft_lowpass` | frequência | passa-baixa gaussiano, `D0=0.25` ciclo/px na escala do modelo, na luminância |
| `fft_high_emphasis` | frequência | ênfase de altas frequências `H = 1 + 0.5·(1 − passa-baixa gaussiano)`, `D0=0.1`, na luminância |

Os parâmetros de vizinhança são definidos na escala do modelo e convertidos para a resolução original. Assim, uma foto de 4000 px recebe o mesmo efeito, visto pelo modelo, que uma imagem de 640 px.

### Erros

Os erros de validação têm corpo JSON `{"code": "...", "detail": "mensagem em português"}`.

| Status | `code` | Quando |
|---|---|---|
| 400 | `empty` | Arquivo com 0 bytes. |
| 400 | `corrupt` | Assinatura JPEG/PNG válida, mas o conteúdo está truncado ou danificado. |
| 413 | `too_large` | Arquivo acima de 10 MiB. Também quando o `Content-Length` declarado passa de 10 MiB + 64 KiB, caso em que a requisição é recusada antes de ser lida. |
| 413 | `too_many_pixels` | Imagem com mais de 40 milhões de pixels. |
| 415 | `unsupported_format` | Qualquer formato que não seja JPEG ou PNG (GIF, BMP, WebP, TIFF, HEIC…). |
| 422 | n/a | Campo `file` ausente ou `technique` desconhecida. Corpo padrão do FastAPI: `{"detail": [...]}`. |

## `GET /health`

Responde `{"status": "ok", "techniques": [...], "default": "none"}`.

## Exemplos

```bash
curl -sS -D - -o processada.png \
  -F "file=@foto.jpg" "http://localhost:8001/process?technique=none"
```

```python
import requests

PROCESS_URL = "http://localhost:8001/process"  # nosso serviço
DETECT_URL = "http://<endereço do serviço do Vini>/detect"  # ver "Em aberto"

with open("foto.jpg", "rb") as handle:
    processed = requests.post(
        PROCESS_URL,
        files={"file": ("foto.jpg", handle, "image/jpeg")},
        timeout=30,
    )
processed.raise_for_status()
detections = requests.post(
    DETECT_URL,
    files={"file": ("imagem.png", processed.content, "image/png")},
    timeout=30,
)
```

## Como rodar

```bash
cd ~/unisagrado/epi-computer-vision-genai
.venv/bin/python -m uvicorn --app-dir src epi_imgproc.api:app --port 8001 --host 127.0.0.1
```

**Porta.** O nosso serviço usa a 8001 por padrão (`DEFAULT_PORT` em `src/epi_imgproc/api.py`), para não colidir com o backend do Vini, que sobe na 8000 (`backend/main.py`), quando os dois rodam na mesma máquina.

## Em aberto

- Em que endereço o `/detect` responde na demo e qual nome de arquivo o app repassa a ele.
- Qual versão do dataset no Roboflow o Vini usou no treino. O checkpoint aponta `PPE-detection-1`. Até isso ser confirmado, a avaliação A/B usa o split de validação local e compara as técnicas só entre si.
- Se o `/detect` deve continuar com `conf=0.25` e `iou=0.45`. A regra do A/B mede o recall em `conf=0.25`.
