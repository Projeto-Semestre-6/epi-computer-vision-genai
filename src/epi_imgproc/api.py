"""REST endpoint of the image-processing service.

Run: ``.venv/bin/python -m uvicorn --app-dir src epi_imgproc.api:app --port 8001``
(or ``PYTHONPATH=src .venv/bin/python -m epi_imgproc.api``, which listens on DEFAULT_PORT).
Contract (pt-BR, for the model-service side): docs/contrato-api.md
"""

from __future__ import annotations

from enum import Enum

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from epi_imgproc import frequency, intake, techniques

# Not the model service's port (backend/main.py), so both services can run on one machine.
DEFAULT_PORT = 8001

# Multipart framing (boundaries, part headers) on top of the file itself.
MULTIPART_ALLOWANCE = 64 * 1024

STATUS_BY_CODE = {
    "empty": 400,
    "corrupt": 400,
    "unsupported_format": 415,
    "too_large": 413,
    "too_many_pixels": 413,
}

TechniqueName = Enum("TechniqueName", {name: name for name in techniques.TECHNIQUES}, type=str)

app = FastAPI(
    title="EPI image-processing service",
    version="0.1.0",
    description="Recebe uma imagem, prepara-a para o modelo YOLO e devolve um PNG do mesmo tamanho.",
)


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=STATUS_BY_CODE[code], content={"code": code, "detail": message})


@app.middleware("http")
async def reject_oversized_body(request: Request, call_next):
    """Refuse bodies announced as too large before Starlette spools them to disk."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit():
        if int(declared) > intake.MAX_UPLOAD_BYTES + MULTIPART_ALLOWANCE:
            return _error(
                "too_large",
                f"Requisição com {declared} bytes excede o limite de {intake.MAX_UPLOAD_BYTES} bytes (10 MiB) por imagem.",
            )
    return await call_next(request)


@app.exception_handler(intake.IntakeError)
async def intake_error_handler(request: Request, exc: intake.IntakeError) -> JSONResponse:
    return _error(exc.code, exc.message)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "techniques": list(techniques.TECHNIQUES), "default": techniques.DEFAULT_TECHNIQUE}


@app.post(
    "/process",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG com o mesmo tamanho da imagem orientada."},
        400: {"description": "Arquivo vazio ou corrompido."},
        413: {"description": "Acima de 10 MiB ou de 40 milhões de pixels."},
        415: {"description": "Formato diferente de JPEG ou PNG."},
        422: {"description": "Campo `file` ausente ou técnica desconhecida."},
    },
)
def process(
    file: UploadFile = File(..., description="Imagem JPEG ou PNG, até 10 MiB."),
    technique: TechniqueName = Query(TechniqueName(techniques.DEFAULT_TECHNIQUE), description="Técnica a aplicar."),
) -> Response:
    # Read one byte past the limit so an oversized upload is detected without loading all of it.
    data = file.file.read(intake.MAX_UPLOAD_BYTES + 1)
    decoded = intake.decode_image(data)
    sharpness = frequency.high_frequency_energy_ratio(decoded.pixels)
    processed = techniques.apply_technique(technique.value, decoded.pixels)
    height, width = processed.shape[:2]
    headers = {
        "X-Technique": technique.value,
        "X-Sharpness-HF-Ratio": f"{sharpness:.6f}",
        "X-EXIF-Orientation": str(decoded.exif_orientation),
        "X-Source-Format": decoded.source_format,
        "X-Image-Size": f"{width}x{height}",
    }
    return Response(content=intake.encode_png(processed), media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
