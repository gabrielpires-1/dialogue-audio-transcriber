from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dialogue_audio_transcriber.config import get_settings
from dialogue_audio_transcriber.constants import DIARIZATION_LOCAL
from dialogue_audio_transcriber.cuda import preload_cuda_libraries
from dialogue_audio_transcriber.exceptions import AppError
from dialogue_audio_transcriber.schemas import (
    ErrorResponse,
    TranscriptionAccepted,
    TranscriptionContent,
)
from dialogue_audio_transcriber.service import TranscriptionService

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from faster_whisper import WhisperModel

    from dialogue_audio_transcriber.diarization import load_diarizer

    preload_cuda_libraries()
    settings = get_settings()
    model = WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    diarizer = load_diarizer(settings.hf_token, settings.diarization_device)
    app.state.transcription_service = TranscriptionService(
        model=model, diarizer=diarizer, settings=settings
    )
    yield


settings = get_settings()
app = FastAPI(title="dialogue-audio-transcriber", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail).model_dump(),
    )


@app.post(
    "/transcribe",
    response_model=TranscriptionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def transcribe(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    service: Annotated[TranscriptionService, Depends(get_service)],
    diarization: Annotated[str, Query()] = DIARIZATION_LOCAL,
) -> TranscriptionAccepted:
    mode = service.resolve_diarization_mode(diarization)
    filename, audio_path = await service.create_job(file)
    background_tasks.add_task(service.transcribe_and_write, audio_path, filename, mode)
    return TranscriptionAccepted(filename=filename)


@app.get(
    "/transcribe/{filename}",
    response_model=TranscriptionContent,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_transcription(
    filename: str,
    service: Annotated[TranscriptionService, Depends(get_service)],
) -> TranscriptionContent:
    text = service.read_transcript(filename)
    return TranscriptionContent(filename=filename, text=text)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="dialogue-audio-transcriber",
        description="Run the dialogue audio transcriber API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "dialogue_audio_transcriber.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
