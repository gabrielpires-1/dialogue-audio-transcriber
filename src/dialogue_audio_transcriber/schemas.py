from pydantic import BaseModel


class TranscriptionAccepted(BaseModel):
    filename: str


class TranscriptionContent(BaseModel):
    filename: str
    text: str


class ErrorResponse(BaseModel):
    detail: str
