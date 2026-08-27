class AppError(Exception):
    """Base application error mapped to an HTTP response."""

    status_code: int = 500

    def __init__(self, detail: str) -> None:
        self.detail = detail
        self.status_code = type(self).status_code
        super().__init__(detail)


class InvalidAudioFileError(AppError):
    """Raised when an uploaded file fails validation."""

    status_code = 400


class InvalidDiarizationModeError(AppError):
    """Raised when diarization is not local or api."""

    status_code = 400


class MissingPyannoteApiKeyError(AppError):
    """Raised when API diarization is requested without a configured key."""

    status_code = 400


class TranscriptionError(AppError):
    """Raised for internal failures during job setup."""

    status_code = 500


class TranscriptNotFoundError(AppError):
    """Raised when a transcript file is missing or not ready yet."""

    status_code = 404
