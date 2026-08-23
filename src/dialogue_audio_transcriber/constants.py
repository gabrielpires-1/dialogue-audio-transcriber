TRANSCRIPTION_FAILURE_MESSAGE = "INTERNAL_ERROR: WAS NOT POSSIBLE TO TRANSCRIBE FILE"

TRANSCRIPT_NOT_FOUND_MESSAGE = (
    "The requested resources was not found. It does not exist or was not "
    "created yet. Wait a few seconds and try again."
)

ALLOWED_EXTENSIONS = frozenset({"wav", "mp3", "m4a", "ogg", "flac", "webm"})

ALLOWED_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "application/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/webm",
        "video/webm",
        "application/octet-stream",
    }
)

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
READ_CHUNK_SIZE = 1024 * 1024
