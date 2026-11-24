"""Safe, typed errors used by API services and routes."""


class AudioUploadError(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ModelUnavailableError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass
