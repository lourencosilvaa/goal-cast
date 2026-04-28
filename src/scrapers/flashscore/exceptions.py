class FlashScoreHttpError(Exception):
    """Raised when the HTTP client fails to fetch data from FlashScore."""


class FlashScoreUnavailableError(Exception):
    """Raised when both HTTP and Playwright clients fail."""
