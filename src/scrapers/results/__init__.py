"""Match results — history and live — as a library of interchangeable sources.

Nothing here imports FastAPI or knows an endpoint exists. The dedicated
results service (:mod:`src.results_service`) composes these classes; the main
backend never imports this package at all, which is what keeps Playwright and
Chromium out of the application image.
"""
