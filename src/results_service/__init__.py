"""The dedicated results microservice.

Structurally a second :mod:`src.backend` in miniature — app factory, ``api/``,
a service object, a repository — and deliberately much smaller: no Supabase,
no machine learning, no frontend. It exists as its own deployment so the
scraping dependencies (Playwright, Chromium, ~400 MB) stay out of the
application image, and so a provider outage or a blocked scraper is contained
here instead of degrading the rest of the product.

The main app never imports this package. It talks to it over HTTP through
:mod:`src.backend.services.results_gateway`, using the shared wire contract in
:mod:`src.contracts.results`.
"""
