"""One image ships the whole product, results included.

The results service runs in-process (``results_gateway.mode: in_process``), so
the application image has to carry what it imports. A missing COPY is not a
broken feature — it is a request that 503s in production with a green build.
"""

from pathlib import Path

from config.config_loader import ResultsGatewayConfig

ROOT = Path(__file__).resolve().parents[1]


def _copied() -> list[str]:
    return [
        line.split()[1]
        for line in (ROOT / "Dockerfile").read_text().splitlines()
        if line.startswith("COPY ") and not line.startswith("COPY --from")
    ]


def test_the_image_carries_the_results_service_and_provider_library():
    copied = _copied()
    assert "src/results_service/" in copied
    assert "src/scrapers/" in copied


def test_no_separate_results_image_remains():
    assert not (ROOT / "Dockerfile.results").exists()
    assert not (ROOT / ".github" / "workflows" / "deploy-results.yml").exists()


def test_the_deploy_workflow_rebuilds_on_results_code_changes():
    body = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert '"src/results_service/**"' in body
    assert '"src/scrapers/**"' in body


def test_in_process_is_the_default_gateway_mode():
    assert ResultsGatewayConfig().mode == "in_process"
