"""Service-to-service authentication for the results service.

The rule this file exists to hold: **no key configured means the service does
not start**. Booting into an open-to-the-world state would be a silent
fallback of exactly the kind §7.4 forbids — and the failure mode is a scraping
endpoint anyone can point at whatever quota we are paying for.
"""

import pytest
from fastapi import HTTPException

from config.config_loader import ResultsServiceConfig
from src.results_service.auth import (
    ApiKeyGuard,
    MissingServiceKeyError,
    service_api_key,
)

_CONFIG = ResultsServiceConfig(api_key_env="RESULTS_SERVICE_API_KEY")


class TestReadingTheKey:
    def test_the_key_comes_from_the_configured_variable(self):
        key = service_api_key(_CONFIG, env={"RESULTS_SERVICE_API_KEY": "s3cret"})
        assert key == "s3cret"

    def test_a_different_variable_name_is_honoured(self):
        config = ResultsServiceConfig(api_key_env="OTHER_KEY")
        assert service_api_key(config, env={"OTHER_KEY": "abc"}) == "abc"

    def test_an_unset_variable_refuses_to_start_the_service(self):
        with pytest.raises(MissingServiceKeyError, match="RESULTS_SERVICE_API_KEY"):
            service_api_key(_CONFIG, env={})

    def test_an_empty_variable_is_treated_as_unset(self):
        """An empty string would otherwise authenticate every request that
        also sends nothing."""
        with pytest.raises(MissingServiceKeyError):
            service_api_key(_CONFIG, env={"RESULTS_SERVICE_API_KEY": ""})

    def test_surrounding_whitespace_is_not_part_of_the_key(self):
        key = service_api_key(_CONFIG, env={"RESULTS_SERVICE_API_KEY": " s3cret\n"})
        assert key == "s3cret"


class TestVerifying:
    def test_the_expected_key_is_accepted(self):
        ApiKeyGuard("s3cret").verify("s3cret")

    def test_a_wrong_key_is_rejected_as_unauthorised(self):
        with pytest.raises(HTTPException) as raised:
            ApiKeyGuard("s3cret").verify("wrong")
        assert raised.value.status_code == 401

    def test_a_missing_header_is_rejected(self):
        with pytest.raises(HTTPException) as raised:
            ApiKeyGuard("s3cret").verify(None)
        assert raised.value.status_code == 401

    def test_an_empty_header_is_rejected(self):
        with pytest.raises(HTTPException) as raised:
            ApiKeyGuard("s3cret").verify("")
        assert raised.value.status_code == 401

    def test_a_prefix_of_the_key_is_rejected(self):
        with pytest.raises(HTTPException):
            ApiKeyGuard("s3cret").verify("s3c")

    def test_the_rejection_does_not_echo_the_expected_key(self):
        with pytest.raises(HTTPException) as raised:
            ApiKeyGuard("s3cret").verify("wrong")
        assert "s3cret" not in str(raised.value.detail)

    def test_a_guard_cannot_be_built_without_a_key(self):
        """Belt and braces: the boot path already refuses, and a guard
        constructed with "" would accept a missing header."""
        with pytest.raises(MissingServiceKeyError):
            ApiKeyGuard("")
