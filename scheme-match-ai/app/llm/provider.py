import os
from typing import Protocol

from ..models import ExtractedProfile


class ProviderUnavailableError(RuntimeError):
    pass


class ProfileExtractionProvider(Protocol):
    def extract(self, text: str) -> ExtractedProfile:
        ...


class LocalProfileExtractionProvider:
    """Safe local provider used when no external LLM is configured."""

    def extract(self, text: str) -> ExtractedProfile:
        from ..profile_extraction import extract_local_facts

        return extract_local_facts(text)


def configured_profile_extraction_provider() -> ProfileExtractionProvider:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider or provider == "local":
        return LocalProfileExtractionProvider()
    raise ProviderUnavailableError(f"unsupported configured provider: {provider}")
