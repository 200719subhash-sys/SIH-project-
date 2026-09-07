import json
from pathlib import Path

from pydantic import ValidationError

from .models import Scheme, SchemeCatalogue


class SchemeDataError(ValueError):
    pass


def _validation_message(error: ValidationError, context: str) -> str:
    details = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{context}.{location}: {item['msg']}")
    return "invalid scheme catalogue: " + "; ".join(details)


def load_catalogue(path: Path) -> SchemeCatalogue:
    try:
        with path.open("r", encoding="utf-8") as file:
            raw_catalogue = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemeDataError(f"could not load scheme catalogue: {exc}") from exc

    if not isinstance(raw_catalogue, dict):
        raise SchemeDataError("invalid scheme catalogue: root must be an object with scheme_data_version and schemes")

    raw_schemes = raw_catalogue.get("schemes")
    if not isinstance(raw_schemes, list):
        raise SchemeDataError("invalid scheme catalogue.schemes: expected an array")

    schemes: list[Scheme] = []
    for index, raw_scheme in enumerate(raw_schemes):
        context = f"scheme[{index}]"
        if isinstance(raw_scheme, dict):
            identifier = raw_scheme.get("id")
            if identifier:
                context = f"scheme[{index}] ({identifier})"
        try:
            schemes.append(Scheme.model_validate(raw_scheme))
        except ValidationError as exc:
            raise SchemeDataError(_validation_message(exc, context)) from exc

    raw_catalogue["schemes"] = schemes
    try:
        return SchemeCatalogue.model_validate(raw_catalogue)
    except ValidationError as exc:
        raise SchemeDataError(_validation_message(exc, "catalogue")) from exc