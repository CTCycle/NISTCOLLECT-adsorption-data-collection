from __future__ import annotations

import os
from pathlib import Path
import re

from adsmod_common.config import AdsmodConfig
from adsmod_common.paths import resolve_storage_root

_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")


###############################################################################
def _replace_windows_variable(match: re.Match[str]) -> str:
    return os.environ.get(match.group(1), match.group(0))

###############################################################################
def resolve_database_path(config: AdsmodConfig) -> Path:
    database_value = config.application.database.sqlite_path
    if not database_value:
        raise ValueError(
            "application.database.sqlite_path is required for embedded databases"
        )
    expanded_database = _WINDOWS_VARIABLE.sub(_replace_windows_variable, database_value)
    database = Path(os.path.expandvars(expanded_database)).expanduser()
    if database.is_absolute():
        return database.resolve()
    return (resolve_storage_root(config) / database).resolve()
