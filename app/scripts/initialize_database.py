from __future__ import annotations

import argparse
from pathlib import Path
import time

from adsmod_common.config import load_config
from adsmod_common.paths import resolve_storage_root
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.database.initializer import prepare_database_for_startup


###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize the canonical ADSMOD database."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the canonical adsmod.json configuration.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    config = load_config(args.config)
    storage_root = resolve_storage_root(config)
    result = prepare_database_for_startup(
        config.application.database,
        storage_root=storage_root,
    )
    elapsed = time.perf_counter() - started_at
    logger.info(
        "Database initialization completed in %.2f seconds "
        "(backend=%s, before=%s, after=%s, head=%s, applied_migrations=%s).",
        elapsed,
        result.backend,
        ",".join(result.before) or "base",
        ",".join(result.after) or "base",
        result.head,
        result.applied_migrations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
