from __future__ import annotations

import argparse
import json
from pathlib import Path

from adsmod_common.config import load_config
from adsmod_core.app import create_app


###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAPI JSON for the unified ADSMOD backend.")
    parser.add_argument("--config", required=True, type=Path, help="Canonical adsmod.json path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    schema = create_app(load_config(args.config)).openapi()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"OpenAPI written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
