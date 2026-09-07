from __future__ import annotations

import argparse
import json
from pathlib import Path

from adsmod_common.config import AdsmodConfig

SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"

###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the canonical ADSMOD configuration JSON Schema."
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    schema = AdsmodConfig.model_json_schema()
    schema["$schema"] = SCHEMA_URI
    schema["title"] = "ADSMOD v3 configuration"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(schema, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Configuration schema written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
