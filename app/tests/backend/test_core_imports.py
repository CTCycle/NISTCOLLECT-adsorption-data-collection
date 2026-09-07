from __future__ import annotations

import subprocess
import sys


###############################################################################
def test_core_app_import_does_not_load_ml_runtime() -> None:
    script = (
        "import sys; "
        "import adsmod_core.app; "
        "assert not any(name == 'adsmod_ml' or name.startswith('adsmod_ml.') "
        "for name in sys.modules); "
        "assert not any(name in sys.modules for name in ('torch', 'keras', 'sklearn'))"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
