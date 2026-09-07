from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from playwright.sync_api import Error, Page, TimeoutError, sync_playwright


ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "assets" / "figures"
MANIFEST_PATH = FIGURES_DIR / "manifest.json"
BASE_URL = "http://127.0.0.1:9580"
VIEWPORT = {"width": 1440, "height": 900}
MAX_RETRIES = 3

###############################################################################
@dataclass(slots=True)
class CaptureRecord:
    page_title: str
    route: str
    output_filenames: list[str]
    viewport: dict[str, int]
    notes: str

###############################################################################
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

###############################################################################
def wait_for_idle(page: Page, timeout_ms: int = 15000) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except TimeoutError:
        # Some polling widgets keep the network active; continue after a short settle.
        page.wait_for_timeout(1200)

###############################################################################
def make_deterministic(page: Page) -> None:
    page.add_style_tag(
        content="""
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
            caret-color: transparent !important;
            scroll-behavior: auto !important;
        }
        """
    )

###############################################################################
def dismiss_common_overlays(page: Page) -> None:
    selectors = (
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Close')",
        "button:has-text('Dismiss')",
        "button:has-text('Skip')",
        "button[aria-label='Close']",
        "button[title='Close']",
    )
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.is_visible(timeout=300):
                locator.click(timeout=1200)
                page.wait_for_timeout(300)
        except Error:
            continue

    # Hide common dev/error overlays if present.
    page.evaluate(
        """
        () => {
          const selectors = [
            'vite-error-overlay',
            '#webpack-dev-server-client-overlay',
            '#__next-build-watcher',
            '.intercom-lightweight-app',
            '#intercom-container'
          ];
          selectors.forEach((sel) => {
            document.querySelectorAll(sel).forEach((el) => {
              el.style.display = 'none';
            });
          });
        }
        """
    )

###############################################################################
def goto_with_retry(
    page: Page, url: str, description: str, retries: int = MAX_RETRIES
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            wait_for_idle(page)
            page.wait_for_timeout(700)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                page.wait_for_timeout(900 * attempt)
    raise RuntimeError(
        f"Failed to load {description} at {url}: {last_error}"
    ) from last_error

###############################################################################
def click_with_retry(
    action: Callable[[], None], label: str, retries: int = MAX_RETRIES
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            action()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(0.6 * attempt)
    raise RuntimeError(f"Failed action '{label}': {last_error}") from last_error

###############################################################################
def take_screenshot_with_segments(
    page: Page, output_name: str
) -> tuple[list[str], str]:
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(250)
    page.screenshot(path=str(FIGURES_DIR / output_name), full_page=False)
    return [output_name], "Viewport screenshot."

###############################################################################
def maybe_capture_scrolled_container(
    page: Page, container_selector: str, output_stem: str
) -> list[str]:
    extra_files: list[str] = []
    container_count = page.locator(container_selector).count()
    for idx in range(min(container_count, 2)):
        selector = f"{container_selector} >> nth={idx}"
        has_overflow = bool(
            page.locator(selector).evaluate(
                """
                (el) => (el.scrollHeight - el.clientHeight) > 40
                """
            )
        )
        if not has_overflow:
            continue
        page.locator(selector).evaluate(
            """
            (el) => { el.scrollTop = Math.max(0, el.scrollHeight - el.clientHeight); }
            """
        )
        page.wait_for_timeout(300)
        file_name = f"{output_stem}-scroll-{idx + 1}.png"
        page.screenshot(path=str(FIGURES_DIR / file_name), full_page=False)
        extra_files.append(file_name)
        page.locator(selector).evaluate("(el) => { el.scrollTop = 0; }")
    return extra_files

###############################################################################
def expect_visible(page: Page, selector: str, timeout_ms: int = 15000) -> None:
    page.locator(selector).first.wait_for(state="visible", timeout=timeout_ms)

###############################################################################
def capture_view(
    page: Page,
    title: str,
    route: str,
    output_name: str,
    wait_selector: str,
    notes: str = "",
    scroll_container_selector: str | None = None,
) -> CaptureRecord:
    expect_visible(page, wait_selector)
    wait_for_idle(page)
    page.wait_for_timeout(600)
    make_deterministic(page)
    dismiss_common_overlays(page)
    files, auto_notes = take_screenshot_with_segments(page, output_name)
    if scroll_container_selector:
        extra = maybe_capture_scrolled_container(
            page, scroll_container_selector, Path(output_name).stem
        )
        files.extend(extra)
        if extra:
            auto_notes += " Includes extra internal-scroll captures."

    merged_notes = auto_notes if not notes else f"{notes} {auto_notes}"
    return CaptureRecord(
        page_title=title,
        route=route,
        output_filenames=files,
        viewport=VIEWPORT,
        notes=merged_notes,
    )

###############################################################################
def main() -> int:
    ensure_dir(FIGURES_DIR)

    # Remove old generated files with the naming scheme used here.
    purge_patterns: Iterable[str] = (
        "home*.png",
        "fitting*.png",
        "training-data-processing*.png",
        "training-datasets*.png",
        "training-checkpoints*.png",
        "dashboard*.png",
    )
    for pattern in purge_patterns:
        for path in FIGURES_DIR.glob(pattern):
            path.unlink(missing_ok=True)

    failures: list[dict[str, str]] = []
    captures: list[CaptureRecord] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            locale="en-US",
            color_scheme="light",
            reduced_motion="reduce",
            ignore_https_errors=True,
        )
        page = context.new_page()

        def safe_step(name: str, fn: Callable[[], CaptureRecord]) -> None:
            try:
                captures.append(fn())
            except Exception as exc:  # noqa: BLE001
                failures.append({"step": name, "error": str(exc)})

        goto_with_retry(page, BASE_URL, "frontend root")
        expect_visible(page, "nav[aria-label='Primary']")

        safe_step(
            "landing-custom-datasets",
            lambda: capture_view(
                page=page,
                title="Custom Datasets",
                route="/datasets",
                output_name="home.png",
                wait_selector=".custom-datasets-page",
                notes="No login required.",
            ),
        )

        click_with_retry(
            lambda: page.get_by_role("link", name="Fitting").click(timeout=6000),
            "open fitting tab",
        )
        safe_step(
            "fitting",
            lambda: capture_view(
                page=page,
                title="Fitting Configuration",
                route="/ (fitting tab)",
                output_name="fitting.png",
                wait_selector=".models-page",
                notes="Settings-focused page.",
            ),
        )

        click_with_retry(
            lambda: page.get_by_role("link", name="Training").click(timeout=6000),
            "open training tab",
        )

        safe_step(
            "training-processing",
            lambda: capture_view(
                page=page,
                title="Training - Data Processing",
                route="/ (training tab, processing view)",
                output_name="training-data-processing.png",
                wait_selector=".training-workspace",
                notes="No login required.",
            ),
        )

        click_with_retry(
            lambda: page.locator(
                ".training-view-tab:has-text('Train datasets')"
            ).first.click(timeout=6000),
            "open train datasets view",
        )
        safe_step(
            "training-datasets",
            lambda: capture_view(
                page=page,
                title="Training - Datasets List",
                route="/ (training tab, datasets view)",
                output_name="training-datasets.png",
                wait_selector=".training-setup-container",
                notes="List-style view.",
                scroll_container_selector=".split-selection-card-content",
            ),
        )

        click_with_retry(
            lambda: page.locator(
                ".training-view-tab:has-text('Checkpoints')"
            ).first.click(timeout=6000),
            "open checkpoints view",
        )
        safe_step(
            "training-checkpoints",
            lambda: capture_view(
                page=page,
                title="Training - Checkpoints",
                route="/ (training tab, checkpoints view)",
                output_name="training-checkpoints.png",
                wait_selector=".training-setup-container",
                notes="List/detail entry point for checkpoint inspection.",
                scroll_container_selector=".split-selection-card-content",
            ),
        )

        click_with_retry(
            lambda: page.locator(
                ".training-view-tab:has-text('Training Dashboard')"
            ).first.click(timeout=6000),
            "open training dashboard view",
        )
        safe_step(
            "training-dashboard",
            lambda: capture_view(
                page=page,
                title="Training Dashboard",
                route="/ (training tab, dashboard view)",
                output_name="dashboard.png",
                wait_selector=".training-dashboard",
                notes="Dashboard page.",
            ),
        )

        browser.close()

    manifest = {
        "base_url": BASE_URL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "viewport": VIEWPORT,
        "captures": [
            {
                "page_title": c.page_title,
                "route_or_url": c.route,
                "output_filenames": c.output_filenames,
                "viewport": c.viewport,
                "notes": c.notes,
            }
            for c in captures
        ],
        "failures": failures,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Captured pages: {len(captures)}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f" - {failure['step']}: {failure['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
