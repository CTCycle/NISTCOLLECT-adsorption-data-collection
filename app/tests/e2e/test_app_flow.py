"""E2E tests for UI navigation and page rendering."""

from __future__ import annotations

from playwright.sync_api import Page, expect

###############################################################################
class TestHomepage:
    """Tests for the main application shell and custom datasets page."""

    # -------------------------------------------------------------------------
    def test_homepage_shows_shell_and_custom_dataset_controls(
        self, page: Page, base_url: str
    ) -> None:
        """Verify the homepage exposes its primary shell and upload controls."""
        page.goto(base_url)

        expect(page).to_have_title("ADSMOD Adsorption Modeling")
        expect(page.locator(".console-brand-name")).to_have_text("ADSMOD")
        expect(page.locator("nav[aria-label='Primary']").first).to_be_visible()
        expect(page.locator(".custom-datasets-page")).to_be_visible()
        expect(page.locator("input[type='file']").first).to_be_attached()

###############################################################################
class TestHeaderNavigation:
    """Tests for header navigation between pages."""

    # -------------------------------------------------------------------------
    def test_navigate_to_models_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the fitting page exposes its model controls."""
        page.goto(base_url)
        page.get_by_role("link", name="Fitting").click()

        expect(page.locator("section:not([hidden]) .models-grid")).to_be_visible()
        expect(page.locator(".model-grid-card").first).to_be_visible()
        expect(page.locator("#model-card-langmuir")).to_be_visible()
        expect(page.locator("button:has-text('Start Fitting')")).to_be_visible()

    # -------------------------------------------------------------------------
    def test_navigate_to_training_page(self, page: Page, base_url: str) -> None:
        """Verify navigation to the Training page."""
        page.goto(base_url)
        page.get_by_role("link", name="Training").click()

        expect(page.locator(".route-workspace-training")).to_be_visible()
        expect(page.get_by_role("heading", name="Data Processing")).to_be_visible()
