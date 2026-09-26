from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def browser():
    playwright_sync = pytest.importorskip("playwright.sync_api")
    try:
        pw = playwright_sync.sync_playwright().start()
        b = pw.chromium.launch()
    except Exception as exc:
        pytest.skip(f"Playwright Chromium not available: {exc}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    p = context.new_page()
    yield p
    context.close()


def _fixture_url(name: str) -> str:
    return f"file://{FIXTURES_DIR / name}"


def test_extract_fields_resolves_labels_via_label_for_attribute(page):
    from agent.browser import _extract_fields

    page.goto(_fixture_url("greenhouse_form.html"))
    extracted = _extract_fields(page)

    labels = {e.spec.label: e for e in extracted}
    assert labels["Full Name"].spec.field_type == "text"
    assert labels["Full Name"].spec.required is True
    assert labels["Full Name"].label_resolved is True
    assert labels["Why do you want to work here?"].spec.field_type == "textarea"
    assert labels["Preferred Location"].spec.field_type == "select"
    assert labels["Preferred Location"].spec.options == ["Remote", "New York"]
    assert labels["Willing to relocate"].spec.field_type == "checkbox"
    assert labels["Resume"].spec.field_type == "file"
    assert all(e.label_resolved for e in extracted)


def test_extract_fields_resolves_labels_via_aria_label(page):
    from agent.browser import _extract_fields

    page.goto(_fixture_url("lever_form.html"))
    extracted = _extract_fields(page)

    labels = {e.spec.label for e in extracted}
    assert labels == {
        "Full Name", "Why do you want to work here?", "Preferred Location",
        "Willing to relocate", "Resume",
    }
    assert all(e.label_resolved for e in extracted)


def test_extract_fields_resolves_labels_via_nearest_text(page):
    from agent.browser import _extract_fields

    page.goto(_fixture_url("workday_form.html"))
    extracted = _extract_fields(page)

    labels = {e.spec.label for e in extracted}
    assert labels == {
        "Full Name", "Why do you want to work here?", "Preferred Location",
        "Willing to relocate", "Resume",
    }
    assert all(e.label_resolved for e in extracted)


def test_extract_fields_flags_unresolved_labels_on_broken_form(page):
    from agent.browser import _extract_fields

    page.goto(_fixture_url("broken_form.html"))
    extracted = _extract_fields(page)

    assert len(extracted) == 4
    assert all(e.label_resolved is False for e in extracted)


def test_apply_value_fills_text_field(page):
    from agent.browser import _apply_value, _extract_fields

    page.goto(_fixture_url("greenhouse_form.html"))
    extracted = _extract_fields(page)
    field = next(e for e in extracted if e.spec.label == "Full Name")

    _apply_value(field.locator, field.spec.field_type, "Jane Doe")

    assert field.locator.input_value() == "Jane Doe"


def test_apply_value_selects_option(page):
    from agent.browser import _apply_value, _extract_fields

    page.goto(_fixture_url("greenhouse_form.html"))
    extracted = _extract_fields(page)
    field = next(e for e in extracted if e.spec.label == "Preferred Location")

    _apply_value(field.locator, field.spec.field_type, "New York")

    assert field.locator.input_value() == "nyc"


def test_apply_value_checks_checkbox(page):
    from agent.browser import _apply_value, _extract_fields

    page.goto(_fixture_url("greenhouse_form.html"))
    extracted = _extract_fields(page)
    field = next(e for e in extracted if e.spec.label == "Willing to relocate")

    _apply_value(field.locator, field.spec.field_type, "true")

    assert field.locator.is_checked() is True


def test_apply_value_sets_file_input(page):
    from agent.browser import _apply_value, _extract_fields

    page.goto(_fixture_url("greenhouse_form.html"))
    extracted = _extract_fields(page)
    field = next(e for e in extracted if e.spec.label == "Resume")

    _apply_value(field.locator, field.spec.field_type, str(FIXTURES_DIR / "sample_resume.txt"))

    assert field.locator.evaluate("el => el.files.length") == 1


def test_fetch_fields_returns_field_specs_for_wellformed_form(browser):
    from agent.browser import PlaywrightPageFetcher

    fetcher = PlaywrightPageFetcher(browser)
    fields = fetcher.fetch_fields(_fixture_url("greenhouse_form.html"))

    labels = {f.label for f in fields}
    assert labels == {
        "Full Name", "Why do you want to work here?", "Preferred Location",
        "Willing to relocate", "Resume",
    }


def test_fetch_fields_raises_form_unparseable_for_broken_form(browser):
    from agent.browser import PlaywrightPageFetcher
    from agent.models import FormUnparseable

    fetcher = PlaywrightPageFetcher(browser)

    with pytest.raises(FormUnparseable):
        fetcher.fetch_fields(_fixture_url("broken_form.html"))


def test_fill_does_not_raise_and_does_not_submit(browser):
    from agent.browser import PlaywrightFormFiller

    filler = PlaywrightFormFiller(browser)
    filler.fill(_fixture_url("greenhouse_form.html"), {
        "Full Name": "Jane Doe",
        "Why do you want to work here?": "I love the mission",
        "Preferred Location": "New York",
        "Willing to relocate": "true",
        "Resume": str(FIXTURES_DIR / "sample_resume.txt"),
    })


def test_submit_does_not_raise(browser):
    from agent.browser import PlaywrightFormFiller

    filler = PlaywrightFormFiller(browser)
    filler.submit(_fixture_url("greenhouse_form.html"), {
        "Full Name": "Jane Doe",
        "Why do you want to work here?": "I love the mission",
        "Preferred Location": "New York",
        "Willing to relocate": "true",
        "Resume": str(FIXTURES_DIR / "sample_resume.txt"),
    })


def test_fetch_fields_raises_form_unparseable_for_empty_form(browser):
    from agent.browser import PlaywrightPageFetcher
    from agent.models import FormUnparseable

    fetcher = PlaywrightPageFetcher(browser)

    with pytest.raises(FormUnparseable):
        fetcher.fetch_fields(_fixture_url("empty_form.html"))


def test_fill_raises_form_unparseable_when_no_labels_match(browser):
    from agent.browser import PlaywrightFormFiller
    from agent.models import FormUnparseable

    filler = PlaywrightFormFiller(browser)

    with pytest.raises(FormUnparseable):
        filler.fill(_fixture_url("greenhouse_form.html"), {"Nonexistent Field": "value"})


def test_submit_raises_form_unparseable_when_no_labels_match(browser):
    from agent.browser import PlaywrightFormFiller
    from agent.models import FormUnparseable

    filler = PlaywrightFormFiller(browser)

    with pytest.raises(FormUnparseable):
        filler.submit(_fixture_url("greenhouse_form.html"), {"Nonexistent Field": "value"})
