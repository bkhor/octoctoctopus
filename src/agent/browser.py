from dataclasses import dataclass

from playwright.sync_api import Browser, Locator, Page

from .models import FieldSpec, FormUnparseable

RESOLVED_LABEL_THRESHOLD = 0.5

EXTRACT_SCRIPT = """
() => {
  function getLabelForId(id) {
    if (!id) return null;
    const lbl = document.querySelector(`label[for="${CSS.escape(id)}"]`);
    return lbl ? lbl.textContent.trim() : null;
  }
  function nearestText(el) {
    const wrapper = el.closest('label') || el.parentElement;
    if (!wrapper) return null;
    const clone = wrapper.cloneNode(true);
    clone.querySelectorAll('input, textarea, select').forEach(n => n.remove());
    const text = clone.textContent.trim();
    return text || null;
  }
  function resolveLabel(el, index) {
    let label = getLabelForId(el.id);
    let resolved = label !== null && label !== '';
    if (!resolved) {
      const ariaLabel = el.getAttribute('aria-label');
      if (ariaLabel) { label = ariaLabel.trim(); resolved = true; }
    }
    if (!resolved) {
      const labelledBy = el.getAttribute('aria-labelledby');
      if (labelledBy) {
        const ref = document.getElementById(labelledBy);
        if (ref && ref.textContent.trim()) { label = ref.textContent.trim(); resolved = true; }
      }
    }
    if (!resolved) {
      const near = nearestText(el);
      if (near) { label = near; resolved = true; }
    }
    if (!resolved) {
      const placeholder = el.getAttribute('placeholder');
      if (placeholder) { label = placeholder.trim(); resolved = true; }
    }
    if (!resolved) {
      label = `field_${index}`;
    }
    return { label, resolved };
  }
  function groupLabelFor(radios, optionTexts) {
    let ancestor = radios[0].parentElement;
    while (ancestor && !radios.every(r => ancestor.contains(r))) {
      ancestor = ancestor.parentElement;
    }
    if (!ancestor) return null;
    const optionTextSet = new Set(optionTexts);
    const labels = Array.from(ancestor.querySelectorAll('label'));
    for (const lbl of labels) {
      const text = lbl.textContent.trim();
      if (text && !optionTextSet.has(text)) return text;
    }
    return null;
  }

  const allInputs = Array.from(document.querySelectorAll('input, textarea, select'));
  const results = [];
  let index = 0;

  const radioGroups = {};
  for (const el of allInputs) {
    const isRadio = el.tagName.toLowerCase() === 'input' && (el.getAttribute('type') || '').toLowerCase() === 'radio';
    if (isRadio && el.name) {
      if (!radioGroups[el.name]) radioGroups[el.name] = [];
      radioGroups[el.name].push(el);
    }
  }

  const handledRadios = new Set();
  for (const name in radioGroups) {
    const radios = radioGroups[name];
    const optionResults = radios.map((r, i) => {
      const { label } = resolveLabel(r, index + i);
      return { element: r, text: label };
    });
    const optionTexts = optionResults.map(o => o.text);
    const groupLabel = groupLabelFor(radios, optionTexts);
    const resolved = groupLabel !== null;
    const finalLabel = groupLabel !== null ? groupLabel : `field_${index}`;
    const radioOptions = [];
    for (const opt of optionResults) {
      opt.element.setAttribute('data-octo-index', String(index));
      radioOptions.push({ text: opt.text, octo_index: index });
      handledRadios.add(opt.element);
      index++;
    }
    results.push({
      label: finalLabel,
      field_type: 'radio',
      options: radioOptions.map(o => o.text),
      required: radios.some(r => r.hasAttribute('required')),
      label_resolved: resolved,
      octo_index: null,
      radio_options: radioOptions,
    });
  }

  for (const el of allInputs) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (handledRadios.has(el)) continue;
    if (tag === 'input' && ['hidden', 'submit', 'button', 'reset', 'radio'].includes(type)) continue;

    let fieldType = 'text';
    if (tag === 'textarea') fieldType = 'textarea';
    else if (tag === 'select') fieldType = 'select';
    else if (type === 'checkbox') fieldType = 'checkbox';
    else if (type === 'file') fieldType = 'file';

    const { label, resolved } = resolveLabel(el, index);

    const options = tag === 'select'
      ? Array.from(el.querySelectorAll('option')).map(o => o.textContent.trim())
      : [];

    el.setAttribute('data-octo-index', String(index));
    results.push({
      label,
      field_type: fieldType,
      options,
      required: el.hasAttribute('required'),
      label_resolved: resolved,
      octo_index: index,
      radio_options: null,
    });
    index++;
  }

  return results;
}
"""


@dataclass
class ExtractedField:
    spec: FieldSpec
    locator: Locator | None
    label_resolved: bool
    option_locators: dict[str, Locator] | None = None


def _extract_fields(page: Page) -> list[ExtractedField]:
    raw = page.evaluate(EXTRACT_SCRIPT)
    extracted = []
    for item in raw:
        spec = FieldSpec(
            label=item["label"],
            field_type=item["field_type"],
            options=item["options"],
            required=item["required"],
        )
        if item["field_type"] == "radio":
            option_locators = {
                opt["text"]: page.locator(f'[data-octo-index="{opt["octo_index"]}"]')
                for opt in item["radio_options"]
            }
            extracted.append(ExtractedField(
                spec=spec, locator=None, label_resolved=item["label_resolved"],
                option_locators=option_locators,
            ))
        else:
            locator = page.locator(f'[data-octo-index="{item["octo_index"]}"]')
            extracted.append(ExtractedField(spec=spec, locator=locator, label_resolved=item["label_resolved"]))
    return extracted


def _apply_value(locator: Locator, field_type: str, value: str) -> None:
    if field_type in ("text", "textarea"):
        locator.fill(value)
    elif field_type == "select":
        locator.select_option(label=value)
    elif field_type == "checkbox":
        if value.lower() in ("true", "yes", "1"):
            locator.check()
        else:
            locator.uncheck()
    elif field_type == "file":
        locator.set_input_files(value)


def _apply_radio_value(extracted: ExtractedField, value: str) -> bool:
    option_locator = extracted.option_locators.get(value)
    if option_locator is None:
        return False
    option_locator.check()
    return True


class PlaywrightPageFetcher:
    def __init__(self, browser: Browser) -> None:
        self._browser = browser

    def fetch_fields(self, url: str) -> list[FieldSpec]:
        context = self._browser.new_context()
        try:
            page = context.new_page()
            page.goto(url)
            extracted = _extract_fields(page)
            if not extracted:
                raise FormUnparseable("no fields could be extracted from the page")
            resolved_count = sum(1 for e in extracted if e.label_resolved)
            if resolved_count / len(extracted) < RESOLVED_LABEL_THRESHOLD:
                raise FormUnparseable(f"only {resolved_count}/{len(extracted)} fields had a resolvable label")
            return [e.spec for e in extracted]
        finally:
            context.close()


class PlaywrightFormFiller:
    def __init__(self, browser: Browser) -> None:
        self._browser = browser

    def fill(self, url: str, field_values: dict[str, str]) -> None:
        context = self._browser.new_context()
        try:
            page = context.new_page()
            page.goto(url)
            extracted = _extract_fields(page)
            applied_count = 0
            for e in extracted:
                value = field_values.get(e.spec.label)
                if value is None:
                    continue
                if e.spec.field_type == "radio":
                    if _apply_radio_value(e, value):
                        applied_count += 1
                    continue
                _apply_value(e.locator, e.spec.field_type, value)
                applied_count += 1
            if extracted and applied_count == 0:
                raise FormUnparseable("no field values matched any extracted label")
        finally:
            context.close()

    def submit(self, url: str, field_values: dict[str, str]) -> None:
        context = self._browser.new_context()
        try:
            page = context.new_page()
            page.goto(url)
            extracted = _extract_fields(page)
            applied_count = 0
            for e in extracted:
                value = field_values.get(e.spec.label)
                if value is None:
                    continue
                if e.spec.field_type == "radio":
                    if _apply_radio_value(e, value):
                        applied_count += 1
                    continue
                _apply_value(e.locator, e.spec.field_type, value)
                applied_count += 1
            if extracted and applied_count == 0:
                raise FormUnparseable("no field values matched any extracted label; refusing to submit blind")
            page.locator('button[type="submit"], input[type="submit"]').first.click()
            page.wait_for_load_state("networkidle")
        finally:
            context.close()
