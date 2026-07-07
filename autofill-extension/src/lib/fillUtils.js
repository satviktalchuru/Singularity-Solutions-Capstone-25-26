// Shared DOM helpers used by every site adapter. Loaded before the adapters
// in manifest.json, exposed as window.FillUtils.
//
// The single hardest problem this file solves: Workday, LinkedIn, and Ashby
// all render forms through React (or something React-like). Doing
// `input.value = "x"` does NOT update their internal state -- React tracks
// values through its own synthetic event system and a plain value assignment
// is invisible to it, so the UI shows your text for a frame and then the
// framework stomps it back to empty on the next render. The fix is to call
// the *native* HTMLInputElement value setter (bypassing any setter React has
// patched onto the instance) and then dispatch a real InputEvent, which
// React's listener does pick up.

(function () {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Set a value on <input>/<textarea> in a way React (and friends) will see.
  function setNativeValue(element, value) {
    const proto = element.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const nativeSetter = Object.getOwnPropertyDescriptor(proto, "value").set;
    nativeSetter.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function setCheckbox(element, checked) {
    if (element.checked === checked) return;
    element.click(); // .click() fires the full event chain frameworks expect
  }

  // Given a list of same-name radio inputs, click whichever one's associated
  // label text best matches `valueText` (case-insensitive substring).
  function setRadioGroup(radios, valueText) {
    const target = valueText.toLowerCase();
    for (const radio of radios) {
      const label = getLabelText(radio).toLowerCase();
      if (label.includes(target) || target.includes(label)) {
        radio.click();
        return true;
      }
    }
    return false;
  }

  // Resolve the human-readable label for a form field, trying every common
  // association pattern in order of reliability.
  function getLabelText(element) {
    if (element.id) {
      const byFor = document.querySelector(`label[for="${CSS.escape(element.id)}"]`);
      if (byFor) return byFor.textContent.trim();
    }
    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel) return ariaLabel.trim();

    const labelledBy = element.getAttribute("aria-labelledby");
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map((id) => document.getElementById(id)?.textContent?.trim())
        .filter(Boolean);
      if (parts.length) return parts.join(" ");
    }

    const closestLabel = element.closest("label");
    if (closestLabel) return closestLabel.textContent.trim();

    // Walk up a few ancestors looking for the nearest preceding label-ish text
    // node (covers custom component libraries with no <label> at all).
    let node = element;
    for (let i = 0; i < 4 && node; i++) {
      node = node.parentElement;
      if (!node) break;
      const candidate = node.querySelector(
        "label, [class*='label' i], legend, [class*='question' i]"
      );
      if (candidate && candidate.textContent.trim()) return candidate.textContent.trim();
    }

    return element.getAttribute("placeholder") || element.name || "";
  }

  // Keyword -> profile-path scoring table used to map an arbitrary label to
  // a known field. Each entry is tried as a substring match against the
  // lowercased label; first match wins, ordered specific-before-generic.
  const FIELD_SYNONYMS = [
    { path: "personal.firstName", keywords: ["first name", "given name", "legal first"] },
    { path: "personal.lastName", keywords: ["last name", "family name", "surname", "legal last"] },
    { path: "personal.email", keywords: ["email"] },
    { path: "personal.phone", keywords: ["phone", "mobile", "telephone"] },
    { path: "personal.address", keywords: ["address line", "street address", "address 1"] },
    { path: "personal.city", keywords: ["city", "town"] },
    { path: "personal.state", keywords: ["state", "province", "region"] },
    { path: "personal.zip", keywords: ["zip", "postal"] },
    { path: "personal.country", keywords: ["country"] },
    { path: "personal.linkedin", keywords: ["linkedin"] },
    { path: "personal.github", keywords: ["github"] },
    { path: "personal.portfolio", keywords: ["portfolio", "personal website", "website"] },
    { path: "workAuth.authorizedToWorkIn", keywords: ["authorized to work", "legally authorized", "work authorization"] },
    { path: "workAuth.needsSponsorship", keywords: ["sponsorship", "visa sponsor"] },
  ];

  function resolveProfilePath(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
  }

  // Best-effort: given a label string, return the matching profile value (or
  // null if nothing in FIELD_SYNONYMS matches -- caller falls back to
  // answerMemory / leaves the field alone rather than guessing wrong).
  function matchProfileField(label, profile) {
    const lower = label.toLowerCase();
    for (const { path, keywords } of FIELD_SYNONYMS) {
      if (keywords.some((kw) => lower.includes(kw))) {
        const value = resolveProfilePath(profile, path);
        if (value !== undefined && value !== null && value !== "") return { path, value };
      }
    }
    return null;
  }

  // Simulates dropping a File object into a <input type="file">, since real
  // file inputs can't be set via .value for security reasons. `resumeFile`
  // is { name, type, dataUrl } as stored by storage.js.
  async function fillFileInput(input, resumeFile) {
    if (!resumeFile) return false;
    const res = await fetch(resumeFile.dataUrl);
    const blob = await res.blob();
    const file = new File([blob], resumeFile.name, { type: resumeFile.type });
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function isVisible(element) {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  }

  // Second-pass sweep: for any visible, empty, fillable field inside `root`
  // not already handled by a site-specific adapter, try (a) a known profile
  // field match, then (b) a remembered answer from a past application.
  // Deliberately does NOT touch checkboxes/radios other than a Yes/No work-
  // authorization group -- auto-checking "I agree to the terms" or consent
  // boxes on the user's behalf would be a genuinely bad idea, so those are
  // always left for the human to tick themselves.
  async function fillGenerically(root, profile, resumeFile) {
    let filled = 0;
    const fields = root.querySelectorAll(
      "input[type='text'], input[type='email'], input[type='tel'], " +
      "input:not([type]), textarea, select, input[type='file']"
    );

    for (const el of fields) {
      if (!isVisible(el)) continue;

      if (el.tagName === "INPUT" && el.type === "file") {
        if (await fillFileInput(el, resumeFile)) filled++;
        continue;
      }

      if (el.value) continue; // don't clobber anything already filled

      const label = getLabelText(el);
      if (!label) continue;

      const match = matchProfileField(label, profile);
      if (match) {
        if (el.tagName === "SELECT") {
          const option = [...el.options].find(
            (o) => o.textContent.trim().toLowerCase() === String(match.value).toLowerCase()
          );
          if (option) { el.value = option.value; el.dispatchEvent(new Event("change", { bubbles: true })); filled++; }
        } else {
          setNativeValue(el, match.value);
        }
        filled++;
        continue;
      }

      // No structured profile field matches -- this is a free-text/custom
      // question ("Why do you want to work here?", "Describe a challenge
      // you solved"). Check whether the user has answered it before.
      if ((el.tagName === "TEXTAREA" || el.tagName === "INPUT") && window.Storage) {
        const remembered = await window.Storage.recallAnswer(label);
        if (remembered) { setNativeValue(el, remembered); filled++; }
      }
    }
    return filled;
  }

  window.FillUtils = {
    sleep, setNativeValue, setCheckbox, setRadioGroup, getLabelText,
    matchProfileField, fillFileInput, isVisible, fillGenerically, FIELD_SYNONYMS,
  };
})();
