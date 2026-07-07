// Workday adapter. Workday career sites (myworkdayjobs.com) run the same
// underlying product across every tenant, so form fields consistently carry
// `data-automation-id` attributes -- these selectors target the ones seen
// most often across tenants (contact info, address, phone). Workday also
// renders real visible <label> text for accessibility, so the shared
// generic label-matching sweep (FillUtils.fillGenerically) runs afterward
// and picks up anything the explicit selectors miss or that drifted on a
// particular tenant's build.
//
// NOTE: because this can't be verified against a live authenticated Workday
// candidate account from here, treat the explicit `data-automation-id`
// selectors below as best-effort -- if a field doesn't fill, check devtools
// for that tenant's actual attribute and add it to FIELD_MAP.
(function () {
  function matches(url) {
    return /myworkdayjobs\.com|\.workday\.com/i.test(url.hostname);
  }

  // automation-id -> profile path. Each entry finds the *input* inside the
  // element carrying that data-automation-id (Workday usually wraps the
  // real <input> in a labeled container div with the automation id).
  const FIELD_MAP = [
    { automationId: "legalNameSection_firstName", path: "personal.firstName" },
    { automationId: "legalNameSection_lastName", path: "personal.lastName" },
    { automationId: "firstName", path: "personal.firstName" },
    { automationId: "lastName", path: "personal.lastName" },
    { automationId: "email", path: "personal.email" },
    { automationId: "phone-number", path: "personal.phone" },
    { automationId: "addressSection_addressLine1", path: "personal.address" },
    { automationId: "addressSection_city", path: "personal.city" },
    { automationId: "addressSection_postalCode", path: "personal.zip" },
  ];

  function resolvePath(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
  }

  function fillKnownFields(profile) {
    let filled = 0;
    for (const { automationId, path } of FIELD_MAP) {
      const value = resolvePath(profile, path);
      if (!value) continue;
      const container = document.querySelector(`[data-automation-id="${automationId}"]`);
      if (!container) continue;
      const input = container.matches("input") ? container : container.querySelector("input");
      if (input && window.FillUtils.isVisible(input) && !input.value) {
        window.FillUtils.setNativeValue(input, value);
        filled++;
      }
    }
    return filled;
  }

  async function fillResume(resumeFile) {
    const input = document.querySelector(
      '[data-automation-id="file-upload-input-ref"], input[type="file"][accept*="pdf" i]'
    );
    if (input) return await window.FillUtils.fillFileInput(input, resumeFile);
    return false;
  }

  async function fill(profile, resumeFile) {
    let filled = fillKnownFields(profile);
    if (await fillResume(resumeFile)) filled++;
    filled += await window.FillUtils.fillGenerically(document.body, profile, resumeFile);
    return filled;
  }

  window.WorkdayAdapter = { id: "workday", matches, fill };
})();
