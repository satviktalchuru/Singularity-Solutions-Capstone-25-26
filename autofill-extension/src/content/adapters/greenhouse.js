// Greenhouse adapter. The classic Greenhouse hosted job board
// (boards.greenhouse.io / job-boards.greenhouse.io) uses a small set of
// stable element IDs across every company's postings, since it's the same
// embedded form template. EEO questions (if present) use their own stable
// IDs too, but are left unfilled unless the corresponding profile.eeo field
// is explicitly set by the user -- see fillEeo() below.
(function () {
  function matches(url) {
    return /(^|\.)greenhouse\.io$/i.test(url.hostname);
  }

  const TEXT_FIELD_MAP = [
    { id: "first_name", path: "personal.firstName" },
    { id: "last_name", path: "personal.lastName" },
    { id: "email", path: "personal.email" },
    { id: "phone", path: "personal.phone" },
  ];

  function resolvePath(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
  }

  function fillKnownFields(profile) {
    let filled = 0;
    for (const { id, path } of TEXT_FIELD_MAP) {
      const el = document.getElementById(id);
      const value = resolvePath(profile, path);
      if (el && value && window.FillUtils.isVisible(el) && !el.value) {
        window.FillUtils.setNativeValue(el, value);
        filled++;
      }
    }
    return filled;
  }

  function fillEeo(profile) {
    const map = [
      { id: "gender", value: profile.eeo.gender },
      { id: "race_ethnicity", value: profile.eeo.race },
      { id: "veteran_status", value: profile.eeo.veteranStatus },
      { id: "disability_status", value: profile.eeo.disabilityStatus },
    ];
    let filled = 0;
    for (const { id, value } of map) {
      if (!value) continue; // never guess EEO answers -- only fill if the user set one
      const select = document.getElementById(id);
      if (!select || select.tagName !== "SELECT") continue;
      const option = [...select.options].find(
        (o) => o.textContent.trim().toLowerCase() === value.toLowerCase()
      );
      if (option) {
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        filled++;
      }
    }
    return filled;
  }

  async function fillResume(resumeFile) {
    const input = document.getElementById("resume") || document.querySelector("input[type='file']");
    if (input) return await window.FillUtils.fillFileInput(input, resumeFile);
    return false;
  }

  async function fill(profile, resumeFile) {
    let filled = fillKnownFields(profile);
    filled += fillEeo(profile);
    if (await fillResume(resumeFile)) filled++;
    filled += await window.FillUtils.fillGenerically(document.body, profile, resumeFile);
    return filled;
  }

  window.GreenhouseAdapter = { id: "greenhouse", matches, fill };
})();
