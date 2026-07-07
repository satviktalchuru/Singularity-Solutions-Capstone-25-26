// Ashby adapter. Ashby application forms use real, visible labels (no heavy
// obfuscation like Workday), so the generic label-matching sweep already
// does most of the work here. The one Ashby-specific quirk: Ashby usually
// asks for a single "Name" field rather than separate first/last name
// inputs, which FIELD_SYNONYMS in fillUtils.js doesn't cover (it only knows
// "first name"/"last name" as separate concepts) -- handled explicitly below.
(function () {
  function matches(url) {
    return /ashbyhq\.com$/i.test(url.hostname) || /jobs\.ashbyhq\.com/i.test(url.hostname);
  }

  function fillFullNameField(profile) {
    const { firstName, lastName } = profile.personal;
    if (!firstName && !lastName) return 0;
    const fullName = `${firstName} ${lastName}`.trim();
    let filled = 0;
    document.querySelectorAll("input[type='text'], input:not([type])").forEach((input) => {
      if (!window.FillUtils.isVisible(input) || input.value) return;
      const label = window.FillUtils.getLabelText(input).toLowerCase().trim();
      if (label === "name" || label === "full name") {
        window.FillUtils.setNativeValue(input, fullName);
        filled++;
      }
    });
    return filled;
  }

  async function fillResume(resumeFile) {
    const input = document.querySelector(
      "input[type='file'][accept*='pdf' i], input[type='file'][name*='resume' i]"
    ) || [...document.querySelectorAll("input[type='file']")].find((el) => {
      const label = window.FillUtils.getLabelText(el).toLowerCase();
      return label.includes("resume") || label.includes("cv");
    });
    if (input) return await window.FillUtils.fillFileInput(input, resumeFile);
    return false;
  }

  async function fill(profile, resumeFile) {
    let filled = fillFullNameField(profile);
    if (await fillResume(resumeFile)) filled++;
    filled += await window.FillUtils.fillGenerically(document.body, profile, resumeFile);
    return filled;
  }

  window.AshbyAdapter = { id: "ashby", matches, fill };
})();
