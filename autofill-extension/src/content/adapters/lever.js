// Lever adapter. jobs.lever.co postings use a consistent `name` attribute
// scheme across every posting (same hosted application template), including
// a "urls[LinkedIn]" / "urls[GitHub]" / "urls[Portfolio]" convention for
// profile links.
(function () {
  function matches(url) {
    return /(^|\.)jobs\.lever\.co$/i.test(url.hostname) || /lever\.co$/i.test(url.hostname);
  }

  const NAME_FIELD_MAP = [
    { name: "name", path: "personal.fullName" }, // handled specially below
    { name: "email", path: "personal.email" },
    { name: "phone", path: "personal.phone" },
    { name: "urls[LinkedIn]", path: "personal.linkedin" },
    { name: "urls[GitHub]", path: "personal.github" },
    { name: "urls[Portfolio]", path: "personal.portfolio" },
  ];

  function resolvePath(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
  }

  function fillKnownFields(profile) {
    const withFullName = {
      ...profile,
      personal: {
        ...profile.personal,
        fullName: `${profile.personal.firstName || ""} ${profile.personal.lastName || ""}`.trim(),
      },
    };
    let filled = 0;
    for (const { name, path } of NAME_FIELD_MAP) {
      const el = document.querySelector(`[name="${CSS.escape(name)}"]`);
      const value = resolvePath(withFullName, path);
      if (el && value && window.FillUtils.isVisible(el) && !el.value) {
        window.FillUtils.setNativeValue(el, value);
        filled++;
      }
    }
    return filled;
  }

  async function fillResume(resumeFile) {
    const input = document.querySelector("input[name='resume']") ||
      document.querySelector("input[type='file']");
    if (input) return await window.FillUtils.fillFileInput(input, resumeFile);
    return false;
  }

  async function fill(profile, resumeFile) {
    let filled = fillKnownFields(profile);
    if (await fillResume(resumeFile)) filled++;
    filled += await window.FillUtils.fillGenerically(document.body, profile, resumeFile);
    return filled;
  }

  window.LeverAdapter = { id: "lever", matches, fill };
})();
