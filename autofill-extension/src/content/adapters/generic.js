// Fallback adapter for any job board without a dedicated adapter (Workable,
// iCIMS, SmartRecruiters, BambooHR, custom career pages, etc.). Pure label-
// matching against the whole page -- no site-specific selectors to lean on,
// so it's the least accurate adapter, but it's better than nothing and
// covers a long tail of ATSes without needing one file per site.
(function () {
  function matches() {
    return true; // always matches -- registered last, used only if nothing else claims the page
  }

  async function fill(profile, resumeFile) {
    return await window.FillUtils.fillGenerically(document.body, profile, resumeFile);
  }

  window.GenericAdapter = { id: "generic", matches, fill };
})();
