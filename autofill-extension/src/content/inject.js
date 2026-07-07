// Content script entry point. Picks the right adapter for the current page,
// runs it on demand (never automatically -- only when the popup's "Fill
// this page" button sends FILL_PAGE), and sets up "answer memory" learning
// so freeform questions you type an answer to once get reused automatically
// on future applications.

const ADAPTER_PRIORITY = [
  "WorkdayAdapter", "LinkedInAdapter", "AshbyAdapter",
  "GreenhouseAdapter", "LeverAdapter", "GenericAdapter",
];

function pickAdapter() {
  const url = new URL(window.location.href);
  for (const name of ADAPTER_PRIORITY) {
    const adapter = window[name];
    if (adapter && adapter.matches(url)) return adapter;
  }
  return null;
}

// After a fill pass, any textarea/text input still empty and labeled like a
// real question is worth remembering the *next* time the user fills it in
// by hand -- this is what makes recurring free-text questions ("Why do you
// want to work here?") get progressively less repetitive across companies.
function attachAnswerMemoryListeners(root, site) {
  const candidates = root.querySelectorAll("textarea, input[type='text'], input:not([type])");
  candidates.forEach((el) => {
    if (el.dataset.autofillMemoryAttached) return;
    el.dataset.autofillMemoryAttached = "1";
    el.addEventListener("blur", async () => {
      const value = el.value.trim();
      if (value.length < 8) return; // too short to be a meaningful custom answer
      const label = window.FillUtils.getLabelText(el);
      if (!label || label.length < 5) return;
      // Don't bother remembering things that are clearly structured profile
      // fields (name/email/etc.) -- only genuinely freeform questions.
      if (window.FillUtils.matchProfileField(label, { personal: {}, workAuth: {} })) return;
      await window.Storage.rememberAnswer(label, value, site);
    });
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "FILL_PAGE") return;

  (async () => {
    try {
      const adapter = pickAdapter();
      if (!adapter) {
        sendResponse({ ok: false, error: "No adapter matched this page." });
        return;
      }
      const profile = await window.Storage.getProfile();
      const resumeFile = await window.Storage.getResumeFile();
      const filledCount = await adapter.fill(profile, resumeFile);
      attachAnswerMemoryListeners(document.body, window.location.hostname);
      sendResponse({ ok: true, filledCount, adapter: adapter.id });
    } catch (err) {
      sendResponse({ ok: false, error: err.message });
    }
  })();

  return true; // keep the message channel open for the async response
});
