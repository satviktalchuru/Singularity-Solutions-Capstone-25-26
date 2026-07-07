// LinkedIn adapter. Scoped strictly to the "Easy Apply" modal -- LinkedIn's
// DOM is full of unrelated inputs (search bar, post composer, comment boxes)
// and this must never touch those. If the modal isn't open, this adapter
// reports nothing to fill rather than fill the wrong thing.
//
// Safety note (applies to every adapter, called out here because LinkedIn's
// flow is the most button-heavy): this only fills fields. It never clicks
// Next/Review/Submit -- you always drive the actual submission.
(function () {
  function matches(url) {
    return /(^|\.)linkedin\.com$/i.test(url.hostname);
  }

  function findModal() {
    return document.querySelector(
      ".jobs-easy-apply-modal, [data-test-modal-id='easy-apply-modal'], div.artdeco-modal[role='dialog']"
    );
  }

  async function fillResume(modal, resumeFile) {
    const input = modal.querySelector("input[type='file']");
    if (input) return await window.FillUtils.fillFileInput(input, resumeFile);
    return false;
  }

  async function fill(profile, resumeFile) {
    const modal = findModal();
    if (!modal) {
      throw new Error("Open the Easy Apply modal first, then click Fill again.");
    }
    let filled = 0;
    if (await fillResume(modal, resumeFile)) filled++;
    filled += await window.FillUtils.fillGenerically(modal, profile, resumeFile);
    return filled;
  }

  window.LinkedInAdapter = { id: "linkedin", matches, fill };
})();
