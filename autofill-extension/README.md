# Job Application Autofill

A Chrome extension that fills out job applications on LinkedIn, Workday,
Ashby, Greenhouse, Lever, and (best-effort) any other job board — from a
profile built off your resume. Everything runs and stays on your machine:
no server, no account, no sync, no data leaving the browser. Your resume
file and every profile field live in `chrome.storage.local` only.

## Install (unpacked, ~1 minute)

1. Open `chrome://extensions`
2. Toggle **Developer mode** on (top right)
3. Click **Load unpacked**, select this `autofill-extension/` folder
4. The options page opens automatically — that's step 1 below

## Set up your profile (one time)

1. On the options page: upload your resume PDF (parsed **locally in the
   browser** via a bundled copy of pdf.js — it never leaves your machine) or
   paste the text directly.
2. Click **Parse resume**. It fills in a best-effort guess at your name,
   email, phone, links, education, experience, and skills.
3. **Review and fix anything wrong** — resume layouts vary a lot, so treat
   the parse as a rough draft, not a final answer. This is the only step
   where accuracy matters; from here on autofill just replays whatever you
   confirm.
4. Fill in work authorization and (optional) EEO/demographic answers. EEO
   fields are only ever filled on an application if you've explicitly set
   them here — the extension never guesses or defaults them.
5. Click **Save profile**.

## Using it

Open any job application, click the extension icon, click **Fill this
page**. It fills whatever it recognizes; you review and submit yourself —
**the extension never clicks Submit/Apply/Next for you.**

- **LinkedIn**: open the Easy Apply modal first, then click Fill.
- **Workday / Ashby / Greenhouse / Lever**: fill covers most standard steps;
  multi-page flows may need Fill clicked again per page.
- **Anything else**: falls back to matching visible field labels against
  your profile — works reasonably well on standard forms, less so on heavily
  custom ones.

### It gets less repetitive over time

Freeform questions ("Why do you want to work here?") aren't something a
profile field can answer. The first time you type an answer to one and tab
away, it's remembered (matched by the question's own text, not the site) and
auto-filled the next time an application asks something similar. See/edit/
delete everything it's remembered on the options page under **Remembered
answers**.

## How it's built

```
manifest.json                  MV3, one extension, all sites
src/
  background.js                 opens the profile page on first install
  lib/
    storage.js                  chrome.storage.local wrapper (profile, resume file, answer memory)
    fillUtils.js                 shared DOM helpers -- see below
    resumeParser.js               resume text -> structured profile (regex heuristics)
    pdfExtract.js / pdf.min.js    local PDF -> text (vendored pdf.js, no network)
  popup/                        "Fill this page" button
  options/                      profile builder / review form
  content/
    inject.js                   picks the right adapter for the current page, wires answer-memory learning
    adapters/
      workday.js                 data-automation-id conventions + generic fallback
      linkedin.js                 scoped to the Easy Apply modal only
      ashby.js                    handles Ashby's single "Name" field convention
      greenhouse.js                standard first_name/last_name/email/... element IDs
      lever.js                     name="urls[LinkedIn]" etc. conventions
      generic.js                   pure label-matching fallback for everything else
```

**The hard technical problem**: Workday, LinkedIn, and Ashby render forms
through React. Setting `input.value = "x"` directly is invisible to React —
it tracks state through its own synthetic events, so the UI shows your text
for a frame and then snaps back to empty. `fillUtils.js`'s `setNativeValue`
works around this by calling the *native* `HTMLInputElement` value setter
(bypassing whatever setter React patched onto the element) and then
dispatching a real `InputEvent`, which React's listener does pick up. Verified
against an actual React-rendered controlled input during development —
confirmed React's own internal state updates, not just the raw DOM attribute.

**Safety by design**: the generic fallback and every adapter only ever fill
text/file inputs and answer memory. Checkboxes and consent/agreement boxes
are never auto-checked, EEO fields are never guessed, and nothing ever
clicks a submit/next/continue button — you always drive the actual
submission.

## Known limitations

- **Resume parsing is best-effort.** Every resume template is different;
  the review step exists because the parser *will* sometimes misplace a
  field. Always check it before saving.
- **Workday and LinkedIn selectors are based on well-documented, widely
  observed conventions**, not verified against every tenant's live build (no
  authenticated Workday candidate account or LinkedIn login was available
  while building this). If a field doesn't fill on a specific company's
  Workday tenant, open devtools, find that field's actual
  `data-automation-id`, and add it to `FIELD_MAP` in `workday.js`.
- **Multi-page application wizards** (common on Workday) need Fill clicked
  again on each page — this is intentional, not a bug, so you can review
  each page before moving on.
- The generic adapter (used for ATSes without a dedicated file — iCIMS,
  SmartRecruiters, Workable, custom career pages, etc.) is the least
  accurate since it has no site-specific selectors to lean on.

## Privacy

No network requests, no analytics, no remote code (the Content Security
Policy for extension pages is `script-src 'self'` — everything, including
PDF parsing, runs from bundled local files). Uninstalling the extension
deletes everything (`chrome.storage.local` is cleared automatically).
