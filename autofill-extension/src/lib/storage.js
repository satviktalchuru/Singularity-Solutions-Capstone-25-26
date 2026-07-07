// Shared storage layer. Everything lives in chrome.storage.local -- nothing
// ever leaves the machine, there is no server, no sync, no account.
//
// Shape:
//   profile: {
//     personal: { firstName, lastName, email, phone, address, city, state,
//                 zip, country, linkedin, github, portfolio, website },
//     education: [{ school, degree, field, startYear, endYear, gpa }],
//     experience: [{ company, title, location, startDate, endDate,
//                    current, description }],
//     skills: [string],
//     workAuth: { authorizedToWorkIn, needsSponsorship },
//     eeo: { gender, race, veteranStatus, disabilityStatus },  // optional,
//           default "Prefer not to answer" -- never filled without the
//           corresponding profile field being explicitly set.
//     resumeText: string,        // parsed plain text, used for form fields
//                                 // that just want a pasted resume/summary
//   }
//   resumeFile: { name, type, dataUrl }   // for actual file-upload inputs
//   answerMemory: { [normalizedQuestion]: { answer, site, savedAt } }
//                                 // freeform Q&A the user has answered
//                                 // before (e.g. "Why do you want to work
//                                 // here?"), reused across applications.

const DEFAULTS = {
  profile: {
    personal: {}, education: [], experience: [], skills: [],
    workAuth: {}, eeo: {}, resumeText: "",
  },
  resumeFile: null,
  answerMemory: {},
};

async function getAll() {
  const data = await chrome.storage.local.get(Object.keys(DEFAULTS));
  return {
    profile: { ...DEFAULTS.profile, ...(data.profile || {}) },
    resumeFile: data.resumeFile || null,
    answerMemory: data.answerMemory || {},
  };
}

async function getProfile() {
  const { profile } = await getAll();
  return profile;
}

async function saveProfile(profile) {
  await chrome.storage.local.set({ profile });
}

async function saveResumeFile(resumeFile) {
  await chrome.storage.local.set({ resumeFile });
}

async function getResumeFile() {
  const { resumeFile } = await getAll();
  return resumeFile;
}

function normalizeQuestion(text) {
  return text.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();
}

async function rememberAnswer(question, answer, site) {
  const { answerMemory } = await getAll();
  const key = normalizeQuestion(question);
  if (!key) return;
  answerMemory[key] = { answer, site, savedAt: new Date().toISOString() };
  await chrome.storage.local.set({ answerMemory });
}

async function recallAnswer(question) {
  const { answerMemory } = await getAll();
  const key = normalizeQuestion(question);
  return answerMemory[key]?.answer ?? null;
}

async function isProfileComplete() {
  const p = await getProfile();
  return !!(p.personal.firstName && p.personal.lastName && p.personal.email);
}

// Exposed as a global (no bundler in this project -- plain <script> tags /
// content-script includes), so every consumer just calls window.Storage.*
window.Storage = {
  getAll, getProfile, saveProfile, saveResumeFile, getResumeFile,
  rememberAnswer, recallAnswer, normalizeQuestion, isProfileComplete,
};
