// Profile builder page logic: resume intake (PDF or pasted text) -> local
// parse -> editable review form -> save to chrome.storage.local.

let pendingResumeFile = null; // { name, type, dataUrl } queued for saving

function setByPath(obj, path, value) {
  const parts = path.split(".");
  let node = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    node[parts[i]] = node[parts[i]] || {};
    node = node[parts[i]];
  }
  node[parts[parts.length - 1]] = value;
}

function getByPath(obj, path) {
  return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
}

function fillSimpleFields(profile) {
  document.querySelectorAll("#profileForm [name]").forEach((el) => {
    const value = getByPath(profile, el.name);
    if (value != null) el.value = value;
  });
  document.getElementById("skillsInput").value = (profile.skills || []).join(", ");
}

function readSimpleFields(profile) {
  document.querySelectorAll("#profileForm [name]").forEach((el) => {
    setByPath(profile, el.name, el.value);
  });
  profile.skills = document.getElementById("skillsInput").value
    .split(",").map((s) => s.trim()).filter(Boolean);
  return profile;
}

function renderRepeatable(containerId, templateId, items, fields) {
  const container = document.getElementById(containerId);
  const template = document.getElementById(templateId);
  container.innerHTML = "";
  (items.length ? items : [{}]).forEach((item) => addRow(container, template, item, fields));
}

function addRow(container, template, item, fields) {
  const node = template.content.firstElementChild.cloneNode(true);
  fields.forEach((field) => {
    const input = node.querySelector(`[data-field="${field}"]`);
    if (!input) return;
    if (input.type === "checkbox") input.checked = !!item[field];
    else input.value = item[field] || "";
  });
  node.querySelector(".remove").addEventListener("click", () => node.remove());
  container.appendChild(node);
}

function collectRepeatable(containerId, fields) {
  const rows = document.querySelectorAll(`#${containerId} .repeat-row`);
  const results = [];
  rows.forEach((row) => {
    const entry = {};
    let hasValue = false;
    fields.forEach((field) => {
      const input = row.querySelector(`[data-field="${field}"]`);
      if (!input) return;
      const value = input.type === "checkbox" ? input.checked : input.value;
      entry[field] = value;
      if (value) hasValue = true;
    });
    if (hasValue) results.push(entry);
  });
  return results;
}

const EDUCATION_FIELDS = ["school", "degree", "field", "gpa", "startYear", "endYear"];
const EXPERIENCE_FIELDS = ["title", "company", "location", "startDate", "endDate", "current", "description"];

async function applyParsedProfile(parsed) {
  const existing = await window.Storage.getProfile();
  // Merge: parsed resume data fills in fields, but never silently overwrites
  // ones the user already reviewed/edited and are non-empty.
  const merged = {
    personal: { ...parsed.personal, ...Object.fromEntries(
      Object.entries(existing.personal).filter(([, v]) => v)) },
    education: existing.education.length ? existing.education : parsed.education,
    experience: existing.experience.length ? existing.experience : parsed.experience,
    skills: existing.skills.length ? existing.skills : parsed.skills,
    workAuth: existing.workAuth,
    eeo: existing.eeo,
    resumeText: parsed.resumeText,
  };
  populateForm(merged);
}

function populateForm(profile) {
  fillSimpleFields(profile);
  renderRepeatable("educationList", "educationRowTpl", profile.education, EDUCATION_FIELDS);
  renderRepeatable("experienceList", "experienceRowTpl", profile.experience, EXPERIENCE_FIELDS);
}

async function renderAnswerMemory() {
  const { answerMemory } = await window.Storage.getAll();
  const container = document.getElementById("answerMemoryList");
  const entries = Object.entries(answerMemory);
  if (!entries.length) {
    container.innerHTML = "<p class='hint'>No remembered answers yet.</p>";
    return;
  }
  container.innerHTML = "";
  entries.forEach(([question, { answer, site }]) => {
    const row = document.createElement("div");
    row.className = "answer-row";
    row.innerHTML = `
      <div>
        <div class="q">${question} <span class="hint">(${site || "unknown site"})</span></div>
        <div class="a">${answer}</div>
      </div>`;
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.addEventListener("click", async () => {
      const all = await window.Storage.getAll();
      delete all.answerMemory[question];
      await chrome.storage.local.set({ answerMemory: all.answerMemory });
      renderAnswerMemory();
    });
    row.appendChild(del);
    container.appendChild(row);
  });
}

function init() {
  document.getElementById("pasteToggle").addEventListener("click", () => {
    const ta = document.getElementById("resumeTextArea");
    ta.style.display = ta.style.display === "none" ? "block" : "none";
  });

  document.getElementById("parseBtn").addEventListener("click", async () => {
    const status = document.getElementById("parseStatus");
    status.textContent = "Parsing...";
    try {
      const fileInput = document.getElementById("resumeFileInput");
      let text = document.getElementById("resumeTextArea").value.trim();

      if (fileInput.files[0]) {
        const file = fileInput.files[0];
        text = await window.PdfExtract.extractTextFromPdf(file);
        const dataUrl = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(file);
        });
        pendingResumeFile = { name: file.name, type: file.type, dataUrl };
      }

      if (!text) {
        status.textContent = "Upload a PDF or paste text first.";
        return;
      }

      const parsed = window.ResumeParser.parseResumeText(text);
      await applyParsedProfile(parsed);
      status.textContent = "Parsed — review the fields below, then Save.";
    } catch (err) {
      console.error(err);
      status.textContent = `Couldn't parse that file (${err.message}). Try pasting the text instead.`;
    }
  });

  document.querySelectorAll("[data-add]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const kind = btn.dataset.add;
      if (kind === "education") {
        addRow(document.getElementById("educationList"),
              document.getElementById("educationRowTpl"), {}, EDUCATION_FIELDS);
      } else {
        addRow(document.getElementById("experienceList"),
              document.getElementById("experienceRowTpl"), {}, EXPERIENCE_FIELDS);
      }
    });
  });

  document.getElementById("profileForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const profile = readSimpleFields({ personal: {}, workAuth: {}, eeo: {} });
    profile.education = collectRepeatable("educationList", EDUCATION_FIELDS);
    profile.experience = collectRepeatable("experienceList", EXPERIENCE_FIELDS);
    profile.resumeText = (await window.Storage.getProfile()).resumeText || "";

    await window.Storage.saveProfile(profile);
    if (pendingResumeFile) {
      await window.Storage.saveResumeFile(pendingResumeFile);
      pendingResumeFile = null;
    }
    const status = document.getElementById("saveStatus");
    status.textContent = "Saved.";
    setTimeout(() => (status.textContent = ""), 2500);
  });

  window.Storage.getProfile().then(populateForm);
  renderAnswerMemory();
}

document.addEventListener("DOMContentLoaded", init);
