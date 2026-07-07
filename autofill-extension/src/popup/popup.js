async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function init() {
  const statusEl = document.getElementById("status");
  const fillBtn = document.getElementById("fillBtn");
  const siteInfo = document.getElementById("siteInfo");

  const complete = await window.Storage.isProfileComplete();
  if (!complete) {
    statusEl.textContent = "Profile incomplete — add your name and email first.";
    statusEl.className = "status warn";
  } else {
    statusEl.textContent = "Profile ready.";
    statusEl.className = "status ok";
  }

  const tab = await getActiveTab();
  if (tab?.url) {
    try {
      const host = new URL(tab.url).hostname;
      siteInfo.textContent = `This tab: ${host}`;
    } catch { /* non-http tab (chrome://, etc.) */ }
  }

  fillBtn.disabled = !complete;
  fillBtn.addEventListener("click", async () => {
    fillBtn.disabled = true;
    fillBtn.textContent = "Filling...";
    try {
      const response = await chrome.tabs.sendMessage(tab.id, { type: "FILL_PAGE" });
      if (response?.ok) {
        statusEl.textContent = `Filled ${response.filledCount} field(s) via the ${response.adapter} adapter.`;
        statusEl.className = "status ok";
      } else {
        statusEl.textContent = response?.error || "Nothing recognized on this page.";
        statusEl.className = "status warn";
      }
    } catch (err) {
      statusEl.textContent = "Couldn't reach this page (try reloading it first).";
      statusEl.className = "status warn";
    }
    fillBtn.disabled = false;
    fillBtn.textContent = "Fill this page";
  });

  document.getElementById("editBtn").addEventListener("click", () => {
    chrome.runtime.openOptionsPage();
  });
}

document.addEventListener("DOMContentLoaded", init);
