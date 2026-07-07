// Minimal service worker: just opens the profile page on first install so
// there's an obvious next step instead of a silent, empty extension icon.
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    chrome.runtime.openOptionsPage();
  }
});
