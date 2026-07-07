// Extracts plain text from a PDF File entirely client-side using the
// vendored pdf.js build (src/lib/pdf.min.js + pdf.worker.min.js). No network
// request, no server -- the file never leaves the browser.
//
// Loaded only on the options page (not in content scripts), after
// pdf.min.js, which defines the global `pdfjsLib`.

(function () {
  if (window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      chrome.runtime.getURL("src/lib/pdf.worker.min.js");
  }

  async function extractTextFromPdf(file) {
    const buffer = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data: buffer }).promise;
    const pageTexts = [];
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const content = await page.getTextContent();
      // pdf.js gives us individual text runs with position info but no
      // reliable line breaks; grouping by vertical position (`transform[5]`,
      // the y-coordinate) reconstructs lines well enough for the regex-based
      // parser in resumeParser.js to work with.
      const lines = new Map();
      for (const item of content.items) {
        const y = Math.round(item.transform[5]);
        if (!lines.has(y)) lines.set(y, []);
        lines.get(y).push(item.str);
      }
      const sortedY = [...lines.keys()].sort((a, b) => b - a); // top to bottom
      pageTexts.push(sortedY.map((y) => lines.get(y).join(" ")).join("\n"));
    }
    return pageTexts.join("\n\n");
  }

  window.PdfExtract = { extractTextFromPdf };
})();
