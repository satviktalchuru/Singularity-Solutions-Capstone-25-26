// Best-effort resume -> structured profile parser. Runs entirely client-side
// (options page), on text either pasted directly or extracted from an
// uploaded PDF via pdf.js. This is deliberately conservative: it's meant to
// pre-fill the profile review form, not to be perfect. Every field it guesses
// is shown back to the user for confirmation/editing before anything is
// saved (see options.js).

(function () {
  const EMAIL_RE = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
  const PHONE_RE = /(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/;
  const LINKEDIN_RE = /(https?:\/\/)?(www\.)?linkedin\.com\/[^\s,)]+/i;
  const GITHUB_RE = /(https?:\/\/)?(www\.)?github\.com\/[^\s,)]+/i;
  const URL_RE = /(https?:\/\/)?(www\.)?[a-z0-9-]+\.[a-z]{2,}(\/[^\s,)]*)?/i;

  const SECTION_HEADERS = {
    experience: /^(work\s+)?experience|employment\s+history|professional\s+experience$/i,
    education: /^education(al background)?$/i,
    skills: /^(technical\s+)?skills|technologies$/i,
  };

  function normalizeUrl(url) {
    if (!url) return "";
    return url.startsWith("http") ? url : `https://${url}`;
  }

  function guessName(lines) {
    // Heuristic: the name is almost always the first non-empty line that
    // isn't an email/phone/URL and doesn't look like a section header, and
    // is short (a real name, not a sentence).
    for (const line of lines.slice(0, 6)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (EMAIL_RE.test(trimmed) || PHONE_RE.test(trimmed) || URL_RE.test(trimmed)) continue;
      if (trimmed.split(/\s+/).length > 5) continue;
      if (/^(resume|curriculum vitae|cv)$/i.test(trimmed)) continue;
      const words = trimmed.split(/\s+/).filter(Boolean);
      if (words.length >= 2 && words.length <= 4) {
        return { firstName: words[0].replace(/[^a-zA-Z'-]/g, ""),
                lastName: words[words.length - 1].replace(/[^a-zA-Z'-]/g, "") };
      }
    }
    return { firstName: "", lastName: "" };
  }

  function splitIntoSections(lines) {
    const sections = { header: [], experience: [], education: [], skills: [], other: [] };
    let current = "header";
    for (const rawLine of lines) {
      const line = rawLine.trim();
      let matchedHeader = null;
      for (const [name, re] of Object.entries(SECTION_HEADERS)) {
        if (re.test(line)) { matchedHeader = name; break; }
      }
      if (matchedHeader) { current = matchedHeader; continue; }
      sections[current].push(rawLine);
    }
    return sections;
  }

  function parseSkills(skillLines) {
    const text = skillLines.join(", ");
    return text
      .split(/[,•|\n]/)
      .map((s) => s.trim())
      .filter((s) => s.length > 1 && s.length < 40);
  }

  // Experience/education entries are notoriously free-form across resume
  // templates. This groups consecutive non-empty lines into blocks
  // (paragraphs separated by blank lines) and does light field-guessing
  // within each block, rather than trying to fully parse arbitrary layouts.
  function groupIntoBlocks(lines) {
    const blocks = [];
    let current = [];
    for (const line of lines) {
      if (line.trim() === "") {
        if (current.length) blocks.push(current);
        current = [];
      } else {
        current.push(line.trim());
      }
    }
    if (current.length) blocks.push(current);
    return blocks;
  }

  const DATE_RANGE_RE = /((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{1,2}\/\d{4}|\d{4})\s*(?:-|–|to)\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{1,2}\/\d{4}|\d{4}|present|current)/i;

  function parseExperience(expLines) {
    return groupIntoBlocks(expLines).slice(0, 10).map((block) => {
      const text = block.join(" ");
      const dateMatch = text.match(DATE_RANGE_RE);
      const isCurrent = /present|current/i.test(dateMatch?.[2] || "");
      return {
        title: block[0] || "",
        company: block[1] || "",
        location: "",
        startDate: dateMatch?.[1] || "",
        endDate: isCurrent ? "" : (dateMatch?.[2] || ""),
        current: isCurrent,
        description: block.slice(2).join(" "),
      };
    });
  }

  const DEGREE_RE = /(bachelor|master|ph\.?d|associate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)[^,]*/i;

  function parseEducation(eduLines) {
    return groupIntoBlocks(eduLines).slice(0, 6).map((block) => {
      const text = block.join(" ");
      const dateMatch = text.match(DATE_RANGE_RE) || text.match(/\d{4}/);
      // Search line-by-line (not the whole joined block) so the degree match
      // stops at that line's end instead of swallowing GPA/date lines that
      // follow it with no comma in between.
      const degreeLine = block.find((line) => DEGREE_RE.test(line));
      const degreeMatch = degreeLine ? degreeLine.match(DEGREE_RE) : null;
      return {
        school: block[0] || "",
        degree: degreeMatch ? degreeMatch[0].trim() : "",
        field: "",
        startYear: "",
        endYear: Array.isArray(dateMatch) ? (dateMatch[2] || dateMatch[0] || "") : "",
        gpa: (text.match(/gpa[:\s]*([\d.]+)/i) || [])[1] || "",
      };
    });
  }

  // Entry point: raw resume text -> partial profile object (same shape as
  // storage.js's `profile`, minus fields resumes never contain like eeo).
  function parseResumeText(text) {
    const lines = text.split(/\r?\n/);
    const name = guessName(lines);
    const email = (text.match(EMAIL_RE) || [""])[0];
    const phone = (text.match(PHONE_RE) || [""])[0];
    const linkedin = normalizeUrl((text.match(LINKEDIN_RE) || [""])[0]);
    const github = normalizeUrl((text.match(GITHUB_RE) || [""])[0]);

    const sections = splitIntoSections(lines);

    return {
      personal: {
        firstName: name.firstName,
        lastName: name.lastName,
        email, phone, linkedin, github,
        address: "", city: "", state: "", zip: "", country: "",
        portfolio: "", website: "",
      },
      education: parseEducation(sections.education),
      experience: parseExperience(sections.experience),
      skills: parseSkills(sections.skills),
      workAuth: {},
      eeo: {},
      resumeText: text,
    };
  }

  window.ResumeParser = { parseResumeText };
})();
