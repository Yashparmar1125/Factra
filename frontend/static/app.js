function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function highlightWords(explanation, suspiciousWords) {
  if (!Array.isArray(suspiciousWords) || suspiciousWords.length === 0) return escapeHtml(explanation);
  let html = escapeHtml(explanation);
  suspiciousWords.forEach((w) => {
    if (!w) return;
    const escaped = w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(escaped, "gi");
    html = html.replace(re, (match) => `<mark class="bg-primary/20 text-primary px-1 rounded">${match}</mark>`);
  });
  return html;
}

function setModeUI(mode) {
  const textBox = document.getElementById("textBox");
  const urlBox = document.getElementById("urlBox");
  if (mode === "url") {
    textBox.classList.add("hidden");
    urlBox.classList.remove("hidden");
  } else {
    urlBox.classList.add("hidden");
    textBox.classList.remove("hidden");
  }
}

function updateGauge(score) {
  const g = document.getElementById("gauge");
  const v = document.getElementById("scoreValue");
  const num = Number(score);
  
  // Animate the number
  let current = 0;
  const step = Math.ceil(num / 30) || 1;
  const interval = setInterval(() => {
    current += step;
    if (current >= num) {
      current = num;
      clearInterval(interval);
    }
    v.textContent = current;
  }, 20);

  // Update CSS variables for visual gauge
  g.style.setProperty("--p", num);
  // Anthropic accents: Green (#788c5d), Yellow (#f59e0b), Orange (#d97757)
  const color = num > 75 ? "#788c5d" : num > 40 ? "#f59e0b" : "#d97757";
  g.style.setProperty("--c", color);
}

function updateHistorySidebar(newScan = null) {
  const container = document.getElementById("historyItems");
  let history = JSON.parse(localStorage.getItem("fnd_history") || "[]");

  if (newScan) {
    history.unshift(newScan);
    history = history.slice(0, 10); // Keep last 10
    localStorage.setItem("fnd_history", JSON.stringify(history));
  }

  if (history.length === 0) {
    container.innerHTML = `<p class="text-slate-500 text-sm italic py-4 text-center">No recent scans</p>`;
    return;
  }

  container.innerHTML = history.map((item, idx) => `
    <div class="p-4 bg-white/40 dark:bg-black/20 hover:bg-white/60 dark:hover:bg-black/40 border border-white/30 dark:border-white/5 rounded-2xl cursor-pointer mb-3 transition-all hover:shadow-lg group" onclick="loadFromHistory(${idx})">
      <div class="font-bold text-slate-800 dark:text-white text-[13px] mb-1.5 truncate group-hover:text-primary transition-colors tracking-wide">${escapeHtml(item.text.slice(0, 40))}...</div>
      <div class="text-[10px] uppercase font-black tracking-widest text-slate-400 dark:text-slate-500 flex items-center gap-2">
        <span class="${item.prediction.toLowerCase() === 'fake' ? 'text-danger flex items-center gap-1 before:content-[\'\'] before:w-1.5 before:h-1.5 before:bg-danger before:rounded-full before:animate-pulse' : 'text-success'}">${item.prediction}</span> • ${item.score}%
      </div>
    </div>
  `).join("");
}

window.loadFromHistory = function(idx) {
  const history = JSON.parse(localStorage.getItem("fnd_history") || "[]");
  const item = history[idx];
  if (!item) return;
  
  // Create a pseudo-data object that matches what the dashboard expects
  const data = {
    prediction: item.prediction,
    credibility_score: item.score,
    confidence: item.confidence || 0.999, // Fallback for old items
    explanation: item.explanation || "Historical data record.",
    suspicious_words: item.suspicious || [],
    text: item.text,
    tone: item.tone,
    comp: item.comp,
    obj: item.obj,
    web_consensus: item.web
  };

  document.getElementById("mode").value = "text";
  setModeUI("text");
  document.getElementById("textInput").value = item.text;
  
  renderReport(data);
};

function checkHandOff() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("handoff") !== "1") return false;

  const data = {
    prediction: params.get("p"),
    credibility_score: params.get("s"),
    confidence: params.get("c"),
    explanation: params.get("e"),
    suspicious_words: params.get("k") ? params.get("k").split(",") : [],
    text: params.get("t"),
    tone: params.get("tone"),
    comp: params.get("comp"),
    obj: params.get("obj"),
    web_consensus: null
  };

  if (params.get("web")) {
    try {
      data.web_consensus = JSON.parse(params.get("web"));
    } catch (e) {
      console.warn("Could not parse web_consensus data from URL (likely truncated).");
    }
  }

  renderReport(data);
  return true;
}

function renderReport(data) {
  const welcome = document.getElementById("welcomeMessage");
  if (welcome) welcome.classList.add("hidden");

  const report = document.getElementById("resultsContainer");
  if (report) report.classList.remove("hidden");

  // Populate results
  updateGauge(data.credibility_score);
  
  const conf = (Number(data.confidence) * 100).toFixed(2);
  const confBar = document.getElementById("confBar");
  const confValue = document.getElementById("confValue");
  if (confBar) confBar.style.width = `${conf}%`;
  if (confValue) confValue.textContent = `${conf}%`;

  const pred = String(data.prediction).toUpperCase();
  const badg = document.getElementById("predBadge");
  if (badg) {
    badg.textContent = pred;
    const predColor = pred.toLowerCase() === "fake" ? "text-danger border-danger/40 bg-danger/10 shadow-[0_0_15px_rgba(244,63,94,0.3)]" : "text-success border-success/40 bg-success/10 shadow-[0_0_15px_rgba(16,185,129,0.3)]";
    badg.className = `inline-block px-4 py-1.5 font-black text-xs tracking-widest rounded-lg border backdrop-blur-md transition-all ${predColor}`;
  }

  const expEl = document.getElementById("explanationText");
  if (expEl) {
    const explanationHtml = highlightWords(data.explanation || "", data.suspicious_words || []);
    expEl.innerHTML = explanationHtml;
  }

  const kList = document.getElementById("keywordsList");
  if (kList) {
    if (data.suspicious_words?.length > 0) {
      kList.innerHTML = data.suspicious_words.map(w => `<span class="inline-block bg-white/50 dark:bg-white/5 border border-white/40 dark:border-white/10 text-slate-800 dark:text-slate-300 text-xs px-3 py-1.5 rounded-lg shadow-sm font-bold tracking-wide transition-all hover:scale-105 hover:border-danger/40">${escapeHtml(w)}</span>`).join("");
    } else {
      kList.innerHTML = `<span class="text-slate-400 dark:text-slate-500 text-xs font-bold tracking-widest uppercase py-2">0 Anomalies Detected.</span>`;
    }
  }

  // Advanced Analysis (safe fallbacks)
  const toneVal = document.getElementById("toneValue");
  const compVal = document.getElementById("complexityValue");
  const objVal = document.getElementById("objectivityValue");

  if (toneVal) toneVal.textContent = data.tone || "Neutral";
  if (compVal) compVal.textContent = data.comp || "Medium";
  if (objVal) objVal.textContent = data.obj || "Objective";

  // Web Consensus populate
  const cData = data.web_consensus;
  if (cData) {
    const cCircle = document.getElementById("consensusCircle");
    const cText = document.getElementById("consensusText");
    const cStatus = document.getElementById("consensusStatus");
    const cSummary = document.getElementById("consensusSummary");
    const cSources = document.getElementById("consensusSources");

    const score = Math.round(cData.consensus_score || 0);
    if (cCircle) cCircle.style.strokeDasharray = `${score}, 100`;
    if (cText) cText.textContent = `${score}%`;
    if (cStatus) {
      cStatus.textContent = cData.status === "Success" ? "Verified via Search" : cData.status;
      const statusColor = score > 60 ? "text-success" : score > 30 ? "text-yellow-500" : "text-danger";
      cStatus.className = `text-sm font-bold uppercase tracking-wider ${statusColor}`;
    }
    if (cSummary) {
      cSummary.textContent = score > 70 
        ? "Global indexing match density extremely high." 
        : score > 30 ? "Partial footprint detected. Exercise caution." 
        : "Null footprint. Claim likely isolated or fabricated.";
    }

    if (cSources && cData.sources) {
      cSources.innerHTML = cData.sources.map(s => `
        <div class="mb-4 p-4 bg-white/50 dark:bg-black/30 rounded-xl border border-white/40 dark:border-white/5 backdrop-blur-sm block transition-all hover:shadow-lg hover:border-primary/30 group">
          <div class="flex justify-between items-start mb-2">
            <a href="${s.link}" target="_blank" class="text-primary dark:text-accent font-black text-[13px] tracking-wide line-clamp-1 flex-1 group-hover:text-blue-500 transition-colors">${escapeHtml(s.title)}</a>
            ${s.reputable ? '<span class="text-[8px] font-black tracking-widest text-success border border-success/30 bg-success/10 px-2 py-1 rounded shadow-[0_0_8px_rgba(16,185,129,0.2)] ml-3 shrink-0">VERIFIED</span>' : ""}
          </div>
          <div class="text-xs text-slate-600 dark:text-slate-400 leading-relaxed line-clamp-2">${escapeHtml(s.snippet)}</div>
        </div>
      `).join("");
    }
  }

  const factBtn = document.getElementById("factCheckBtn");
  if (factBtn) {
    factBtn.onclick = () => {
      const query = encodeURIComponent((data.text || "").slice(0, 50));
      window.open(`https://www.google.com/search?q=fact+check+${query}`, "_blank");
    };
  }

  // Save to history with V2 enrichment for future instant-load
  updateHistorySidebar({
    text: data.text || "Hand-off from extension",
    prediction: data.prediction,
    score: data.credibility_score,
    confidence: data.confidence,
    explanation: data.explanation,
    suspicious: data.suspicious_words,
    date: new Date().toLocaleTimeString(),
    tone: data.tone,
    comp: data.comp,
    obj: data.obj,
    web: data.web_consensus
  });
}

document.addEventListener("DOMContentLoaded", () => {
  updateHistorySidebar();
  const hasHandoff = checkHandOff();
  
  // Load demo data if the dashboard was opened natively without history
  if (!hasHandoff && JSON.parse(localStorage.getItem("fnd_history") || "[]").length === 0) {
    renderReport({
      prediction: "Fake",
      credibility_score: 12,
      confidence: 0.985,
      explanation: "This article contains multiple sensationalist claims without citing credible sources. The vocabulary relies heavily on emotive language rather than objective facts, a pattern highly correlated with generated misinformation.",
      suspicious_words: ["shocking", "conspiracy", "hidden truth", "they don't want you to know"],
      text: "A demo report on a suspicious article. Use the Factra extension to run real analyses!",
      tone: "Sensational",
      comp: "Basic",
      obj: "Highly Biased",
      web_consensus: {
        status: "Failed",
        consensus_score: 5,
        sources: [
          { title: "Snopes Fact Check", link: "#", snippet: "This widespread claim has been thoroughly debunked by multiple agencies.", reputable: true },
          { title: "Unknown Blog", link: "#", snippet: "I heard from a friend that this conspiracy is 100% true.", reputable: false }
        ]
      }
    });
  }

  const themeToggle = document.getElementById('themeToggle');
  themeToggle?.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    const isDark = document.documentElement.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });
});
