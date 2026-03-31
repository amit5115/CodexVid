/**
 * Premium learning UI: /api/codexvid/upload → teaching pack + video + chat.
 */
(function () {
  const $ = (sel, el = document) => el.querySelector(sel);

  const screens = {
    upload: $("#screen-upload"),
    processing: $("#screen-processing"),
    workspace: $("#screen-workspace"),
  };

  const elErrUpload = $("#err-upload");
  const elVideo = $("#learnVideo");
  const elTeachTabs = $("#teachTabs");
  const elPanelLearn = $("#panel-learn");
  const elPanelChat = $("#panel-chat");
  const elTeachingBody = $("#teaching-body");
  const elChatLog = $("#chat-log");
  const elChatForm = $("#chat-form");
  const elChatInput = $("#chat-input");
  const elModel = $("#learn-model");
  const elThemeToggle = $("#theme-toggle");
  const elDropzoneHint = $("#dropzone-hint");
  const elProcessingProgress = $("#processing-progress");
  const elProcessingStage = $("#processing-stage");

  let sessionId = null;
  let chatBusy = false;
  let chaptersData = [];       // populated after upload
  let segmentContext = null;   // {title, start, end, text} — active segment for scoped chat
  const segmentCache = {};     // {[chapterIndex]: {answer, mode, meta}} — avoids re-fetching
  let processingTimer = null;

  function startProcessingAnimation() {
    if (!elProcessingProgress || !elProcessingStage) return;
    const stages = [
      "Preparing upload…",
      "Reading video stream…",
      "Transcribing speech…",
      "Building smart chunks…",
      "Generating lesson + quiz…",
    ];
    let idx = 0;
    let progress = 8;
    elProcessingProgress.style.width = "8%";
    elProcessingStage.textContent = stages[0];
    clearInterval(processingTimer);
    processingTimer = setInterval(() => {
      idx = (idx + 1) % stages.length;
      progress = Math.min(progress + 12, 92);
      elProcessingProgress.style.width = `${progress}%`;
      elProcessingStage.textContent = stages[idx];
    }, 900);
  }

  function stopProcessingAnimation() {
    clearInterval(processingTimer);
    processingTimer = null;
    if (elProcessingProgress) elProcessingProgress.style.width = "100%";
  }

  function showScreen(name) {
    Object.values(screens).forEach((s) => s.classList.remove("active"));
    screens[name].classList.add("active");
    document.body.classList.toggle("is-workspace", name === "workspace");
    document.body.classList.toggle("is-processing", name === "processing");
    if (name === "processing") startProcessingAnimation();
    else stopProcessingAnimation();
  }

  function bindPremiumInteractions() {
    document.querySelectorAll(".interactive-surface").forEach((el) => {
      el.addEventListener("mousemove", (ev) => {
        const rect = el.getBoundingClientRect();
        const mx = ((ev.clientX - rect.left) / rect.width) * 100;
        const my = ((ev.clientY - rect.top) / rect.height) * 100;
        el.style.setProperty("--mx", `${mx}%`);
        el.style.setProperty("--my", `${my}%`);
      });
    });

    if (elChatInput) {
      const resize = () => {
        elChatInput.style.height = "auto";
        elChatInput.style.height = `${Math.min(elChatInput.scrollHeight, 140)}px`;
      };
      elChatInput.addEventListener("input", resize);
      resize();
    }
  }

  function parseMmss(label) {
    const parts = String(label).trim().split(":");
    if (parts.length === 2) {
      return parseInt(parts[0], 10) * 60 + parseFloat(parts[1]);
    }
    if (parts.length === 3) {
      return (
        parseInt(parts[0], 10) * 3600 +
        parseInt(parts[1], 10) * 60 +
        parseFloat(parts[2])
      );
    }
    return 0;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatAssistantHtml(text) {
    let t = escapeHtml(text);
    t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const lines = t.split("\n");
    const parts = lines.map((line) => {
      const m = line.match(
        /^📍\s*(\d{1,2}:\d{2})\s*[–-]\s*(\d{1,2}:\d{2})/
      );
      if (m) {
        const start = parseMmss(m[1]);
        return `<button type="button" class="ts-jump" data-start="${String(start)}">${line}</button>`;
      }
      return line;
    });
    return parts.join("<br>");
  }

  function bindTimestampClicks(container) {
    container.querySelectorAll(".ts-jump").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sec = parseFloat(String(btn.getAttribute("data-start") || "0"));
        seekVideo(sec);
      });
    });
  }

  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "msg user";
    div.textContent = text;
    elChatLog.appendChild(div);
    elChatLog.scrollTop = elChatLog.scrollHeight;
  }

  function appendAssistantMessage(text, mode, meta) {
    meta = meta || {};
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    const body = document.createElement("div");
    body.className = "body";
    let html = formatAssistantHtml(text);
    const hasSegment =
      (meta.chunks_used | 0) > 0 &&
      typeof meta.timestamp_start === "number" &&
      !Number.isNaN(meta.timestamp_start);
    if (hasSegment) {
      const ts = meta.timestamp_start;
      const te =
        typeof meta.timestamp_end === "number" ? meta.timestamp_end : ts;
      const line = `📍 Jump to segment (${formatChapterTime(ts)} – ${formatChapterTime(te)})`;
      html =
        `<button type="button" class="ts-jump" data-start="${String(ts)}">${escapeHtml(
          line
        )}</button><br>` + html;
    }
    if (meta.key_points && meta.key_points.length) {
      html +=
        "<p class='meta'><strong>Key points</strong></p><ul class='key-points'>" +
        meta.key_points.map((k) => `<li>${escapeHtml(k)}</li>`).join("") +
        "</ul>";
    }
    body.innerHTML = html;
    wrap.appendChild(body);
    if (mode) {
      const b = document.createElement("div");
      b.className = "badge-mode";
      b.textContent = "Teaching mode: " + mode;
      wrap.appendChild(b);
    }
    elChatLog.appendChild(wrap);
    bindTimestampClicks(wrap);
    elChatLog.scrollTop = elChatLog.scrollHeight;
  }

  function formatChapterTime(sec) {
    if (sec == null || Number.isNaN(Number(sec))) return "";
    const s = Math.max(0, Math.floor(Number(sec)));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  function renderTeaching(teaching) {
    if (!teaching) {
      elTeachingBody.innerHTML = "<p class='muted'>No teaching pack.</p>";
      return;
    }
    const ch = teaching.chapters || [];
    const kt = teaching.key_takeaways || [];
    const qz = teaching.quiz || [];

    // Store for active chapter tracking and segment chat
    chaptersData = ch.map((c, i) => ({
      title: c.title || "Section " + (i + 1),
      start: c.start != null ? Number(c.start) : 0,
      end: c.end != null ? Number(c.end) : 0,
      text: c.explanation || "",
    }));

    let html = "";
    if (ch.length) {
      html += "<h3 class='teaching-section-title'>Chapters</h3>";
      ch.forEach((c, i) => {
        const startSec = c.start != null ? Number(c.start) : 0;
        const endSec = c.end != null ? Number(c.end) : 0;
        const steps = (c.step_by_step || [])
          .map((s) => `<li>${escapeHtml(s)}</li>`)
          .join("");
        const timeBtn =
          c.start != null && c.end != null
            ? `<button type="button" class="chapter-time-btn" data-start="${startSec}" data-end="${endSec}">${escapeHtml(formatChapterTime(c.start))} – ${escapeHtml(formatChapterTime(c.end))}</button>`
            : "";
        html += `<div class="chapter" data-index="${i}" data-start="${startSec}" data-end="${endSec}">
          <h3>${escapeHtml(c.title || "Section " + (i + 1))}</h3>
          <div class="chapter-actions">
            ${timeBtn}
            <button type="button" class="btn-segment-chat" data-index="${i}">&#x1F4AC; Ask about this</button>
          </div>
          <p>${escapeHtml(c.explanation || "")}</p>
          ${steps ? `<p class="meta">Steps</p><ul>${steps}</ul>` : ""}
          ${c.analogy ? `<p><strong>Analogy:</strong> ${escapeHtml(c.analogy)}</p>` : ""}
          ${c.example ? `<p><strong>Example:</strong> ${escapeHtml(c.example)}</p>` : ""}
          ${c.common_mistake ? `<p><strong>Common mistake:</strong> ${escapeHtml(c.common_mistake)}</p>` : ""}
        </div>`;
      });
    }
    if (kt.length) {
      html += "<h3>Key takeaways</h3><ul class='takeaways'>";
      kt.forEach((t) => {
        html += `<li>${escapeHtml(t)}</li>`;
      });
      html += "</ul>";
    }
    if (qz.length) {
      html += "<h3>Quiz</h3>";
      qz.forEach((q, i) => {
        html += `<div class="quiz-item"><strong>Q${i + 1}.</strong> ${escapeHtml(q.question || "")}<br/><span class='meta'>Answer: ${escapeHtml(q.answer || "")}</span></div>`;
      });
    }
    elTeachingBody.innerHTML = html || "<p>No structured content returned.</p>";
    bindChapterInteractions();
  }

  // — Chapter interactions: seek on timestamp click, open segment chat —
  function bindChapterInteractions() {
    // Timestamp seek buttons
    elTeachingBody.querySelectorAll(".chapter-time-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const sec = parseFloat(btn.getAttribute("data-start") || "0");
        seekVideo(sec);
      });
    });
    // Segment chat buttons
    elTeachingBody.querySelectorAll(".btn-segment-chat").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.getAttribute("data-index"), 10);
        openSegmentChat(idx);
      });
    });
  }

  function seekVideo(sec) {
    if (!elVideo || Number.isNaN(sec)) return;
    elVideo.currentTime = sec;
    elVideo.play().catch(() => {});
  }

  // — Active chapter tracking via video timeupdate —
  let lastActiveIdx = -1;
  elVideo.addEventListener("timeupdate", () => {
    const t = elVideo.currentTime;
    let activeIdx = -1;
    for (let i = 0; i < chaptersData.length; i++) {
      if (t >= chaptersData[i].start && t <= chaptersData[i].end) {
        activeIdx = i;
        break;
      }
    }
    if (activeIdx === lastActiveIdx) return;
    lastActiveIdx = activeIdx;
    elTeachingBody.querySelectorAll(".chapter").forEach((el) => {
      const idx = parseInt(el.getAttribute("data-index"), 10);
      el.classList.toggle("active", idx === activeIdx);
    });
    // Auto-scroll active chapter into view
    if (activeIdx >= 0) {
      const activeEl = elTeachingBody.querySelector(`.chapter[data-index="${activeIdx}"]`);
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  });

  // — Segment-scoped chat —
  const elSegCtx = document.getElementById("segment-context");
  const elSegCtxText = document.getElementById("segment-context-text");
  const elSegCtxClear = document.getElementById("segment-context-clear");

  /** Switch the right panel to the Chat tab. */
  function switchToChat() {
    elTeachTabs.querySelectorAll("button").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    const bChat = elTeachTabs.querySelector('button[data-tab="chat"]');
    bChat.classList.add("active");
    bChat.setAttribute("aria-selected", "true");
    elPanelLearn.classList.remove("active");
    elPanelChat.classList.add("active");
  }

  /** Enable/disable all "Ask about this" buttons. */
  function setSegmentBtnsDisabled(disabled) {
    elTeachingBody.querySelectorAll(".btn-segment-chat").forEach((btn) => {
      btn.disabled = disabled;
    });
  }

  /** Append an animated "Thinking…" placeholder; returns its element. */
  function appendThinkingMessage() {
    const el = document.createElement("div");
    el.className = "msg assistant msg--thinking";
    el.id = "chat-msg-thinking";
    el.innerHTML =
      `<div class="thinking-dots"><span></span><span></span><span></span></div>` +
      `<span class="thinking-label">Thinking\u2026</span>`;
    elChatLog.appendChild(el);
    elChatLog.scrollTop = elChatLog.scrollHeight;
    return el;
  }

  /** Remove the thinking placeholder and render the real assistant message. */
  function replaceThinking(text, mode, meta) {
    const el = document.getElementById("chat-msg-thinking");
    if (el) el.remove();
    appendAssistantMessage(text, mode, meta);
  }

  /**
   * Auto-generate a segment explanation:
   *  • shows user message + thinking bubble immediately
   *  • fires the chat API scoped to this segment's transcript
   *  • caches the result so a second click is instant
   *  • disables all "Ask about this" buttons while in-flight
   */
  async function generateSegmentExplanation(idx, ch) {
    chatBusy = true;
    setSegmentBtnsDisabled(true);

    appendUserMessage("Explain this segment in simple terms");
    appendThinkingMessage();

    // Scoped query: the LLM sees only this segment's transcript
    const query =
      `[SEGMENT CONTEXT — explain ONLY this segment, do not summarise the whole video]\n` +
      `Segment: "${ch.title}" (${formatChapterTime(ch.start)} – ${formatChapterTime(ch.end)})\n` +
      `Transcript:\n${ch.text}\n\n` +
      `Task: Explain this segment clearly and simply for a beginner. ` +
      `Use only the transcript above.`;

    console.log("[CodexVid] generateSegmentExplanation — ACTIVE SEGMENT:", ch.title,
      `(${formatChapterTime(ch.start)} – ${formatChapterTime(ch.end)})`);

    try {
      const res = await fetch("/api/codexvid/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query,
          model: elModel.value,
          // Tell the backend to restrict FAISS hits to this time window
          segment_start: ch.start,
          segment_end: ch.end,
        }),
      });
      const data = await res.json().catch(() => ({}));
      const answer = res.ok ? (data.answer || "") : (data.error || "Request failed.");
      const mode = data.mode || null;
      // Always use the SELECTED SEGMENT's timestamps — never the FAISS-retrieved ones.
      // This guarantees the jump button always points to the correct chapter.
      const meta = {
        timestamp_start: ch.start,
        timestamp_end: ch.end,
        key_points: data.key_points,
        chunks_used: data.chunks_used || 1,
      };
      console.log("[CodexVid] Explanation received for segment:", ch.title,
        "| jump → ", formatChapterTime(ch.start), "–", formatChapterTime(ch.end));
      // Cache so re-clicking the same chapter is instant
      segmentCache[idx] = { answer, mode, meta };
      replaceThinking(answer, mode, meta);
    } catch (e) {
      replaceThinking(
        "Sorry, something went wrong. Please try again.",
        null,
        {}
      );
    } finally {
      chatBusy = false;
      setSegmentBtnsDisabled(false);
    }
  }

  /**
   * Called when user clicks "Ask about this" on a chapter card.
   * Switches to chat, sets context banner, seeks video, then either
   * serves a cached explanation or fires a new one automatically.
   */
  function openSegmentChat(idx) {
    if (idx < 0 || idx >= chaptersData.length) return;
    if (chatBusy) return;   // debounce: ignore while a fetch is in-flight

    const ch = chaptersData[idx];
    segmentContext = ch;

    // Update the context banner
    elSegCtxText.textContent = `Chatting about: "${ch.title}" (${formatChapterTime(ch.start)} – ${formatChapterTime(ch.end)})`;
    elSegCtx.style.display = "";

    switchToChat();
    seekVideo(ch.start);
    elChatInput.placeholder = `Ask a follow-up about "${ch.title}"…`;

    // Empty transcript guard
    if (!ch.text || !ch.text.trim()) {
      appendUserMessage("Explain this segment in simple terms");
      appendAssistantMessage("No transcript available for this segment.", null, {});
      return;
    }

    // Serve from cache if available (instant, no API call)
    if (segmentCache[idx]) {
      const c = segmentCache[idx];
      appendUserMessage("Explain this segment in simple terms");
      appendAssistantMessage(c.answer, c.mode, c.meta);
      return;
    }

    // Auto-fire: no typing required
    generateSegmentExplanation(idx, ch);
  }

  function clearSegmentContext() {
    segmentContext = null;
    elSegCtx.style.display = "none";
    elChatInput.placeholder = "Ask a question about the video…";
  }

  if (elSegCtxClear) {
    elSegCtxClear.addEventListener("click", clearSegmentContext);
  }

  elTeachTabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tab]");
    if (!btn) return;
    elTeachTabs.querySelectorAll("button").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    const tab = btn.getAttribute("data-tab");
    elPanelLearn.classList.toggle("active", tab === "learn");
    elPanelChat.classList.toggle("active", tab === "chat");
    // Clear segment context when switching to lesson tab
    if (tab === "learn") clearSegmentContext();
  });

  const elDropzone = document.getElementById("dropzone");
  const elFileVideo = document.getElementById("file-video");
  if (elDropzone && elFileVideo) {
    elFileVideo.addEventListener("change", () => {
      const file = elFileVideo.files && elFileVideo.files[0];
      if (elDropzoneHint) {
        elDropzoneHint.textContent = file
          ? `Selected: ${file.name}`
          : "or click to browse · MP4, WebM, MOV, MKV…";
      }
    });
    ["dragenter", "dragover"].forEach((ev) => {
      elDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elDropzone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      elDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elDropzone.classList.remove("is-dragover");
      });
    });
    elDropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (!dt || !dt.files || !dt.files.length) return;
      const f = dt.files[0];
      try {
        const xfer = new DataTransfer();
        xfer.items.add(f);
        elFileVideo.files = xfer.files;
      } catch (_) {
        return;
      }
      elFileVideo.dispatchEvent(new Event("change", { bubbles: true }));
    });
    elDropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        elFileVideo.click();
      }
    });
  }

  if (elThemeToggle) {
    const stored = localStorage.getItem("codexvid_theme");
    if (stored === "vivid") document.body.classList.add("theme-vivid");
    elThemeToggle.addEventListener("click", () => {
      document.body.classList.toggle("theme-vivid");
      localStorage.setItem(
        "codexvid_theme",
        document.body.classList.contains("theme-vivid") ? "vivid" : "default"
      );
    });
  }

  bindPremiumInteractions();

  $("#btn-start-upload").addEventListener("click", async () => {
    elErrUpload.textContent = "";
    const yt = ($("#youtube-url") && $("#youtube-url").value.trim()) || "";
    const fileInput = $("#file-video");
    const file = fileInput.files && fileInput.files[0];
    if (!yt && !file) {
      elErrUpload.textContent = "Paste a YouTube link or choose a video file.";
      return;
    }

    showScreen("processing");
    const fd = new FormData();
    if (yt) {
      fd.append("youtube_url", yt);
    } else if (file) {
      fd.append("file", file);
    }
    fd.append("model", elModel.value);
    fd.append("whisper_model", "base");
    fd.append("language", "en");

    try {
      const res = await fetch("/api/codexvid/upload", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || res.statusText);
      }
      sessionId = data.session_id;
      clearSegmentContext();
      lastActiveIdx = -1;
      // Wipe per-segment cache — new video, new answers
      Object.keys(segmentCache).forEach((k) => delete segmentCache[k]);
      renderTeaching(data.teaching);
      elVideo.src = `/api/codexvid/sessions/${sessionId}/video`;
      elChatLog.innerHTML = "";
      appendAssistantMessage(
        "Ask anything about this video. I answer only from what was said — click any 📍 line to jump.",
        null
      );
      showScreen("workspace");
      const bLearn = elTeachTabs.querySelector('button[data-tab="learn"]');
      const bChat = elTeachTabs.querySelector('button[data-tab="chat"]');
      bLearn.classList.add("active");
      bLearn.setAttribute("aria-selected", "true");
      bChat.classList.remove("active");
      bChat.setAttribute("aria-selected", "false");
      elPanelLearn.classList.add("active");
      elPanelChat.classList.remove("active");
    } catch (e) {
      elErrUpload.textContent = String(e.message || e);
      showScreen("upload");
    }
  });

  elChatForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = elChatInput.value.trim();
    if (!q || !sessionId || chatBusy) return;
    chatBusy = true;
    appendUserMessage(q);
    elChatInput.value = "";

    // Capture segment at submit-time — prevents stale reads if the user
    // switches to a different chapter while this request is in-flight.
    const activeSeg = segmentContext;

    // Build query: prepend segment context so the LLM scopes its answer
    let query = q;
    if (activeSeg) {
      query =
        `[SEGMENT CONTEXT — answer ONLY using this segment]\n` +
        `Segment: "${activeSeg.title}" (${formatChapterTime(activeSeg.start)} – ${formatChapterTime(activeSeg.end)})\n` +
        `Transcript: ${activeSeg.text}\n\n` +
        `Question: ${q}`;
      console.log("[CodexVid] Manual question — ACTIVE SEGMENT:", activeSeg.title,
        `(${formatChapterTime(activeSeg.start)} – ${formatChapterTime(activeSeg.end)})`);
    }

    try {
      const res = await fetch("/api/codexvid/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query,
          model: elModel.value,
          // Restrict FAISS hits to the selected segment's time window
          ...(activeSeg ? { segment_start: activeSeg.start, segment_end: activeSeg.end } : {}),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        appendAssistantMessage(data.error || "Request failed.", null, {});
      } else {
        // When a segment is active, always show the SELECTED segment's timestamps
        // in the jump button — never let FAISS hits from other parts of the video
        // hijack the pointer.
        const meta = activeSeg
          ? {
              timestamp_start: activeSeg.start,
              timestamp_end: activeSeg.end,
              key_points: data.key_points,
              chunks_used: data.chunks_used || 1,
            }
          : {
              timestamp_start: data.timestamp_start,
              timestamp_end: data.timestamp_end,
              key_points: data.key_points,
              chunks_used: data.chunks_used,
            };
        console.log("[CodexVid] Response — jump →",
          formatChapterTime(meta.timestamp_start), "–", formatChapterTime(meta.timestamp_end),
          "| segment:", activeSeg ? activeSeg.title : "(full video)");
        appendAssistantMessage(data.answer || "", data.mode, meta);
      }
    } catch (e) {
      appendAssistantMessage(String(e.message || e), null, {});
    } finally {
      chatBusy = false;
    }
  });
})();
