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

  let sessionId = null;
  let chatBusy = false;

  function showScreen(name) {
    Object.values(screens).forEach((s) => s.classList.remove("active"));
    screens[name].classList.add("active");
    document.body.classList.toggle("is-workspace", name === "workspace");
    document.body.classList.toggle("is-processing", name === "processing");
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
        if (elVideo && !Number.isNaN(sec)) {
          elVideo.currentTime = sec;
          elVideo.play().catch(() => {});
        }
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
      const line = `📍 Jump to segment (${ts.toFixed(3)}s – ${te.toFixed(3)}s)`;
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

    let html = "";
    if (ch.length) {
      html += "<h3 class='teaching-section-title'>Chapters</h3>";
      ch.forEach((c, i) => {
        const steps = (c.step_by_step || [])
          .map((s) => `<li>${escapeHtml(s)}</li>`)
          .join("");
        const meta =
          c.start != null && c.end != null
            ? `<p class="meta chapter-time">${escapeHtml(formatChapterTime(c.start))} – ${escapeHtml(
                formatChapterTime(c.end)
              )}</p>`
            : "";
        html += `<div class="chapter">
          <h3>${escapeHtml(c.title || "Section " + (i + 1))}</h3>
          ${meta}
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
  });

  const elDropzone = document.getElementById("dropzone");
  const elFileVideo = document.getElementById("file-video");
  if (elDropzone && elFileVideo) {
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

    try {
      const res = await fetch("/api/codexvid/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query: q,
          model: elModel.value,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        appendAssistantMessage(data.error || "Request failed.", null);
      } else {
        appendAssistantMessage(data.answer || "", data.mode, {
          timestamp_start: data.timestamp_start,
          timestamp_end: data.timestamp_end,
          key_points: data.key_points,
          chunks_used: data.chunks_used,
        });
      }
    } catch (e) {
      appendAssistantMessage(String(e.message || e), null);
    } finally {
      chatBusy = false;
    }
  });
})();
