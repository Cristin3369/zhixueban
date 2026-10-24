const LS = {
  userId: "zhixueban.user_id",
  name: "zhixueban.name",
  examTrack: "zhixueban.exam_track",
  school: "zhixueban.school",
  major: "zhixueban.major",
  books: "zhixueban.books",
  bookList: "zhixueban.book_list",
  windows: "zhixueban.windows",
  onboarded: "zhixueban.onboarded",
};

const KINDS = ["英语一", "英语二", "政治", "专业课", "数学", "其他"];
const VERSION_RE = /(?:同济(?:大学)?)?第?\s*[0-9一二三四五六七八九十]+\s*版/;

function inferKind(title) {
  const t = title || "";
  if (/英语\s*[（(]?\s*[2二]|英二/.test(t)) return "英语二";
  if (/英语\s*[（(]?\s*[1一]|英一|^英语$/.test(t)) return "英语一";
  if (/政治|肖秀荣|肖爷爷|1000题/.test(t)) return "政治";
  if (/数学分析|解析几何|高等代数|抽象代数|实变|复变/.test(t)) return "专业课";
  if (/数学|高数|线代|概率|张宇|李永乐|汤家凤/.test(t)) return "数学";
  return "其他";
}

function normalizeKind(raw) {
  const s = raw || "";
  if (/英语\s*[（(]?\s*[2二]|英二/.test(s)) return "英语二";
  if (/英语|英一/.test(s)) return "英语一";
  if (/政治|肖/.test(s)) return "政治";
  if (/专业/.test(s)) return "专业课";
  if (/数学|高数/.test(s)) return "数学";
  return s || "其他";
}

function specialTitles(title, edition, kind) {
  if (kind === "英语一" || kind === "英语二") return [kind, ""];
  if (kind === "政治") {
    if (!title || title === "政治" || title === "考研政治" || /肖|1000/.test(title)) {
      return ["肖秀荣1000题", edition || ""];
    }
  }
  return [(title || "").trim(), (edition || "").trim()];
}

function parseOne(chunk) {
  const raw = (chunk || "").replace(/^[，,、;；\s]+|[，,、;；\s]+$/g, "");
  if (!raw) return null;
  const m = raw.match(/^(专业课|数学|英语[一二12]?|英语|政治)\s*[:：]?\s*(.*)$/);
  let kindPrefix = "";
  let rest = raw;
  if (m) {
    kindPrefix = normalizeKind(m[1]);
    rest = (m[2] || "").trim() || m[1];
  }
  let title = rest;
  let edition = "";
  const parts = rest.split(/[，,]/);
  if (parts.length > 1 && VERSION_RE.test(parts[1])) {
    title = parts[0].trim();
    edition = parts.slice(1).join("，").trim();
  } else {
    const vm = rest.match(VERSION_RE);
    if (vm && vm.index > 0) {
      title = rest.slice(0, vm.index).replace(/[，,、\s]+$/g, "");
      edition = rest.slice(vm.index).trim();
    }
  }
  const kind = kindPrefix || inferKind(title || rest);
  const spec = specialTitles(title || rest, edition, kind);
  if (!spec[0]) return null;
  return { title: spec[0], edition: spec[1], kind };
}

function parseBooksText(text) {
  if (!text || !String(text).trim()) return [];
  const blob = String(text).replace(/[；;\n]/g, "，");
  const parts = blob.split(/[，,、]/).map((p) => p.trim()).filter(Boolean);
  const merged = [];
  parts.forEach((p) => {
    const looksVer = VERSION_RE.test(p) && !/英语|政治|专业课|数学分析|解析几何/.test(p);
    if (looksVer && merged.length) merged[merged.length - 1] += "，" + p;
    else merged.push(p);
  });
  const seen = new Set();
  const out = [];
  merged.forEach((item) => {
    const parsed = parseOne(item);
    if (!parsed) return;
    const key = parsed.title + "|" + parsed.edition + "|" + parsed.kind;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(parsed);
  });
  return out;
}

function loadBookList() {
  try {
    const raw = localStorage.getItem(LS.bookList);
    if (raw) {
      const list = JSON.parse(raw);
      if (Array.isArray(list) && list.length) return list;
    }
  } catch (e) {
    /* ignore */
  }
  return parseBooksText(localStorage.getItem(LS.books) || "");
}

const state = {
  userId: localStorage.getItem(LS.userId) || "",
  displayName: localStorage.getItem(LS.name) || "",
  examTrack: localStorage.getItem(LS.examTrack) || "考研",
  school: localStorage.getItem(LS.school) || "",
  major: localStorage.getItem(LS.major) || "",
  books: localStorage.getItem(LS.books) || "",
  bookList: loadBookList(),
  windows: JSON.parse(localStorage.getItem(LS.windows) || "[]"),
  onboarded: localStorage.getItem(LS.onboarded) === "1",
  examDate: "",
  page: "chat",
  subject: "",
  sessionId: "",
  sending: false,
  pendingSwitch: null,
};

const $ = (id) => document.getElementById(id);

function persist() {
  localStorage.setItem(LS.userId, state.userId);
  localStorage.setItem(LS.name, state.displayName);
  localStorage.setItem(LS.examTrack, state.examTrack);
  localStorage.setItem(LS.school, state.school);
  localStorage.setItem(LS.major, state.major);
  localStorage.setItem(LS.books, state.bookList.map((b) => `${b.kind}：${b.title}${b.edition ? "，" + b.edition : ""}`).join("，"));
  localStorage.setItem(LS.bookList, JSON.stringify(state.bookList));
  localStorage.setItem(LS.windows, JSON.stringify(state.windows));
  localStorage.setItem(LS.onboarded, state.onboarded ? "1" : "0");
}

async function api(path, body, method = "POST") {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (method !== "GET" && body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

function profilePayload(name) {
  return {
    user_id: name,
    exam_track: "考研",
    school: state.school,
    major: state.major,
    books: state.bookList,
    books_note: state.bookList.map((b) => `${b.kind}：${b.title}${b.edition ? "，" + b.edition : ""}`).join("，"),
    subjects: state.windows,
  };
}

async function syncProfile(name) {
  const data = await api("/onboard", profilePayload(name));
  const profile = data.profile || {};
  state.examDate = profile.target_exam_date || "";
  state.examTrack = profile.exam_track || "考研";
  if (Array.isArray(profile.books) && profile.books.length) {
    state.bookList = profile.books;
  }
  mergeWindows(profile.subjects);
  mergeWindows((data.windows || []).map((w) => w.subject));
  persist();
  return data;
}

function renderBookEditor(rootId) {
  const root = $(rootId);
  if (!root) return;
  root.innerHTML = `
    <div class="book-rows"></div>
    <div class="book-add">
      <input class="book-quick" placeholder="书本名，或粘贴：专业课：数学分析，同济第4版，英语1，政治" />
      <input class="book-edition" placeholder="版本（可空）" />
      <button type="button" class="btn ghost">添加</button>
    </div>
  `;
  const rows = root.querySelector(".book-rows");
  const quick = root.querySelector(".book-quick");
  const edition = root.querySelector(".book-edition");
  const addBtn = root.querySelector(".book-add button");

  function paint() {
    rows.innerHTML = "";
    if (!state.bookList.length) {
      rows.innerHTML = `<p class="muted small">还没有教材。英语一/英语二直接当书名加进去。</p>`;
      return;
    }
    state.bookList.forEach((book, i) => {
      const row = document.createElement("div");
      row.className = "book-row";
      const kind = document.createElement("select");
      KINDS.forEach((k) => {
        const opt = document.createElement("option");
        opt.value = k;
        opt.textContent = k;
        if (k === book.kind) opt.selected = true;
        kind.appendChild(opt);
      });
      kind.onchange = () => {
        state.bookList[i].kind = kind.value;
        const spec = specialTitles(state.bookList[i].title, state.bookList[i].edition, kind.value);
        state.bookList[i].title = spec[0];
        state.bookList[i].edition = spec[1];
        persist();
        renderBookEditor("login-books");
        renderBookEditor("me-books");
      };
      const title = document.createElement("input");
      title.value = book.title;
      title.placeholder = "书本名";
      title.onchange = () => {
        state.bookList[i].title = title.value.trim();
        persist();
      };
      const ver = document.createElement("input");
      ver.value = book.edition || "";
      ver.placeholder = "版本";
      ver.onchange = () => {
        state.bookList[i].edition = ver.value.trim();
        persist();
      };
      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn ghost";
      del.textContent = "删";
      del.onclick = () => {
        state.bookList.splice(i, 1);
        persist();
        renderBookEditor("login-books");
        renderBookEditor("me-books");
      };
      row.append(kind, title, ver, del);
      rows.appendChild(row);
    });
  }

  function addFromInputs() {
    const blob = quick.value.trim();
    const ver = edition.value.trim();
    if (!blob) return;
    let added = parseBooksText(blob);
    if (!added.length) added = [{ title: blob, edition: ver, kind: inferKind(blob) }];
    else if (added.length === 1 && ver) added[0].edition = added[0].edition || ver;
    added = added.map((b) => {
      const spec = specialTitles(b.title, b.edition, b.kind);
      return { title: spec[0], edition: spec[1], kind: b.kind };
    });
    added.forEach((b) => {
      const exists = state.bookList.some((x) => x.title === b.title && x.edition === b.edition && x.kind === b.kind);
      if (!exists) state.bookList.push(b);
    });
    quick.value = "";
    edition.value = "";
    persist();
    renderBookEditor("login-books");
    renderBookEditor("me-books");
  }

  addBtn.onclick = addFromInputs;
  quick.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addFromInputs();
    }
  });
  paint();
}

function readHash() {
  const raw = decodeURIComponent((location.hash || "").replace(/^#/, ""));
  const segs = raw.split("/").filter(Boolean);
  if (segs[0] === "today") return { page: "today", subject: "" };
  if (segs[0] === "me") return { page: "me", subject: "" };
  if (segs[0] === "chat" && segs[1]) return { page: "chat", subject: segs.slice(1).join("/") };
  return { page: "home", subject: "" };
}

function go(page, subject) {
  if (page === "chat" && subject) location.hash = `#/chat/${encodeURIComponent(subject)}`;
  else if (page === "home" || page === "chat") location.hash = "#/home";
  else location.hash = `#/${page}`;
}

function enterChat() {
  const last = state.subject || state.windows[state.windows.length - 1] || state.windows[0];
  if (last) go("chat", last);
  else openNameModal();
}

function openSheet() {
  renderWindows();
  $("side-sheet").hidden = false;
}

function closeSheet() {
  $("side-sheet").hidden = true;
}

function showOnboard() {
  $("view-onboard").hidden = false;
  $("view-home").hidden = true;
  $("view-chat").hidden = true;
  $("view-today").hidden = true;
  $("view-me").hidden = true;
  closeSheet();
  $("user-name").value = state.displayName;
  $("school").value = state.school;
  $("major").value = state.major;
  renderBookEditor("login-books");
}

function fillMe() {
  $("me-name").value = state.displayName;
  $("me-school").value = state.school;
  $("me-major").value = state.major;
  const dateText = state.examDate
    ? `全国硕士研究生初试日自动定为 ${state.examDate}（12 月第三个星期六）。`
    : "考研初试日按全国统考自动计算。";
  $("exam-date-hint").textContent = dateText;
  renderBookEditor("me-books");
}

function mergeWindows(names) {
  (names || []).forEach((item) => {
    const name = typeof item === "string" ? item : "";
    if (name && !state.windows.includes(name)) state.windows.push(name);
  });
}

function renderHomeWindows() {
  const box = $("home-windows");
  if (!box) return;
  box.innerHTML = "";
  state.windows.forEach((name) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = name;
    btn.onclick = () => go("chat", name);
    box.appendChild(btn);
  });
}

function renderWindows() {
  const box = $("window-list");
  box.innerHTML = "";
  if (!state.windows.length) {
    const p = document.createElement("p");
    p.className = "muted small";
    p.textContent = "还没有对话。";
    box.appendChild(p);
    return;
  }
  state.windows.forEach((name) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = name;
    btn.className = name === state.subject ? "on" : "";
    btn.onclick = () => {
      closeSheet();
      go("chat", name);
    };
    box.appendChild(btn);
  });
}

async function applyRoute() {
  if (!state.onboarded) {
    showOnboard();
    return;
  }
  const route = readHash();
  state.page = route.page;
  $("view-onboard").hidden = true;
  $("view-home").hidden = route.page !== "home";
  $("view-chat").hidden = route.page !== "chat";
  $("view-today").hidden = route.page !== "today";
  $("view-me").hidden = route.page !== "me";
  if (route.page !== "chat") closeSheet();
  document.querySelectorAll("[data-nav]").forEach((b) => {
    b.classList.toggle("on", b.dataset.nav === route.page);
  });
  if (route.page === "home") renderHomeWindows();
  if (route.page === "today") loadDashboard();
  if (route.page === "me") fillMe();
  if (route.page === "chat") {
    if (route.subject) await showConversation(route.subject, state.pendingSwitch);
    else go("home");
    state.pendingSwitch = null;
  }
}

function countdownHtml(cd, extra) {
  if (!cd) return `<p>考研初试日按全国统考自动计算中。</p>`;
  return `
    <p class="phase">${cd.phase_title || ""} · ${cd.urgency || ""}</p>
    <p class="days">${cd.days_left ?? "—"}</p>
    <p>天 · ${cd.message || ""}</p>
    <p>${extra || cd.phase_tip || ""}</p>
  `;
}

async function loadDashboard() {
  $("dash-message").textContent = "正在读取今日学习状态…";
  $("blocks").innerHTML = "";
  try {
    const dash = await api("/dashboard", { user_id: state.userId });
    $("countdown-card").innerHTML = countdownHtml(dash.countdown, dash.phase_hint?.tip);
    $("dash-message").textContent = dash.message || "";
    const blocks = dash.blocks || [];
    if (!blocks.length) {
      $("blocks").innerHTML = `<p class="muted">还没有对话窗口时，这里不会拆科目块。去对话页给窗口起个名字。</p>`;
      return;
    }
    $("blocks").innerHTML = blocks
      .map(
        (b) => `
      <button class="block" data-subject="${b.subject}">
        <strong>${b.subject}</strong>
        <p>${b.task_summary || "巩固"}</p>
        <p class="mins">${b.minutes} 分钟</p>
      </button>`
      )
      .join("");
    $("blocks").querySelectorAll(".block").forEach((el) => {
      el.onclick = () => go("chat", el.dataset.subject);
    });
  } catch (err) {
    $("dash-message").textContent = "今日状态加载失败，可能服务刚重启。正在重新同步画像…";
    try {
      await syncProfile(state.userId);
      $("dash-message").textContent = "画像已同步，请再打开今日。";
    } catch (e) {
      $("countdown-card").innerHTML = `<p>${err.message}</p>`;
    }
  }
}

function openNameModal() {
  $("name-modal").hidden = false;
  $("window-title").value = "";
  $("window-title").focus();
}

function closeNameModal() {
  $("name-modal").hidden = true;
}

async function createWindow(event) {
  event.preventDefault();
  const title = $("window-title").value.trim();
  if (!title) return;
  const data = await api("/session/new", { user_id: state.userId, title });
  if (!state.windows.includes(title)) state.windows.push(title);
  persist();
  renderHomeWindows();
  closeNameModal();
  state.pendingSwitch = data;
  go("chat", data.subject || title);
}

async function showConversation(subject, already) {
  state.subject = subject;
  renderWindows();
  $("chat-title").textContent = subject;
  const sw = already || (await api("/session/switch", { user_id: state.userId, subject }));
  state.sessionId = sw.session_id;
  const box = $("messages");
  box.innerHTML = "";
  box.dataset.loaded = state.sessionId;
  (sw.messages || []).forEach((m) => addBubble(m.role === "user" ? "user" : "assistant", m.content));
  if (!(sw.messages || []).length) {
    addBubble("assistant", `这是「${subject}」专属会话。问这科的题、要计划，或说一声累了。`);
  }
  $("chat-input").placeholder = `问「${subject}」…`;
  $("chat-input").focus();
}

function addBubble(role, text, extraClass) {
  const el = document.createElement("div");
  el.className = `bubble ${role} ${extraClass || ""}`;
  el.textContent = text;
  $("messages").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

async function readSse(res, handlers) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      let event = "message";
      let data = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data = line.slice(5).trim();
      }
      if (!data) continue;
      const json = JSON.parse(data);
      if (handlers[event]) handlers[event](json);
    }
  }
}

async function sendChat(event) {
  event.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text || state.sending) return;
  if (!state.subject) {
    openNameModal();
    return;
  }
  state.sending = true;
  $("btn-send").disabled = true;
  $("chat-input").value = "";
  addBubble("user", text);
  const bot = addBubble("assistant", "正在检索与调度…", "pending");
  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        subject: state.subject,
        session_id: state.sessionId,
        message: text,
        user_energy: 0.7,
        emotion_intensity: 0.0,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    bot.textContent = "";
    await readSse(res, {
      status: (d) => {
        if (d.session_id) state.sessionId = d.session_id;
      },
      token: (d) => {
        bot.classList.remove("pending");
        bot.textContent += (bot.textContent ? "\n\n" : "") + (d.text || "");
        bot.scrollIntoView({ block: "end" });
      },
      done: (d) => {
        if (d.session_id) state.sessionId = d.session_id;
      },
    });
    if (!bot.textContent) bot.textContent = "没有返回内容。";
  } catch (err) {
    bot.classList.remove("pending");
    bot.textContent = "发送失败：" + err.message;
  } finally {
    state.sending = false;
    $("btn-send").disabled = false;
  }
}

async function onboard(event) {
  event.preventDefault();
  const err = $("onboard-error");
  err.hidden = true;
  const displayName = $("user-name").value.trim();
  const school = $("school").value.trim();
  const major = $("major").value.trim();
  if (!displayName || !school || !major) {
    err.textContent = "名字、学校、专业都要填。";
    err.hidden = false;
    return;
  }
  state.displayName = displayName;
  state.userId = displayName;
  state.school = school;
  state.major = major;
  state.examTrack = "考研";
  try {
    await syncProfile(displayName);
    state.onboarded = true;
    persist();
    go("home");
    applyRoute();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  }
}

async function saveProfile(event) {
  event.preventDefault();
  const status = $("me-status");
  status.hidden = true;
  const displayName = $("me-name").value.trim();
  const school = $("me-school").value.trim();
  const major = $("me-major").value.trim();
  if (!displayName || !school || !major) {
    status.textContent = "名字、学校、专业都要填。";
    status.hidden = false;
    return;
  }
  state.displayName = displayName;
  state.userId = displayName;
  state.school = school;
  state.major = major;
  try {
    const data = await api("/profile", profilePayload(displayName));
    if (data.profile && Array.isArray(data.profile.books)) state.bookList = data.profile.books;
    persist();
    fillMe();
    status.textContent = "已保存。对话窗口还在。";
    status.hidden = false;
  } catch (e) {
    status.textContent = e.message;
    status.hidden = false;
  }
}

function bind() {
  $("onboard-form").onsubmit = onboard;
  $("profile-form").onsubmit = saveProfile;
  $("chat-form").onsubmit = sendChat;
  $("name-form").onsubmit = createWindow;
  $("btn-home-new").onclick = openNameModal;
  $("btn-sheet-new").onclick = () => {
    closeSheet();
    openNameModal();
  };
  $("btn-cancel-name").onclick = closeNameModal;
  $("btn-back").onclick = () => go("home");
  $("btn-sheet").onclick = openSheet;
  $("sheet-backdrop").onclick = closeSheet;
  $("name-modal").addEventListener("click", (e) => {
    if (e.target.id === "name-modal") closeNameModal();
  });
  document.querySelectorAll("[data-nav]").forEach((b) => {
    b.onclick = () => {
      if (b.dataset.nav === "chat") enterChat();
      else go(b.dataset.nav);
    };
  });
  $("btn-reset").onclick = () => {
    localStorage.removeItem(LS.onboarded);
    state.onboarded = false;
    state.subject = "";
    state.sessionId = "";
    location.hash = "";
    showOnboard();
  };
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  window.addEventListener("hashchange", applyRoute);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSheet();
  });
}

async function boot() {
  bind();
  if (!state.onboarded) {
    showOnboard();
    return;
  }
  try {
    await syncProfile(state.userId);
    for (const name of state.windows) {
      await api("/session/new", { user_id: state.userId, title: name });
    }
  } catch (e) {
    /* 服务未起时仍进页面 */
  }
  if (!location.hash || location.hash === "#/chat") location.hash = "#/home";
  else applyRoute();
}

boot();
