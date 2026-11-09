/* Shared helpers for all TrustAI pages */
(function () {
  // If your backend runs somewhere else, change this one line.
  const API_BASE = "http://127.0.0.1:8000";
  try { localStorage.removeItem("apiBase"); } catch (e) { /* ignore old setting */ }

  const svg = (inner) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${inner}</svg>`;
  const ICONS = {
    shield: svg('<path d="M12 3l7 3v5c0 4.500-3 8.200-7 10-4-1.800-7-5.500-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/>'),
    scan: svg('<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="10.500" r="2.500"/><path d="M7.500 17c.8-2 2.500-3 4.500-3s3.700 1 4.500 3"/>'),
    chart: svg('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
    sliders: svg('<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>'),
    camera: svg('<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="3.500"/>'),
    upload: svg('<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>'),
    logout: svg('<path d="M9 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h4M16 8l4 4-4 4M20 12H9"/>'),
    alert: svg('<path d="M12 3l10 18H2L12 3zM12 10v5M12 18h.01"/>'),
    sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.600 5.600l1.400 1.400M17 17l1.400 1.400M18.400 5.600L17 7M7 17l-1.400 1.400"/>'),
    moon: svg('<path d="M20 14.500A8.500 8.500 0 0 1 9.500 4a8.500 8.500 0 1 0 10.500 10.500z"/>'),
    lock: svg('<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>'),
    eye: svg('<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
    ok: svg('<circle cx="12" cy="12" r="9"/><path d="M8 12.500l2.700 2.700L16 9.500"/>'),
    download: svg('<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>'),
    search: svg('<circle cx="11" cy="11" r="6.500"/><path d="M16 16l4.500 4.500"/>'),
    printer: svg('<path d="M7 9V4h10v5M7 17H5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-2M7 14h10v6H7z"/>'),
    close: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  /* ---------- session ---------- */
  const getToken = () => { try { return localStorage.getItem("token"); } catch (e) { return null; } };
  const getUser = () => { try { return localStorage.getItem("username"); } catch (e) { return null; } };
  function setSession(token, username) {
    localStorage.setItem("token", token);
    localStorage.setItem("username", username);
  }
  function clearSession() {
    try { localStorage.removeItem("token"); localStorage.removeItem("username"); } catch (e) { /* ignore */ }
  }

  /* ---------- toasts ---------- */
  function toast(msg, type = "") {
    let box = document.getElementById("toasts");
    if (!box) {
      box = document.createElement("div");
      box.id = "toasts"; box.className = "toasts";
      box.setAttribute("role", "status"); box.setAttribute("aria-live", "polite");
      document.body.appendChild(box);
    }
    const t = document.createElement("div");
    t.className = "toast " + type;
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 260); }, 3800);
  }
  // A message to show on the NEXT page (e.g. after a redirect)
  function flash(msg, type = "") {
    try { sessionStorage.setItem("flash", JSON.stringify({ msg, type })); } catch (e) { /* ignore */ }
  }
  function consumeFlash() {
    try {
      const f = sessionStorage.getItem("flash");
      if (!f) return;
      sessionStorage.removeItem("flash");
      const { msg, type } = JSON.parse(f);
      toast(msg, type);
    } catch (e) { /* ignore */ }
  }

  /* ---------- theme ---------- */
  function getTheme() { try { return localStorage.getItem("theme") || "light"; } catch (e) { return "light"; } }
  function setTheme(t) {
    try { localStorage.setItem("theme", t); } catch (e) { /* ignore */ }
    document.documentElement.setAttribute("data-theme", t);
    document.dispatchEvent(new CustomEvent("themechange"));
  }
  function toggleTheme() { setTheme(getTheme() === "dark" ? "light" : "dark"); }
  function mountThemeFab() {
    const b = document.createElement("button");
    b.className = "theme-fab"; b.type = "button";
    const paint = () => {
      const dark = getTheme() === "dark";
      b.innerHTML = dark ? ICONS.sun : ICONS.moon;
      b.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    };
    b.addEventListener("click", toggleTheme);
    document.addEventListener("themechange", paint);
    paint();
    document.body.appendChild(b);
  }

  /* ---------- API ---------- */
  async function api(path, { method = "GET", json, body, auth = true } = {}) {
    const headers = {};
    const token = getToken();
    if (auth && token) headers.Authorization = "Bearer " + token;
    const opts = { method, headers };
    if (json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
    else if (body) { opts.body = body; }

    let res;
    try {
      res = await fetch(API_BASE + path, opts);
    } catch (e) {
      const err = new Error("Couldn't reach the server. Make sure the backend is running at " + API_BASE + ".");
      err.status = 0;
      throw err;
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* no JSON body */ }
    if (!res.ok) {
      if (res.status === 401 && auth && token) {
        clearSession();
        flash("Your session expired. Please sign in again.", "bad");
        window.location.href = "login.html";
      }
      let msg = data && data.detail;
      if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join(", ");
      const err = new Error(msg || "Request failed (" + res.status + ")");
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function requireAuth() { if (!getToken()) window.location.href = "login.html"; }
  function redirectIfAuthed() { if (getToken()) window.location.href = "app_fixed.html"; }

  async function logout() {
    try { await api("/auth/logout", { method: "POST" }); } catch (e) { /* log out locally anyway */ }
    clearSession();
    flash("You're signed out.", "ok");
    window.location.href = "login.html";
  }

  /* ---------- sidebar ---------- */
  function renderSidebar(active) {
    const el = document.getElementById("sidebar");
    if (!el) return;
    const user = getUser() || "Operator";
    const items = [
      ["app_fixed.html", "verify", "New verification", ICONS.scan],
      ["dashboard.html", "dashboard", "Dashboard", ICONS.chart],
      ["settings.html", "settings", "Settings", ICONS.sliders],
    ];
    el.innerHTML = `
      <div class="brand">${ICONS.shield}<span>Trust<b>AI</b></span></div>
      <nav class="nav" aria-label="Main">
        ${items.map(([href, key, label, icon]) =>
          `<a class="nav-item ${key === active ? "active" : ""}" href="${href}" ${key === active ? 'aria-current="page"' : ""}>${icon}<span>${label}</span></a>`
        ).join("")}
      </nav>
      <button class="theme-btn" id="theme-btn" type="button"></button>
      <div class="server-status"><span class="sdot" id="srv-dot"></span><span id="srv-label">Checking server…</span></div>
      <button class="side-user" id="logout-btn" title="Log out">
        <span class="avatar">${esc(user.slice(0, 2).toUpperCase())}</span>
        <span><span class="name">${esc(user)}</span><span class="out">${ICONS.logout}<span>Log out</span></span></span>
      </button>`;
    document.getElementById("logout-btn").addEventListener("click", logout);
    const tb = document.getElementById("theme-btn");
    const paint = () => {
      const dark = getTheme() === "dark";
      tb.innerHTML = (dark ? ICONS.sun : ICONS.moon) + `<span>${dark ? "Light mode" : "Dark mode"}</span>`;
    };
    tb.addEventListener("click", toggleTheme);
    document.addEventListener("themechange", paint);
    paint();
    pingServer();
  }

  async function pingServer() {
    const dot = document.getElementById("srv-dot"), label = document.getElementById("srv-label");
    if (!dot) return;
    try {
      const r = await fetch(API_BASE + "/");
      if (!r.ok) throw new Error();
      dot.className = "sdot up"; label.textContent = "Server connected";
    } catch (e) {
      dot.className = "sdot down"; label.textContent = "Server offline";
    }
  }

  /* ---------- describing a verification record ---------- */
  // Returns rows of [label, plain text, cssClass]. Callers escape the text.
  function describe(rec) {
    const b = rec.biometric_check || {}, d = rec.document_check || {};
    const pct = (n) => Math.round(n * 100) + "%";
    const rows = [];
    rows.push(["Face match", typeof b.match_score === "number" ? pct(b.match_score) : "–", ""]);
    if (typeof b.threshold_used === "number") rows.push(["Pass mark used", pct(b.threshold_used), ""]);
    rows.push(["Document",
      d.tampered ? "Possible tampering" + (d.flagged_field ? " (" + d.flagged_field + ")" : "") : "No tampering found",
      d.tampered ? "badtxt" : "good"]);
    rows.push(["Liveness", b.liveness_passed ? "Passed" : "Failed", b.liveness_passed ? "good" : "badtxt"]);
    if (typeof b.landmark_score === "number") rows.push(["Face proportions", pct(b.landmark_score) + " similar", ""]);
    if (typeof b.id_age_estimate === "number" && typeof b.selfie_age_estimate === "number")
      rows.push(["Estimated age", b.id_age_estimate + " on ID, " + b.selfie_age_estimate + " in selfie", ""]);
    if (typeof b.age_gap_years === "number") rows.push(["Age gap", b.age_gap_years + " years", ""]);
    if (typeof b.embedding_distance === "number")
      rows.push(["Embedding distance", b.embedding_distance.toFixed(3) +
        (b.embedding_model && b.embedding_model !== "mock" ? " (" + b.embedding_model + ")" : ""), ""]);
    return rows;
  }

  /* ---------- printable report (opens its own clean page) ---------- */
  function printReport(rec) {
    const pass = rec.status === "pass";
    const rows = describe(rec);
    rows.push(["Run by", rec.performed_by || "–", ""]);
    rows.push(["Date", new Date(rec.timestamp || Date.now()).toLocaleString(), ""]);
    if (rec.id_image_filename) rows.push(["ID file", rec.id_image_filename, ""]);
    if (rec.selfie_image_filename) rows.push(["Selfie file", rec.selfie_image_filename, ""]);
    if (rec.id) rows.push(["Reference", rec.id, ""]);
    const notes = [(rec.document_check || {}).notes, (rec.biometric_check || {}).notes].filter(Boolean);
    const color = pass ? "#17794F" : "#B42318";

    const w = window.open("", "_blank");
    if (!w) { toast("Allow pop-ups for this page to print the report.", "bad"); return; }
    w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>Verification report</title>
<style>
  body{font-family:"Segoe UI",system-ui,sans-serif;color:#14213D;margin:0;padding:40px 48px;max-width:760px}
  header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid #B88A2E;padding-bottom:14px;margin-bottom:28px}
  h1{font-family:Georgia,serif;margin:0;font-size:26px} .sub{color:#5E6B80;font-size:13px}
  .verdict{display:flex;align-items:center;gap:20px;border:1px solid ${color};border-radius:8px;padding:18px 22px;margin-bottom:24px}
  .stamp{font-family:Georgia,serif;font-weight:700;font-size:22px;letter-spacing:.1em;color:${color};border:3px double ${color};padding:4px 14px;border-radius:6px;transform:rotate(-5deg)}
  table{width:100%;border-collapse:collapse} td{padding:10px 4px;border-bottom:1px solid #E1E5EC;vertical-align:top} td:first-child{color:#5E6B80;width:38%}
  td.good{color:#17794F;font-weight:600} td.badtxt{color:#B42318;font-weight:600}
  .notes{margin-top:22px;font-size:14px;color:#5E6B80} footer{margin-top:36px;font-size:12px;color:#5E6B80;border-top:1px solid #E1E5EC;padding-top:10px}
  @media print{body{padding:0}}
</style></head><body>
<header><div><h1>TrustAI</h1><div class="sub">Identity verification report</div></div><div class="sub">${esc(new Date().toLocaleString())}</div></header>
<div class="verdict"><span class="stamp">${pass ? "VERIFIED" : "FLAGGED"}</span><div>${esc(rec.overall_reason)}</div></div>
<table>${rows.map(([k, v, c]) => `<tr><td>${esc(k)}</td><td class="${c}">${esc(v)}</td></tr>`).join("")}</table>
${notes.length ? `<div class="notes"><b>Notes</b><br>${notes.map(esc).join("<br>")}</div>` : ""}
<footer>Generated by TrustAI. This report records the outcome of an automated check and should be reviewed by a person before acting on it.</footer>
</body></html>`);
    w.document.close();
    w.focus();
    setTimeout(() => w.print(), 350);
  }

  window.TrustAI = {
    API_BASE, ICONS, esc, api, getUser, setSession, requireAuth, redirectIfAuthed, renderSidebar,
    toast, flash, getTheme, setTheme, toggleTheme, mountThemeFab, describe, printReport,
  };
  document.documentElement.setAttribute("data-theme", getTheme());
  consumeFlash();
})();
