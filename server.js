// Skills chat web server. No external dependencies (Node 18+).
// Env:
//   PORT         default 3000
//   MODEL_URL    base URL of the Kaggle model server, e.g. https://xyz.trycloudflare.com
//   MODEL_TOKEN  API token printed by the Kaggle notebook

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT || 3000);
const MODEL_URL = (process.env.MODEL_URL || "").replace(/\/$/, "");
const MODEL_TOKEN = process.env.MODEL_TOKEN || "";
const SKILLS_FILE = path.join(__dirname, "skills.json");
const PUBLIC_DIR = __dirname;

const TOP_K = 3;                 // skills retrieved automatically per question
const PER_SKILL_CHARS = 6000;    // max characters of one skill injected
const TOTAL_SKILL_CHARS = 14000; // max characters of all skills injected

// ---------- load and index skills ----------
let skills = [];
let docs = [];       // {tf: Map<term, count>, len, head: Set<term>}  head = name + description tokens
let df = new Map();  // document frequency per term
let avgLen = 1;

function tokenize(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}_-]*/gu) || [])
    .filter((t) => t.length > 1);
}

function loadSkills() {
  if (!fs.existsSync(SKILLS_FILE)) {
    console.warn(`skills index not found at ${SKILLS_FILE}; run build_index.py`);
    return;
  }
  skills = JSON.parse(fs.readFileSync(SKILLS_FILE, "utf8")).skills || [];
  docs = [];
  df = new Map();
  let total = 0;
  for (const s of skills) {
    const weighted = [
      ...tokenize(s.name).flatMap((t) => [t, t, t]),          // name counts 3x
      ...tokenize(s.description).flatMap((t) => [t, t]),      // description 2x
      ...tokenize((s.content || "").slice(0, 4000)),
    ];
    const tf = new Map();
    for (const t of weighted) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    const head = new Set([...tokenize(s.name), ...tokenize(s.description)]);
    docs.push({ tf, len: weighted.length, head });
    total += weighted.length;
  }
  avgLen = total / Math.max(docs.length, 1);
  console.log(`loaded ${skills.length} skills`);
}

// BM25 scoring
function search(query, limit = TOP_K) {
  const terms = [...new Set(tokenize(query))];
  const N = docs.length;
  const k1 = 1.4, b = 0.75;
  const scored = docs.map((d, i) => {
    let score = 0;
    let headHit = false;
    for (const t of terms) {
      const f = d.tf.get(t);
      if (!f) continue;
      if (d.head.has(t)) headHit = true;
      const idf = Math.log(1 + (N - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
      score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.len / avgLen));
    }
    return { i, score: headHit ? score : 0 };
  });
  return scored
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((x) => skills[x.i]);
}

// ---------- prompt building ----------
const BASE_SYSTEM = `You are a helpful assistant. When skill instructions are provided below, follow them for the matching task and say briefly which skill you used. If no skill applies, answer normally. Reply in the same language as the user.`;

function buildSystemPrompt(selected) {
  let used = 0;
  const parts = [BASE_SYSTEM];
  for (const s of selected) {
    const body = (s.content || "").slice(0, PER_SKILL_CHARS);
    if (used + body.length > TOTAL_SKILL_CHARS) break;
    used += body.length;
    parts.push(`\n### SKILL: ${s.name}\nSource: ${s.source_url}\n\n${body}`);
  }
  return parts.join("\n");
}

// ---------- model health ----------
let modelHealth = { online: false, checkedAt: 0 };
async function checkModel() {
  if (!MODEL_URL) return false;
  if (Date.now() - modelHealth.checkedAt < 20_000) return modelHealth.online;
  let online = false;
  try {
    const res = await fetch(`${MODEL_URL}/health`, { signal: AbortSignal.timeout(5_000) });
    online = res.ok;
  } catch { online = false; }
  modelHealth = { online, checkedAt: Date.now() };
  return online;
}

// ---------- model call ----------
async function callModel(system, messages, maxNew = 1024) {
  if (!MODEL_URL) throw new Error("MODEL_URL is not set");
  const res = await fetch(`${MODEL_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${MODEL_TOKEN}`,
    },
    body: JSON.stringify({ system, messages, max_new_tokens: maxNew, temperature: 0.3 }),
    signal: AbortSignal.timeout(180_000),
  });
  if (!res.ok) throw new Error(`model server returned ${res.status}`);
  return (await res.json()).text;
}

// ---------- HTTP helpers ----------
function sendJson(res, status, obj) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(obj));
}

function readBody(req, limit = 1_000_000) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (c) => {
      data += c;
      if (data.length > limit) {
        reject(new Error("body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try { resolve(data ? JSON.parse(data) : {}); } catch (e) { reject(e); }
    });
    req.on("error", reject);
  });
}

function serveStatic(req, res) {
  // Only the chat page is public; the server code and skills index are never served.
  const pathname = req.url.split("?")[0];
  if (pathname !== "/" && pathname !== "/index.html") { res.writeHead(404); return res.end("not found"); }
  fs.readFile(path.join(PUBLIC_DIR, "index.html"), (err, buf) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(buf);
  });
}

// ---------- routes ----------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");

  if (req.method === "GET" && url.pathname === "/api/status") {
    const modelOnline = await checkModel();
    return sendJson(res, 200, { skills: skills.length, modelConfigured: Boolean(MODEL_URL), modelOnline });
  }

  if (req.method === "GET" && url.pathname === "/api/skills") {
    const q = url.searchParams.get("q") || "";
    const limit = Math.min(Number(url.searchParams.get("limit") || 20), 100);
    const list = q ? search(q, limit) : skills.slice(0, limit);
    return sendJson(res, 200, list.map(({ id, name, description, repo, source_url, license }) =>
      ({ id, name, description, repo, source_url, license })));
  }

  if (req.method === "POST" && url.pathname === "/api/chat") {
    try {
      const body = await readBody(req);
      const messages = (body.messages || []).filter((m) => m && m.content)
        .slice(-20).map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: String(m.content).slice(0, 8000) }));
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      if (!lastUser) return sendJson(res, 400, { error: "no user message" });

      const pinned = (body.pinned || []).map((id) => skills.find((s) => s.id === id)).filter(Boolean);

      // The index is in English. For non-English questions, ask the model for English
      // keywords and search with those too. If that step fails, search the raw text.
      let query = lastUser.content;
      if (body.autoSkills !== false && /[\u0600-\u06FF]/.test(query)) {
        try {
          const kw = await callModel(
            "Extract 5 to 10 English search keywords describing the user's request. Output only the keywords, separated by spaces.",
            [{ role: "user", content: query }],
            40
          );
          query = `${query} ${kw.replace(/[^\p{L}\p{N}\s-]/gu, " ").slice(0, 200)}`;
        } catch (e) {
          console.warn("keyword expansion failed:", e.message);
        }
      }
      const auto = body.autoSkills === false ? [] : search(query, TOP_K);
      const selected = [...pinned, ...auto.filter((s) => !pinned.includes(s))];

      if (!(await checkModel())) {
        return sendJson(res, 503, {
          error: "النموذج غير متصل حالياً. تأكد من تشغيل خادم النموذج ثم حدّث الرابط في إعدادات الموقع.",
        });
      }
      const text = await callModel(buildSystemPrompt(selected), messages);
      return sendJson(res, 200, {
        text,
        skillsUsed: selected.map((s) => ({ id: s.id, name: s.name, source_url: s.source_url })),
      });
    } catch (e) {
      console.error(e);
      return sendJson(res, 500, { error: e.message });
    }
  }

  if (req.method === "GET") return serveStatic(req, res);
  res.writeHead(405); res.end();
});

process.on("unhandledRejection", (err) => console.error("unhandled rejection:", err));
process.on("uncaughtException", (err) => console.error("uncaught exception:", err));
server.on("error", (err) => console.error("server error:", err));

loadSkills();
server.listen(PORT, () => console.log(`chat server on http://localhost:${PORT}`));
