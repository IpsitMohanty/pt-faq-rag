const express = require("express");
const cors = require("cors");

const app = express();

// ✅ Allow your Vite dev server to call the gateway
app.use(cors({ origin: ["http://localhost:5173", "http://127.0.0.1:5173"] }));

app.use(express.json());

const PY_BACKEND = "http://localhost:8000";

app.get("/health", async (_req, res) => {
  try {
    const r = await fetch(`${PY_BACKEND}/health`);
    const j = await r.json();
    res.json(j);
  } catch (e) {
    res.status(500).json({ error: "backend not reachable" });
  }
});

// ✅ Non-streaming chat proxy (hardened)
app.post("/chat", async (req, res) => {
  const controller = new AbortController();
  const timeoutMs = 18000; // keep slightly below UI timeout (20s)
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const r = await fetch(`${PY_BACKEND}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
      signal: controller.signal,
    });

    const text = await r.text(); // safer than r.json()

    const ct = r.headers.get("content-type") || "application/json";
    res.status(r.status).set("Content-Type", ct).send(text);
  } catch (e) {
    const msg =
      e?.name === "AbortError"
        ? `Gateway timeout after ${timeoutMs / 1000}s`
        : `Gateway error: ${String(e)}`;

    res.status(504).json({ answer: msg, sources: [] });
  } finally {
    clearTimeout(timer);
  }
});

// Streaming proxy (later) — note: this may not work with Node fetch streams as written
app.post("/chat/stream", async (req, res) => {
  const upstream = await fetch(`${PY_BACKEND}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  // Might not work on some Node fetch impls; ok for now.
  upstream.body.on("data", (chunk) => res.write(chunk));
  upstream.body.on("end", () => res.end());
});

app.listen(3001, () => {
  console.log("Gateway running on http://localhost:3001");
});
