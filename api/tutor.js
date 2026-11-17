// Mr Marvin, the AI tutor. Vercel serverless function: POST /api/tutor
// Body: { messages: [{ role: "user" | "assistant", content: string }, ...] } ending on a user turn.
// Streams the reply back as plain text. Uses NVIDIA_API_KEY (Nemotron) if set, otherwise ANTHROPIC_API_KEY.
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const SYSTEM = `You are Mr Marvin, a warm, patient AP Precalculus tutor on a study site built for a high-school junior named Santiago. Santiago is preparing for a class test on Unit 1 (Polynomial and Rational Functions: change in tandem, average rates of change, linear and quadratic rates, polynomial zeros, multiplicity and complex conjugate zeros, end behavior, rational functions and their zeros, vertical asymptotes, holes, horizontal and slant asymptotes, equivalent forms and the binomial theorem, transformations, model selection, model construction) and Unit 2 (Exponential and Logarithmic Functions: arithmetic and geometric sequences, linear vs exponential change, exponential functions and their manipulation, exponential modeling, residuals and competing models, composition, inverse functions, logarithmic expressions, logs as inverses of exponentials, log graphs, log rules, exponential and logarithmic equations and inequalities, log modeling, semi-log plots), following the College Board AP Precalculus course. Santiago is going through a lot right now, so be calm and encouraging and never make them feel bad for not knowing something.

How you teach:
- Keep answers short and clear, usually under 180 words. Use short paragraphs or numbered steps.
- Write math in plain text with Unicode symbols (x², √, ≠, ≤, →, ∞, log₂, f⁻¹). Never use LaTeX or dollar-sign math.
- When asked how to solve a problem, go step by step and say why each step works. When asked for a hint, give only the next step.
- End most answers with one short question that checks understanding, or point to a matching part of the site: the Unit 1 or Unit 2 levels, the Visual Lab, the MCQ Arena, the FRQ Gym, or Flashcards.
- For FRQ work, name what earns rubric points: equations written before numbers, units, the words "on average" with the interval, 3 decimal places on calculator answers, and justifications in context.
- Check your arithmetic carefully before answering. If a question is ambiguous, say what you assumed.
- If asked about something outside AP Precalculus, answer briefly if it's school-related, then steer back to the units.
- If Santiago says something that suggests they are in crisis or unsafe, respond with care, encourage them to talk with a trusted adult or school counselor, and mention they can call or text 988 (the Suicide & Crisis Lifeline in the US).`;

const MAX_TURNS = 16;
const MAX_CHARS_PER_MESSAGE = 6000;
const MAX_TOTAL_CHARS = 30000;

function cleanMessages(raw) {
  if (!Array.isArray(raw)) return null;
  const msgs = raw
    .slice(-MAX_TURNS)
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string" &&
        m.content.trim() !== "",
    )
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_CHARS_PER_MESSAGE) }));
  while (msgs.length && msgs[0].role !== "user") msgs.shift();
  // Merge accidental back-to-back turns from the same role.
  const merged = [];
  for (const m of msgs) {
    const last = merged[merged.length - 1];
    if (last && last.role === m.role) last.content += "\n\n" + m.content;
    else merged.push({ ...m });
  }
  if (!merged.length || merged[merged.length - 1].role !== "user") return null;
  const total = merged.reduce((n, m) => n + m.content.length, 0);
  if (total > MAX_TOTAL_CHARS) return null;
  return merged;
}

// NVIDIA (OpenAI-compatible) endpoint: used when NVIDIA_API_KEY is set.
const NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b";

async function streamNvidia(messages, res) {
  const upstream = await fetch(NVIDIA_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.NVIDIA_API_KEY}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: NVIDIA_MODEL,
      messages: [{ role: "system", content: SYSTEM }, ...messages],
      temperature: 1,
      top_p: 0.95,
      max_tokens: 16384,
      chat_template_kwargs: { enable_thinking: true },
      stream: true,
    }),
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    console.error("nvidia error", upstream.status, detail.slice(0, 500));
    const status = upstream.status === 429 ? 429 : upstream.status === 401 || upstream.status === 403 ? 503 : 502;
    const code = status === 429 ? "rate_limited" : status === 503 ? "not_configured" : "upstream_error";
    const message =
      status === 429
        ? "Mr Marvin is getting a lot of questions. Try again in a minute."
        : status === 503
          ? "Mr Marvin's NVIDIA API key was rejected. Check NVIDIA_API_KEY in Vercel."
          : "Mr Marvin hit a snag. Try again in a moment.";
    return res.status(status).json({ error: code, message });
  }

  res.writeHead(200, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });

  // Only the final answer goes to the student: reasoning_content is skipped, and any
  // <think>...</think> block that shows up inside content is filtered out.
  const decoder = new TextDecoder();
  const reader = upstream.body.getReader();
  let buf = "";
  let inThink = false;
  let pending = "";
  let wrote = false;
  let finish = null;
  const emit = (text) => {
    pending += text;
    let out = "";
    while (pending) {
      if (inThink) {
        const end = pending.indexOf("</think>");
        if (end < 0) { pending = pending.slice(-8); break; }
        pending = pending.slice(end + 8);
        inThink = false;
      } else {
        const start = pending.indexOf("<think>");
        if (start < 0) {
          // Hold back a possible partial "<think" tag at the end.
          const keep = pending.lastIndexOf("<");
          if (keep >= 0 && pending.length - keep < 7) { out += pending.slice(0, keep); pending = pending.slice(keep); }
          else { out += pending; pending = ""; }
          break;
        }
        out += pending.slice(0, start);
        pending = pending.slice(start + 7);
        inThink = true;
      }
    }
    if (out) {
      if (!wrote) out = out.replace(/^\s+/, "");
      if (out) { res.write(out); wrote = true; }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      let chunk;
      try { chunk = JSON.parse(data); } catch { continue; }
      const choice = chunk.choices && chunk.choices[0];
      if (!choice) continue;
      if (choice.delta && typeof choice.delta.content === "string") emit(choice.delta.content);
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  }
  if (!inThink && pending) { res.write(pending); wrote = true; }
  if (!wrote) res.write("Sorry, I lost my train of thought. Could you ask that again?");
  if (finish === "length") res.write("\n\n(I ran out of room there. Say \"keep going\" and I'll finish.)");
  return res.end();
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "method_not_allowed", message: "Use POST." });
  }
  if (!process.env.NVIDIA_API_KEY && !process.env.ANTHROPIC_API_KEY) {
    return res.status(503).json({
      error: "not_configured",
      message: "Mr Marvin isn't connected yet: set NVIDIA_API_KEY (or ANTHROPIC_API_KEY) on the server.",
    });
  }

  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      body = null;
    }
  }
  const messages = cleanMessages(body && body.messages);
  if (!messages) {
    return res.status(400).json({
      error: "bad_request",
      message: "Send { messages: [...] } ending with a user message, under 30,000 characters.",
    });
  }

  if (process.env.NVIDIA_API_KEY) {
    try {
      return await streamNvidia(messages, res);
    } catch (err) {
      console.error("nvidia stream error", err && err.message);
      if (!res.headersSent) {
        return res.status(502).json({ error: "upstream_error", message: "Mr Marvin hit a snag. Try again in a moment." });
      }
      res.write("\n\n[Mr Marvin hit a snag. Try again in a moment.]");
      return res.end();
    }
  }

  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    });
  };

  try {
    const stream = client.beta.messages.stream({
      model: "claude-opus-5-5",
      max_tokens: 8000,
      system: SYSTEM,
      thinking: { type: "adaptive" },
      output_config: { effort: "medium" },
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      messages,
    });

    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        start();
        res.write(event.delta.text);
      }
    }

    const final = await stream.finalMessage();
    start();
    if (final.stop_reason === "refusal") {
      res.write("\n\nI can't help with that one. Let's get back to precalc. What topic are you working on?");
    } else if (final.stop_reason === "max_tokens") {
      res.write("\n\n(I ran out of room there. Say \"keep going\" and I'll finish.)");
    }
    return res.end();
  } catch (err) {
    let status = 502;
    let code = "upstream_error";
    let message = "Mr Marvin hit a snag. Try again in a moment.";
    if (err instanceof Anthropic.AuthenticationError) {
      status = 503;
      code = "not_configured";
      message = "Mr Marvin's API key was rejected. Check ANTHROPIC_API_KEY in Vercel.";
    } else if (err instanceof Anthropic.RateLimitError) {
      status = 429;
      code = "rate_limited";
      message = "Mr Marvin is getting a lot of questions. Try again in a minute.";
    } else if (err instanceof Anthropic.BadRequestError) {
      status = 400;
      code = "bad_request";
      message = "Mr Marvin couldn't read that question. Try rephrasing it.";
    } else if (err instanceof Anthropic.APIError) {
      status = 502;
    }
    console.error("tutor error", err && err.status, err && err.message);
    if (!started) return res.status(status).json({ error: code, message });
    res.write("\n\n[" + message + "]");
    return res.end();
  }
}
