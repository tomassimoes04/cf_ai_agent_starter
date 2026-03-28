import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
  type ModelMessage
} from "ai";
import { z } from "zod";


// --- RAG utilities ---
async function extractTextFromFile(file: ArrayBuffer, mediaType: string): Promise<string> {
  if (mediaType === "text/plain") {
    return new TextDecoder().decode(file);
  }
  // PDF support removed; only .txt files are supported
  throw new Error("Unsupported file type for text extraction");
}

async function embedText(env: Env, text: string): Promise<number[]> {
  // Use Cloudflare Vectorize binding
  // See: https://developers.cloudflare.com/vectorize/
  const response = await env.VECTORIZE.embed({ text });
  return response.embedding;
}

async function storeEmbedding(env: Env, embedding: number[], text: string, meta: Record<string, any>) {
  // Store in D1: id, embedding (as JSON), text, meta
  const stmt = env.D1.prepare(
    `INSERT INTO rag_docs (embedding, text, meta) VALUES (?1, ?2, ?3)`
  ).bind(JSON.stringify(embedding), text, JSON.stringify(meta));
  await stmt.run();
}

// --- End RAG utilities ---

/**
 * The AI SDK's downloadAssets step runs `new URL(data)` on every file
 * part's string data. Data URIs parse as valid URLs, so it tries to
 * HTTP-fetch them and fails. Decode to Uint8Array so the SDK treats
 * them as inline data instead.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    // --- RAG retrieval: fetch top docs for user query ---
    let ragContext = "";
    const lastUserMsg = this.messages.slice().reverse().find(m => m.role === "user" && typeof m.content === "string");
    if (lastUserMsg && typeof lastUserMsg.content === "string" && lastUserMsg.content.trim().length > 0) {
      try {
        const resp = await fetch("/rag/query", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ query: lastUserMsg.content })
        });
        if (resp.ok) {
          const data = await resp.json();
          if (data.results && data.results.length > 0) {
            ragContext = data.results.map((r: any, i: number) => `RAG Source [${i+1}]:\n${r.text}\n`).join("\n");
          }
        }
      } catch {}
    }
    // --- End RAG retrieval ---

    const result = streamText({
      model: workersai("@cf/moonshotai/kimi-k2.5", {
        sessionAffinity: this.sessionAffinity
      }),
      system:
        (ragContext ? `You have access to the following reference material from user documents. Use it to answer questions factually and cite the source if possible.\n\n${ragContext}\n\n` : "") +
        `You are a helpful assistant that can understand images. You can check the weather, get the user's timezone, run calculations, and schedule tasks. When users share images, describe what you see and answer questions about them.\n\n${getSchedulePrompt({ date: new Date() })}\n\nIf the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Uses OpenWeatherMap API. Set your API key in the environment as WEATHER_API_KEY.
            const apiKey = (typeof process !== 'undefined' && process.env && process.env.WEATHER_API_KEY) || (globalThis as any).WEATHER_API_KEY || "YOUR_OPENWEATHERMAP_API_KEY";
            if (!apiKey || apiKey === "YOUR_OPENWEATHERMAP_API_KEY") {
              return { error: "Weather API key not set. Please set WEATHER_API_KEY in your environment." };
            }
            const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${apiKey}&units=metric`;
            try {
              const resp = await fetch(url);
              if (!resp.ok) {
                return { error: `Weather API error: ${resp.statusText}` };
              }
              const data = await resp.json();
              return {
                city: data.name,
                temperature: data.main.temp,
                condition: data.weather[0].description,
                unit: "celsius"
              };
            } catch (error) {
              return { error: `Failed to fetch weather: ${error}` };
            }
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

// Exported fetch handler (outside class)
export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);
    // --- RAG file upload endpoint ---
    if (url.pathname === "/rag/upload" && request.method === "POST") {
      try {
        const contentType = request.headers.get("content-type") || "";
        if (!contentType.startsWith("multipart/form-data")) {
          return new Response("Expected multipart/form-data", { status: 400 });
        }
        const form = await request.formData();
        const file = form.get("file");
        if (!(file instanceof File)) {
          return new Response("No file uploaded", { status: 400 });
        }
        const arrayBuffer = await file.arrayBuffer();
        const text = await extractTextFromFile(arrayBuffer, file.type);
        const embedding = await embedText(env, text);
        await storeEmbedding(env, embedding, text, { filename: file.name, type: file.type });
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      } catch (e: any) {
        return new Response(`RAG upload error: ${e.message || e}`, { status: 500 });
      }
    }
    // --- RAG query endpoint ---
    if (url.pathname === "/rag/query" && request.method === "POST") {
      try {
        const { query } = await request.json();
        if (!query) return new Response("Missing query", { status: 400 });
        const queryEmbedding = await embedText(env, query);
        // Fetch all docs and compute cosine similarity (for demo; production should use vector search API)
        const docs = await env.D1.prepare("SELECT id, embedding, text, meta FROM rag_docs").all();
        function cosineSim(a: number[], b: number[]) {
          const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
          const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
          const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
          return dot / (normA * normB);
        }
        const scored = docs.results.map((row: any) => ({
          ...row,
          embedding: JSON.parse(row.embedding),
          meta: JSON.parse(row.meta),
          score: cosineSim(queryEmbedding, JSON.parse(row.embedding))
        }));
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, 3).map((r) => ({ text: r.text, meta: r.meta, score: r.score }));
        return new Response(JSON.stringify({ results: top }), { status: 200 });
      } catch (e: any) {
        return new Response(`RAG query error: ${e.message || e}`, { status: 500 });
      }
    }
    // --- End RAG endpoints ---
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
