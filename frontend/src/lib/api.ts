const API_PREFIX = "/api";
const DEFAULT_TIMEOUT_MS = 15000;

async function request<T>(
  path: string,
  init: RequestInit = {},
  token?: string | null,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init.headers || {}),
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("后端服务响应超时，请确认 127.0.0.1:8000 已正常启动。");
    }
    throw new Error("无法连接后端服务，请确认 127.0.0.1:8000 已正常启动。");
  } finally {
    window.clearTimeout(timeoutId);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "请求失败" }));
    throw new Error(payload.detail || "请求失败");
  }
  return response.json() as Promise<T>;
}

type StreamHandlers = {
  onEvent: (event: string, payload: any) => void;
};

async function stream(path: string, body: unknown, token: string, handlers: StreamHandlers, signal?: AbortSignal) {
  const response = await fetch(`${API_PREFIX}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "请求失败" }));
    throw new Error(payload.detail || "请求失败");
  }
  if (!response.body) {
    throw new Error("服务端未返回流式内容");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const flushBlock = (block: string) => {
    const lines = block.split(/\r?\n/);
    let event = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) {
      return;
    }
    const raw = dataLines.join("\n");
    try {
      const payload = JSON.parse(raw);
      handlers.onEvent(event, payload);
    } catch {
      console.warn("SSE: failed to parse data block:", raw);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      if (block.trim()) {
        flushBlock(block);
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    flushBlock(tail);
  }
}

export const api = {
  get: <T>(path: string, token?: string | null, timeoutMs?: number) =>
    request<T>(path, { method: "GET" }, token, timeoutMs),
  post: <T>(path: string, body: unknown, token?: string | null, timeoutMs?: number) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }, token, timeoutMs),
  delete: <T>(path: string, token?: string | null, timeoutMs?: number) =>
    request<T>(path, { method: "DELETE" }, token, timeoutMs),
  stream: (path: string, body: unknown, token: string, handlers: StreamHandlers, signal?: AbortSignal) =>
    stream(path, body, token, handlers, signal),
};
