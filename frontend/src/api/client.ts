// ---------------------------------------------------------------------------
// Minimal fetch wrapper for the Early Sepsis Alert System API.
//
// Uses relative URLs so the Vite dev proxy (localhost:5173 → localhost:8000)
// routes requests transparently.  No external HTTP libraries.
// ---------------------------------------------------------------------------

import { ApiError } from "./types";

const REQUEST_ID_HEADER = "X-Request-ID";

interface RequestOptions extends RequestInit {
  /** JSON body — will be serialised automatically. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  json?: unknown;
}

function extractRequestId(response: Response): string | null {
  return response.headers.get(REQUEST_ID_HEADER);
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new SyntaxError(`Invalid JSON in response body (status ${response.status})`);
  }
}

/**
 * Send a request to the backend API.
 *
 * - `json` option is serialised as a JSON body with the correct Content-Type.
 * - 2xx responses: parsed JSON is returned.
 * - Non-2xx responses: throws `ApiError` with status, parsed body, and request ID.
 * - Network failures: the native fetch `TypeError` propagates.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { json, init } = decomposeOptions(options);

  const headers = new Headers(init.headers);
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, {
    ...init,
    headers,
    body: json !== undefined ? JSON.stringify(json) : init.body,
  });

  if (!response.ok) {
    const body = await parseResponseBody(response);
    throw new ApiError(response.status, body, extractRequestId(response));
  }

  return (await parseResponseBody(response)) as T;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function decomposeOptions(options: RequestOptions): {
  json: unknown;
  init: RequestInit;
} {
  const { json, ...init } = options;
  return { json, init };
}
