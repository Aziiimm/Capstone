export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  catalogue_size: number;
}

export interface SearchResult {
  asin: string;
  item_idx: number;
  title: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface Recommendation {
  asin: string;
  title: string;
}

export interface RecommendResponse {
  asin: string;
  query_title: string;
  recommendations: Recommendation[];
  inference_ms: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/health", { signal }),

  search: (q: string, limit = 10, signal?: AbortSignal) =>
    request<SearchResponse>(
      `/search?q=${encodeURIComponent(q)}&limit=${limit}`,
      { signal },
    ),

  recommend: (asin: string, top_k = 10, signal?: AbortSignal) =>
    request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify({ asin, top_k }),
      signal,
    }),
};
