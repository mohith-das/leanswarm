export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request(path: string, options: RequestInit = {}) {
  options.credentials = 'same-origin';
  options.headers = { ...options.headers, 'Content-Type': 'application/json' };
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = 'API Error';
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch (e) {}
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

interface TierModels { flagship: string; standard: string; cheap: string }

export interface StartRunRequest {
  seed_document: string;
  question: string;
  rounds: number;
  max_agents: number;
  group_size: number;
  active_agent_fraction: number;
  convergence_threshold: number;
  random_seed: number;
  live: boolean;
  models: TierModels;
  credentials: Record<string, string>;
  api_base?: string | null;
  api_key?: string | null;
  title?: string | null;
  source_urls: string[];
  use_search: boolean;
  max_sources: number;
}

export interface EstimateRequest {
  rounds: number;
  max_agents: number;
  group_size: number;
  active_agent_fraction: number;
  models: TierModels;
  seed_chars: number;
}

export interface DoctorRequest {
  models: TierModels;
  credentials: Record<string, string>;
  api_base?: string | null;
  api_key?: string | null;
  ping: boolean;
}

export const api = {
  startRun: (data: StartRunRequest) => request('/api/runs', { method: 'POST', body: JSON.stringify(data) }),
  getRun: (id: string) => request(`/api/runs/${id}`),
  estimate: (data: EstimateRequest) => request('/api/estimate', { method: 'POST', body: JSON.stringify(data) }),
  doctor: (data: DoctorRequest) => request('/api/doctor', { method: 'POST', body: JSON.stringify(data) }),
  saveRun: (id: string, title?: string) => request(`/api/runs/${id}/save`, { method: 'POST', body: JSON.stringify({ title }) }),
  publishRun: (id: string, title?: string) => request(`/api/runs/${id}/publish`, { method: 'POST', body: JSON.stringify({ title }) }),
  myRuns: () => request('/api/runs'),
  deleteRun: (id: string) => request(`/api/runs/${id}`, { method: 'DELETE' }),
  gallery: (limit: number, offset: number) => request(`/api/gallery?limit=${limit}&offset=${offset}`),
  galleryItem: (id: string) => request(`/api/gallery/${id}`),
  register: (email: string, password: string) => request('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }),
  login: (email: string, password: string) => request('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
  me: () => request('/api/auth/me'),
  chat: (id: string, body: Record<string, unknown>) => request(`/api/runs/${id}/chat`, { method: 'POST', body: JSON.stringify(body) }),
  report: (id: string, body: Record<string, unknown>) => request(`/api/runs/${id}/report`, { method: 'POST', body: JSON.stringify(body) }),
};
