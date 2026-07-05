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

export const api = {
  startRun: (data: any) => request('/api/runs', { method: 'POST', body: JSON.stringify(data) }),
  getRun: (id: string) => request(`/api/runs/${id}`),
  estimate: (data: any) => request('/api/estimate', { method: 'POST', body: JSON.stringify(data) }),
};
