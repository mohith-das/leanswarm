# Web

This directory contains a minimal Next.js app for inspecting Lean Swarm simulation output.
It is optional and separate from the Python package. You do not need Node.js to install or use
`leanswarm`, the API, or the CLI.

## What it does

- Paste a simulation JSON payload from the CLI or API.
- Inspect the normalized world snapshot, prediction report, and recent ticks.
- Click agents in the list or graph to focus the post-simulation world view.

## Run locally

```bash
cd web
npm install
npm run dev
```

The page accepts both the current `simulate` payload shape and common `result` / `world_snapshot`
wrappers, so you can paste CLI output directly without reshaping it first.
