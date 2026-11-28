# StudyQue — Backend

Flask API behind [StudyQue](https://study-flow-ai-lac.vercel.app): turn
lecture audio and study material into transcripts, summaries, explanations,
quizzes, flashcards, tests, oral exams, AI chat over your material, and PDF
exports.

Live: **https://study-flow-ai-backend.vercel.app**

## What it does

- **Transcription** (`POST /api/transcribe`) — audio/video upload (max
  100 MB), transcribed with Gemini first, then Groq Whisper, then OpenAI
  (paid fallback, off by default). Transcripts are cleaned and can be
  rewritten into the language's native script.
- **Study generation** — `POST /api/material` (extract text from
  PDF/DOCX/PPTX/images), `POST /api/generate` (summary / explain),
  `POST /api/flashcards`, `POST /api/test-concepts`, `POST /api/chat`
  (RAG over the material), `POST /api/oral-exam`,
  `POST /api/generate-pdf`.
- **Feedback** (`POST /api/feedback`) — sends feedback email via Resend.
- **Billing scaffolding** (`POST /api/billing/checkout`,
  `POST /api/billing/webhook`) — Stripe Checkout + webhook. Inert while
  the product is free-only (see below).

AI provider chain: Gemini → Groq → OpenRouter (free models) → OpenAI
(only if `ENABLE_OPENAI_FALLBACK=true`).

## Authentication

Every `/api/*` endpoint except `/api/test` requires a Firebase ID token:

```
Authorization: Bearer <idToken>
```

The backend verifies the token with Firebase Admin. Missing/invalid token
→ `401`. If the server has no Firebase service account configured, every
protected endpoint answers `503` (fail closed — the server never runs
unauthenticated). `GET /api/auth/status` is a diagnostic that proves a
token verifies; it also returns the caller's `plan`.

## Quotas and plans

Usage is tracked in memory per Firebase UID (resets on cold start /
redeploy; per-instance on Vercel). Only successful (2xx) responses
consume quota. Over-quota responses are `429` JSON with
`retry_after_seconds` and a `Retry-After` header.

| Limit | Free | Pro |
|---|---|---|
| AI generation calls/day | 5 | 150 |
| Transcription minutes/day | 20 | 280 |
| Feedback messages/hour | 10 | 10 |

The plan is read from Firestore `users/{uid}` field `plan` (`"free"` or
`"pro"`; missing/invalid → `"free"`). If Firestore is unavailable the
plan lookup fails safe to free limits — never unlimited. A global rate
limit of 120 requests/minute per client IP applies to all `/api/*`
except `/api/test`.

## Billing (free-only for now)

The product is **free** until the owner wires up Stripe. The billing
endpoints exist so payments can be switched on later **without a code
change**: set the three `STRIPE_*` env vars in Vercel and they start
working. Until then:

- `POST /api/billing/checkout` (auth required) returns `503`
  `"Billing is not configured"` and makes **no** Stripe call. When
  configured it creates a Stripe Checkout session for the Pro monthly
  price (`STRIPE_PRICE_PRO_MONTHLY`) and returns `checkout_url` /
  `session_id`. Optional JSON body: `success_url`, `cancel_url`
  (defaults point at the production frontend dashboard).
- `POST /api/billing/webhook` (called by Stripe, no Firebase auth —
  the Stripe signature is verified with `STRIPE_WEBHOOK_SECRET`)
  returns `503` while the secret is unset and `400` on a bad signature.
  On `checkout.session.completed`,
  `customer.subscription.updated` and `customer.subscription.deleted`
  it sets `users/{uid}` `plan` to `"pro"` / `"free"` and stores
  `stripeCustomerId`.

No Stripe key, price ID or webhook secret is hardcoded anywhere.

## Environment variables

Copy `.env.example` to `.env` for local development. Never commit real
keys or the service-account JSON.

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Yes | At least one AI key; the server refuses to start without it. |
| `GROQ_API_KEY` | No | Groq fallback + Whisper transcription. |
| `OPENROUTER_API_KEY` | No | OpenRouter free-model fallback. |
| `OPENAI_API_KEY` + `ENABLE_OPENAI_FALLBACK=true` | No | Paid OpenAI fallback, off by default. |
| `FIREBASE_SERVICE_ACCOUNT` or `FIREBASE_SERVICE_ACCOUNT_BASE64` | Yes (prod) | Service-account JSON (raw or base64). Without it, protected endpoints answer 503. |
| `FIREBASE_PROJECT_ID` | No | Override the Firebase project used to verify tokens. |
| `QUOTA_GENERATION_PER_DAY` | No | Free-plan AI calls/day (default 5; Pro is fixed at 150/day). |
| `QUOTA_TRANSCRIPTION_MINUTES_PER_DAY` | No | Free-plan audio minutes/day (default 20; Pro is fixed at 280/day). |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No | Global per-IP limit (default 120). |
| `FEEDBACK_PER_HOUR` | No | Feedback messages/hour/user (default 10). |
| `RESEND_API_KEY`, `FEEDBACK_TO_EMAIL`, `RESEND_FROM_EMAIL` | No | Feedback email; without them `/api/feedback` answers 503. |
| `STRIPE_SECRET_KEY` | No | Stripe secret key. Unset = billing disabled. |
| `STRIPE_PRICE_PRO_MONTHLY` | No | Price ID of the Pro monthly plan. Unset = billing disabled. |
| `STRIPE_WEBHOOK_SECRET` | No | Webhook signing secret. Unset = webhook disabled. |
| `FRONTEND_ORIGINS` | No | Comma-separated allowed CORS origins. |
| `GROQ_TIMEOUT_SECONDS`, `OPENROUTER_TIMEOUT_SECONDS`, `MAX_MATERIAL_CHARS`, `MAX_AI_PROMPT_CHARS`, `GEMINI_MODELS`, `GROQ_MODELS`, `OPENROUTER_MODELS` | No | Tuning / model overrides. See `.env.example`. |

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python app.py          # Flask dev server (debug)
```

Smoke check (with a dummy key; Firebase auth stays disabled locally
unless a service account is set):

```bash
GEMINI_API_KEY=dummy python -c "import app; c = app.app.test_client(); print(c.get('/').status_code, c.get('/api/test').status_code)"
```

## Deploy

Deployed on Vercel from `main` (auto-deploy). `vercel.json` routes all
requests to `app.py`. Set the environment variables in the Vercel
project settings — in particular `FIREBASE_SERVICE_ACCOUNT`, or every
protected endpoint will answer 503.

## Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/` | No | Health check. |
| GET | `/api/test` | No | Health check (exempt from rate limiting). |
| POST | `/api/transcribe` | Yes | Audio/video → transcript. Transcription quota. |
| GET | `/api/auth/status` | Yes | Token diagnostic; returns `uid` and `plan`. |
| POST | `/api/material` | Yes | Extract text from uploaded files. Generation quota. |
| POST | `/api/generate` | Yes | Summary / explain. Generation quota. |
| POST | `/api/flashcards` | Yes | Generation quota. |
| POST | `/api/test-concepts` | Yes | Generation quota. |
| POST | `/api/chat` | Yes | RAG chat over material. Generation quota (POST only). |
| POST | `/api/oral-exam` | Yes | Generation quota. |
| POST | `/api/generate-pdf` | Yes | Generation quota. |
| POST | `/api/feedback` | Yes | Feedback limit (10/hour). |
| POST | `/api/billing/checkout` | Yes | 503 while Stripe is unconfigured. |
| POST | `/api/billing/webhook` | Stripe signature | 503 while webhook secret is unset; 400 on bad signature. |

## License

MIT — see [LICENSE](LICENSE).
