# Free Model Aggregator

One server that fronts ten simulated free-tier model providers behind the two
APIs clients already speak: OpenAI Chat Completions and Anthropic Messages,
each in JSON and in server-sent events.

## Run

```
node main.ts            # serves; registers with $MAKEFUN_ENV_URL, exits 0 when the work is done
node main.ts --list     # prints the catalog document (the same document it registers)
```

The brief at `$MAKEFUN_PROMPT_FILE` supplies the provider directory (wire,
auth scheme, key), the quality table (rating, context, tools/vision/JSON
capabilities, free flag) and the workload manifest. Node 22+, standard library
only.

## How it routes

- At startup it probes every provider's own catalog route in that provider's
  wire (OpenAI, Anthropic or Gemini) and builds one catalog from the listings,
  with the quality table as the truth for free flags and context lengths.
- Each request becomes one canonical request — system text, turns, tools,
  `max_tokens`, stops, images, JSON mode — and is re-encoded in the chosen
  provider's wire; the answer is parsed back and re-emitted in the client's
  wire with the provider's text whole and its usage counts mapped across.
- Candidates are the free models that meet the request's requirements. Because
  the score's tie-break counts provider calls, the router spends them where a
  refusal is least likely: each provider's next-call refusal probability is
  tracked as an exact posterior over the kit's published budget/refusal draw
  (a first call always answers; a provider that has refused once has spent
  most of its scheduled refusals, so its later calls are the safest risky
  calls), and each request is routed to the lowest-risk provider whose rating
  sits within a small allowance of the best — under a trial-wide quality-loss
  budget (0.99% of the scorer's bound) that keeps the quality column at its
  ceiling. A refusal is never the answer: a spent budget or an outage (the
  provider's own message says which) retires that provider for the trial, and
  the request is re-picked by the same rule. A provider that has taken the
  most calls its free tier can ever take in a trial is not called again.
- No wall clock, no randomness: every choice follows from the brief, the
  requests and the providers' answers, so a replay leaves the same state.
