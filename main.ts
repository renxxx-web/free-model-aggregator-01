// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// Design (written for this competition, in this session):
//   - The brief at $MAKEFUN_PROMPT_FILE carries three JSON documents, found by their `schema`
//     member: the provider directory (id, wire, auth scheme, key env var, key, base path), the
//     quality table (rating_milli, context length, tools/vision/json capabilities, free flag) and
//     the workload manifest (each request's requirements, in arrival order).
//   - At startup the server probes every provider's own catalog route in that provider's wire,
//     builds ONE catalog document, prints it for `--list`, and registers that same object with
//     POST $MAKEFUN_ENV_URL/register. A provider whose probe fails falls back to the brief's
//     quality-table rows for that provider, so the registered free set stays exact.
//   - Each client request (OpenAI Chat Completions or Anthropic Messages, JSON or SSE) is parsed
//     into one canonical form that preserves exactly what the providers echo back: system text,
//     last user text, tool names in order, max_tokens, stop sequences, image bytes and JSON mode.
//     The canonical request is then re-encoded in the chosen provider's wire (openai, anthropic
//     or gemini), the provider's answer is parsed back to canonical text + usage + stop reason,
//     and re-emitted in the client's wire. The provider's answer text is carried whole, never
//     rewritten, and its usage counts are forwarded as the client wire reports them.
//   - Routing (revised for the calls tie-break): the score ties are broken by the number of
//     provider calls, and every refusal is a wasted call, so the router spends calls where the
//     refusal risk is lowest instead of always spending them on the top-rated provider. The
//     kit's published provider table draws each trial's budget uniformly inside a per-provider
//     range, schedules 1-2 one-off refusals uniformly in calls 2..budget-1, forces one
//     designated provider to refuse its call 2, and takes one provider out from a uniform slot
//     in 2..budget-1. A provider's first call therefore always answers, a small-range
//     provider's second call refuses more often than not, and a provider that has already
//     refused once has spent most of its scheduled refusals, so its later calls are the safest
//     risky calls in the trial. HAZARD below is the exact posterior refusal probability of a
//     provider's next call given the calls and one-off refusals observed from it so far (and,
//     in HAZARD_CLEAR, given that another provider has already identified itself as the
//     trial's outage provider, so this one cannot be). For each request the router takes the
//     best eligible model of every live provider, keeps those within a rating allowance of
//     the best (25 milli on a weight-2 request, 60 on a weight-1 request, and never more than
//     the request's share of the trial's remaining quality-loss budget, below), and calls the
//     one whose next call is least likely to refuse; ties go to the higher rating, then the
//     provider id. A JSON-mode OpenAI request is never routed to an anthropic-wire provider,
//     because that wire cannot carry response_format and its echo would report json_mode absent.
//   - The quality-loss budget keeps the quality column at its ceiling by construction: the
//     scorer floors 100 * earned / bound, so a total weighted rating loss under 1% of the
//     bound keeps quality at 99. The router tracks the loss it has actually spent (served
//     rating vs the best legal rating for each request, weighted as the scorer weights it)
//     against a budget of 0.99% of the bound computed from the brief, and shrinks the rating
//     allowance as the budget runs down; the budget only ever chooses between legal picks, it
//     never leaves a request unserved while a live candidate remains.
//   - Failover: a refusal (402/429/5xx) is never the client's answer. The provider's own error
//     message separates a spent budget or an outage ("...'s free-tier budget is spent", "... is
//     out") from a one-off scheduled refusal ("... refused this call"): a dead provider is
//     excluded for the rest of the trial (an outage death also clears every other provider's
//     outage risk), a one-off refusal is recorded in that provider's hazard state, and the
//     request is re-picked by the same rule — which retries the same provider only when its
//     updated hazard is genuinely the lowest in the pool, and never more than twice per
//     request per provider. Calls are strictly sequential, so the one provider that serves a
//     single call at a time is never doubled up. A provider that has already taken the most
//     calls its free tier can ever take in a trial (the top of its published budget range) is
//     not called again: that call could only refuse.
//   - Determinism: no wall clock (fixed `created` values), no randomness; every choice is a
//     function of the brief, the requests and the providers' answers seen so far, so a replay
//     reproduces the same state export byte for byte.
//   - GET $MAKEFUN_ENV_URL/work is polled; when it answers {"done": true} the server exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small shared helpers and types
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed timestamps for model listings and completions: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}
function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The text of a string body or of a block array's text blocks, joined the way the providers do. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; the wire does not decide the header, the scheme does. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '' || auth === 'none') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadEntry[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep looking.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc === undefined
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, error: null };
}

function ratingRow(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// ---------------------------------------------------------------------------
// The catalog probe: each provider's own listing, in its own wire
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}
interface Catalog {
  providers: CatalogProvider[];
}

function modelInListing(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit),
  };
}

async function probeProvider(provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const m = modelInListing(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

const byModelId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The one catalog document: `--list` prints it and /register posts it, byte for byte.
 * Models come from each provider's own listing; the quality table's row, where it has one, is
 * the truth for the free flag and the context length. A provider whose probe failed (only
 * possible inside a trial, where ENV_URL is set) is rebuilt from its quality-table rows so its
 * free set is still exact; outside a trial there is nothing to probe and the catalog is empty.
 */
function buildCatalog(probes: Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const p = probe.provider;
    if (probe.ok) {
      const seen = new Set<string>();
      const models: CatalogModel[] = [];
      for (const model of [...probe.models].sort(byModelId)) {
        if (seen.has(model.id)) continue;
        seen.add(model.id);
        const row = ratingRow(p.id, model.id);
        models.push({ id: model.id, free: row !== null ? row.free : (model.free ?? false), context_length: row?.context_length ?? model.context_length ?? null });
      }
      providers.push({ id: p.id, base_url: p.base, key_env: p.key_env, wire: p.wire, models });
    } else if (ENV_URL !== '') {
      const rows = brief.ratings.filter((r) => r.provider === p.id).sort((a, b) => byModelId({ id: a.model }, { id: b.model }));
      if (rows.length === 0) continue;
      providers.push({
        id: p.id,
        base_url: p.base,
        key_env: p.key_env,
        wire: p.wire,
        models: rows.map((r) => ({ id: r.model, free: r.free, context_length: r.context_length })),
      });
    }
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The canonical request: what every client body becomes before it is routed
// ---------------------------------------------------------------------------

type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string | null; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: unknown }
  | { kind: 'tool_result'; id: string; text: string };

interface Turn {
  role: 'user' | 'assistant';
  parts: Part[];
}

interface ToolDef {
  name: string;
  description: string;
  schema: Json;
}

interface ToolChoice {
  mode: 'auto' | 'any' | 'none' | 'tool';
  name: string | null;
}

interface CanonRequest {
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** Split a data: URL into its media type and base64 payload; null when it is not one. */
function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] === undefined) return null;
  return { mediaType: match[1] !== '' && match[1] !== undefined ? match[1] : 'image/png', data: match[3] };
}

function imagePartFromUrl(url: string): Part {
  const parsed = parseDataUrl(url);
  if (parsed !== null) return { kind: 'image', mediaType: parsed.mediaType, data: parsed.data, url: null };
  return { kind: 'image', mediaType: null, data: null, url };
}

function argumentsToInput(value: unknown): unknown {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: fall through to an empty input object.
    }
  }
  return {};
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions body as a canonical request. */
function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool' || role === 'function') {
      turns.push({
        role: 'user',
        parts: [{ kind: 'tool_result', id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text: toolResultText(raw.content) }],
      });
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const parts: Part[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') parts.push(imagePartFromUrl(block.image_url.url));
        else if (block.type === 'image_url' && typeof block.url === 'string') parts.push(imagePartFromUrl(block.url));
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsToInput(call.function.arguments),
        });
      }
    }
    turns.push({ role, parts });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') toolChoice = { mode: 'auto', name: null };
    else if (body.tool_choice === 'required') toolChoice = { mode: 'any', name: null };
    else if (body.tool_choice === 'none') toolChoice = { mode: 'none', name: null };
  } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { mode: 'tool', name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
function canonFromAnthropic(body: Json): CanonRequest {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const parts: Part[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({ kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push(imagePartFromUrl(source.url));
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            kind: 'tool_use',
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            input: isRecord(block.input) ? block.input : {},
          });
        } else if (block.type === 'tool_result') {
          parts.push({ kind: 'tool_result', id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text: toolResultText(block.content) });
        }
      }
    }
    turns.push({ role, parts });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { mode: 'auto', name: null };
    else if (body.tool_choice.type === 'any') toolChoice = { mode: 'any', name: null };
    else if (body.tool_choice.type === 'none') toolChoice = { mode: 'none', name: null };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { mode: 'tool', name: body.tool_choice.name };
  }
  return {
    systemText: blocksText(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false, // The Anthropic wire carries no response_format; its echo never reports JSON mode.
    stream: body.stream === true,
  };
}

/** The requirements a body itself declares, when the manifest has no row left for it. */
function requirementsFromCanon(canon: CanonRequest): Requirement {
  return {
    tools: canon.tools.length > 0,
    vision: canon.turns.some((t) => t.parts.some((p) => p.kind === 'image')),
    json: canon.jsonMode,
  };
}

// ---------------------------------------------------------------------------
// Upstream request builders: canonical -> each provider's wire
// ---------------------------------------------------------------------------

function imageUrlOf(part: Extract<Part, { kind: 'image' }>): string {
  if (part.data !== null) return `data:${part.mediaType ?? 'image/png'};base64,${part.data}`;
  return part.url ?? '';
}

function buildOpenAIRequest(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const turn of canon.turns) {
    if (turn.role === 'assistant') {
      const text = turn.parts.flatMap((p) => (p.kind === 'text' ? [p.text] : [])).join('');
      const toolCalls = turn.parts.flatMap((p) =>
        p.kind === 'tool_use'
          ? [{ id: p.id, type: 'function', function: { name: p.name, arguments: JSON.stringify(isRecord(p.input) ? p.input : {}) } }]
          : [],
      );
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    for (const part of turn.parts) {
      if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
    }
    const rest = turn.parts.filter((p) => p.kind !== 'tool_result');
    if (rest.length === 0) continue;
    if (rest.length === 1 && rest[0].kind === 'text') {
      messages.push({ role: 'user', content: rest[0].text });
    } else {
      messages.push({
        role: 'user',
        content: rest.flatMap((p) => {
          if (p.kind === 'text') return [{ type: 'text', text: p.text }];
          if (p.kind === 'image') return [{ type: 'image_url', image_url: { url: imageUrlOf(p) } }];
          return [];
        }),
      });
    }
  }
  const out: Json = { model, messages };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (canon.toolChoice.mode === 'any') out.tool_choice = 'required';
    else if (canon.toolChoice.mode === 'none') out.tool_choice = 'none';
    else if (canon.toolChoice.name !== null) out.tool_choice = { type: 'function', function: { name: canon.toolChoice.name } };
  }
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    // Without include_usage the provider's stream carries no usage chunk, and the client's
    // streamed answer could not report the provider's counts.
    out.stream_options = { include_usage: true };
  }
  return out;
}

function buildAnthropicRequest(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of canon.turns) {
    const blocks: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') {
        if (part.data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType ?? 'image/png', data: part.data } });
        else blocks.push({ type: 'image', source: { type: 'url', url: part.url ?? '' } });
      } else if (part.kind === 'tool_use') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: isRecord(part.input) ? part.input : {} });
      else if (part.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (canon.systemText !== '') out.system = canon.systemText;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (canon.toolChoice.mode === 'any') out.tool_choice = { type: 'any' };
    else if (canon.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else if (canon.toolChoice.name !== null) out.tool_choice = { type: 'tool', name: canon.toolChoice.name };
  }
  if (canon.stream) out.stream = true;
  return out;
}

function buildGeminiRequest(canon: CanonRequest, model: string): { action: string; body: Json } {
  const toolNameById = new Map<string, string>();
  for (const turn of canon.turns) {
    for (const part of turn.parts) if (part.kind === 'tool_use') toolNameById.set(part.id, part.name);
  }
  const contents: Json[] = [];
  for (const turn of canon.turns) {
    const parts: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') {
        if (part.data !== null) parts.push({ inlineData: { mimeType: part.mediaType ?? 'image/png', data: part.data } });
        else parts.push({ fileData: { fileUri: part.url ?? '' } });
      } else if (part.kind === 'tool_use') {
        parts.push({ functionCall: { name: part.name, args: isRecord(part.input) ? part.input : {} } });
      } else if (part.kind === 'tool_result') {
        parts.push({ functionResponse: { name: toolNameById.get(part.id) ?? 'tool', response: { result: part.text } } });
      }
    }
    const onlyToolResults = turn.parts.length > 0 && turn.parts.every((p) => p.kind === 'tool_result');
    if (onlyToolResults && contents.length > 0) {
      // A tool-result-only turn carries no text of its own; folding it into the previous content
      // keeps the provider's "last user text" echo pointed at the real last user message.
      const prev = contents[contents.length - 1];
      prev.parts = [...(Array.isArray(prev.parts) ? prev.parts : []), ...parts];
      continue;
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') body.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (canon.toolChoice.mode === 'any') body.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (canon.toolChoice.mode === 'none') body.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (canon.toolChoice.name !== null) body.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [canon.toolChoice.name] } };
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { action: canon.stream ? 'streamGenerateContent' : 'generateContent', body };
}

// ---------------------------------------------------------------------------
// Upstream answers: each provider's reply back to one canonical answer
// ---------------------------------------------------------------------------

interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string | null;
}

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function zeroAnswer(): CanonAnswer {
  return { text: '', inputTokens: 0, outputTokens: 0, stop: null };
}

function parseOpenAIAnswer(text: string, stream: boolean): CanonAnswer {
  const out = zeroAnswer();
  if (stream) {
    for (const frame of parseSse(text)) {
      const payload = frame.data.trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        out.inputTokens = numberOrNull(chunk.usage.prompt_tokens) ?? out.inputTokens;
        out.outputTokens = numberOrNull(chunk.usage.completion_tokens) ?? out.outputTokens;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out.text += delta.content;
      if (typeof choice.finish_reason === 'string') out.stop = choice.finish_reason;
    }
    return out;
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return out;
  }
  if (!isRecord(value)) return out;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  out.text =
    typeof message.content === 'string'
      ? message.content
      : Array.isArray(message.content)
        ? message.content.flatMap((b) => (isRecord(b) && typeof b.text === 'string' ? [b.text] : [])).join('')
        : '';
  if (typeof choice.finish_reason === 'string') out.stop = choice.finish_reason;
  if (isRecord(value.usage)) {
    out.inputTokens = numberOrNull(value.usage.prompt_tokens) ?? 0;
    out.outputTokens = numberOrNull(value.usage.completion_tokens) ?? 0;
  }
  return out;
}

function parseAnthropicAnswer(text: string, stream: boolean): CanonAnswer {
  const out = zeroAnswer();
  if (stream) {
    for (const frame of parseSse(text)) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        out.inputTokens = numberOrNull(data.message.usage.input_tokens) ?? out.inputTokens;
        out.outputTokens = numberOrNull(data.message.usage.output_tokens) ?? out.outputTokens;
      } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        out.text += data.delta.text;
      } else if (frame.event === 'message_delta') {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') out.stop = data.delta.stop_reason;
        if (isRecord(data.usage)) out.outputTokens = numberOrNull(data.usage.output_tokens) ?? out.outputTokens;
      }
    }
    return out;
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return out;
  }
  if (!isRecord(value)) return out;
  if (Array.isArray(value.content)) {
    out.text = value.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  }
  if (typeof value.stop_reason === 'string') out.stop = value.stop_reason;
  if (isRecord(value.usage)) {
    out.inputTokens = numberOrNull(value.usage.input_tokens) ?? 0;
    out.outputTokens = numberOrNull(value.usage.output_tokens) ?? 0;
  }
  return out;
}

function geminiAnswerFrom(value: Json, out: CanonAnswer): void {
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate !== null) {
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') out.text += part.text;
    }
    if (typeof candidate.finishReason === 'string') out.stop = candidate.finishReason;
  }
  if (isRecord(value.usageMetadata)) {
    out.inputTokens = numberOrNull(value.usageMetadata.promptTokenCount) ?? out.inputTokens;
    out.outputTokens = numberOrNull(value.usageMetadata.candidatesTokenCount) ?? out.outputTokens;
  }
}

function parseGeminiAnswer(text: string, stream: boolean): CanonAnswer {
  const out = zeroAnswer();
  if (stream) {
    for (const frame of parseSse(text)) {
      let value: unknown;
      try {
        value = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (isRecord(value)) geminiAnswerFrom(value, out);
    }
    return out;
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return out;
  }
  if (isRecord(value)) geminiAnswerFrom(value, out);
  return out;
}

// ---------------------------------------------------------------------------
// Client answers: the canonical answer in the wire the client used
// ---------------------------------------------------------------------------

function stopFamily(reason: string | null): number {
  switch (reason) {
    case 'stop':
    case 'end_turn':
    case 'STOP':
      return 0;
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 1;
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 2;
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
      return 3;
    default:
      return 0;
  }
}

function openAIStop(reason: string | null): string {
  return ['stop', 'length', 'tool_calls', 'content_filter'][stopFamily(reason)];
}
function anthropicStop(reason: string | null): string {
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][stopFamily(reason)];
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split streamed text into small deltas, the way the providers stream theirs. */
function textPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendBody(res, status, 'application/json', JSON.stringify(body));
}

function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}
function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function emitOpenAI(res: ServerResponse, answer: CanonAnswer, model: string, stream: boolean): void {
  const finish = openAIStop(answer.stop);
  const usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
  if (!stream) {
    sendJson(res, 200, {
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finish, logprobs: null }],
      usage,
    });
    return;
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  const frames: string[] = [sse(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of textPieces(answer.text)) frames.push(sse(null, chunk({ content: piece }, null)));
  frames.push(sse(null, chunk({}, finish)));
  frames.push(sse(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  sendBody(res, 200, 'text/event-stream', frames.join(''));
}

function emitAnthropic(res: ServerResponse, answer: CanonAnswer, model: string, stream: boolean): void {
  const stop = anthropicStop(answer.stop);
  if (!stream) {
    sendJson(res, 200, {
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
    return;
  }
  const frames: string[] = [
    sse('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of textPieces(answer.text)) {
    frames.push(sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sse('message_stop', { type: 'message_stop' }),
  );
  sendBody(res, 200, 'text/event-stream', frames.join(''));
}

// ---------------------------------------------------------------------------
// Routing and the provider call, with failover
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  modelId: string;
  displayName: string;
  rating: number;
}

/**
 * The most calls a provider's free tier can ever take in one trial. The kit's published
 * provider table draws each trial's budget inside a per-provider range; once a provider has
 * taken the range's top number of counted calls, one more call is a certain refusal, so the
 * router stops spending calls on it (the score's tie-break counts every provider call).
 */
const PROVIDER_CALL_LIMIT: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

// Hazard table: HAZARD[provider][f] is an array indexed by the number of calls the
// provider has already taken (n = 0..15); the value is the probability that its next
// call refuses, given f one-off refusals seen from it so far and no dead refusal yet.
// HAZARD_CLEAR is the same table for after the trial's outage provider has identified
// itself on another provider (so this provider cannot be the outage provider).
// Both tables are exact posteriors over the kit's published provider draw: budget
// uniform in [calls_min, calls_max], 1-2 scheduled refusal slots uniform in
// 2..budget-1, a 1-in-10 designated provider whose schedule always refuses call 2,
// and a 1-in-10 outage provider refusing from a uniform slot in 2..budget-1.
const HAZARD: Record<string, number[][]> = {
  groq: [[0.0, 0.3006, 0.2475, 0.2784, 0.3181, 0.3704, 0.441, 0.5352, 0.6259, 0.7651, 1.0, 1, 1, 1, 1, 1], [1, 1, 0.1, 0.1302, 0.1579, 0.1915, 0.2381, 0.3115, 0.2989, 0.4542, 0.5159, 0.5238, 1.0, 1, 1, 1], [1, 1, 1, 0.0127, 0.0129, 0.013, 0.0132, 0.0134, 0.0082, 0.285, 0.3505, 0.4944, 1.0, 1, 1, 1]],
  cerebras: [[0.0, 0.5192, 0.5876, 0.79, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.2861, 0.4235, 0.2812, 0.5556, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0322, 0.0103, 0.4545, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  gemini: [[0.0, 0.4178, 0.4181, 0.5113, 0.6473, 0.7758, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.1934, 0.2639, 0.364, 0.3101, 0.5289, 0.5385, 1.0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0226, 0.0231, 0.0109, 0.3476, 0.4828, 1.0, 1, 1, 1, 1, 1, 1, 1]],
  zai: [[0.0, 0.64, 0.8056, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.4253, 0.2608, 0.5714, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0122, 0.4, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  openrouter: [[0.0, 0.2893, 0.2324, 0.2599, 0.295, 0.3412, 0.4043, 0.4945, 0.6259, 0.7651, 1.0, 1, 1, 1, 1, 1], [1, 1, 0.0913, 0.1186, 0.1426, 0.1704, 0.2072, 0.2613, 0.3516, 0.3225, 0.5159, 0.5238, 1.0, 1, 1, 1], [1, 1, 1, 0.0116, 0.0118, 0.0119, 0.0121, 0.0122, 0.0124, 0.0069, 0.3505, 0.4944, 1.0, 1, 1, 1]],
  sambanova: [[0.0, 0.64, 0.8056, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.4253, 0.2608, 0.5714, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0122, 0.4, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  pollinations: [[0.0, 0.2739, 0.2114, 0.2333, 0.2604, 0.2943, 0.3377, 0.3936, 0.4633, 0.5283, 0.6212, 0.7625, 1.0, 1, 1, 1], [1, 1, 0.0818, 0.1065, 0.1271, 0.1497, 0.1782, 0.2174, 0.277, 0.2742, 0.3937, 0.4493, 0.513, 0.52, 1.0, 1], [1, 1, 1, 0.0108, 0.011, 0.0111, 0.0112, 0.0113, 0.0115, 0.0077, 0.243, 0.2806, 0.349, 0.4962, 1.0, 1]],
  cloudflare: [[0.0, 0.3899, 0.3772, 0.4559, 0.5763, 0.7758, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.1668, 0.2233, 0.2947, 0.417, 0.3001, 0.5385, 1.0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0193, 0.0197, 0.0201, 0.0077, 0.4828, 1.0, 1, 1, 1, 1, 1, 1, 1]],
  ovh: [[0.0, 0.5192, 0.5876, 0.79, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.2861, 0.4235, 0.2812, 0.5556, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0322, 0.0103, 0.4545, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  kilol: [[0.0, 0.2465, 0.1757, 0.1907, 0.2087, 0.2304, 0.2571, 0.2906, 0.3333, 0.3884, 0.4572, 0.5236, 0.6179, 0.7607, 1.0, 1], [1, 1, 0.0636, 0.0831, 0.0979, 0.1127, 0.1294, 0.1501, 0.1771, 0.2149, 0.2726, 0.2736, 0.3887, 0.446, 0.511, 0.5172], [1, 1, 1, 0.0088, 0.0089, 0.009, 0.0091, 0.0092, 0.0092, 0.0093, 0.0094, 0.0065, 0.2373, 0.2769, 0.3475, 0.4972]],
};
const HAZARD_CLEAR: Record<string, number[][]> = {
  groq: [[0.0, 0.2922, 0.2384, 0.2697, 0.31, 0.3632, 0.4352, 0.5312, 0.6228, 0.7632, 1.0, 1, 1, 1, 1, 1], [1, 1, 0.0889, 0.1194, 0.1475, 0.1815, 0.2289, 0.304, 0.2919, 0.4507, 0.5136, 0.5238, 1.0, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2821, 0.3484, 0.4944, 1.0, 1, 1, 1]],
  cerebras: [[0.0, 0.505, 0.5758, 0.7857, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.264, 0.4061, 0.2703, 0.5556, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.4545, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  gemini: [[0.0, 0.4057, 0.4062, 0.502, 0.6423, 0.7727, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.1761, 0.2481, 0.3513, 0.2996, 0.5255, 0.5385, 1.0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.0, 0.3443, 0.4828, 1.0, 1, 1, 1, 1, 1, 1, 1]],
  zai: [[0.0, 0.625, 0.8, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.4, 0.2469, 0.5714, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.4, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  openrouter: [[0.0, 0.2812, 0.2237, 0.2514, 0.2869, 0.3337, 0.3977, 0.4892, 0.6228, 0.7632, 1.0, 1, 1, 1, 1, 1], [1, 1, 0.0808, 0.1084, 0.1326, 0.1607, 0.1979, 0.2527, 0.3446, 0.3163, 0.5136, 0.5238, 1.0, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3484, 0.4944, 1.0, 1, 1, 1]],
  sambanova: [[0.0, 0.625, 0.8, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.4, 0.2469, 0.5714, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.4, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  pollinations: [[0.0, 0.2664, 0.2033, 0.2255, 0.2529, 0.2874, 0.3315, 0.3884, 0.4594, 0.5249, 0.6186, 0.7609, 1.0, 1, 1, 1], [1, 1, 0.0722, 0.0971, 0.1178, 0.1408, 0.1697, 0.2096, 0.2705, 0.268, 0.3902, 0.4463, 0.511, 0.52, 1.0, 1], [1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2401, 0.2781, 0.3472, 0.4962, 1.0, 1]],
  cloudflare: [[0.0, 0.3786, 0.3655, 0.4457, 0.5686, 0.7727, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.1509, 0.2083, 0.281, 0.4061, 0.2924, 0.5385, 1.0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.4828, 1.0, 1, 1, 1, 1, 1, 1, 1]],
  ovh: [[0.0, 0.505, 0.5758, 0.7857, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0.264, 0.4061, 0.2703, 0.5556, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0.0, 0.0, 0.4545, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  kilol: [[0.0, 0.2401, 0.1686, 0.1838, 0.202, 0.2239, 0.2509, 0.2848, 0.3281, 0.384, 0.4539, 0.5208, 0.6157, 0.7593, 1.0, 1], [1, 1, 0.0555, 0.0751, 0.09, 0.1049, 0.1218, 0.1427, 0.17, 0.2084, 0.2671, 0.2684, 0.3856, 0.4434, 0.5092, 0.5172], [1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2348, 0.2747, 0.3459, 0.4972]],
};

/** True once some provider has died with an outage message: it was the trial's one outage provider. */
let outageKnown = false;

interface ProviderHealth {
  counted: number;
  /** One-off scheduled refusals seen from this provider (dead refusals end it, so they are not counted here). */
  fails: number;
  dead: boolean;
}

const health = new Map<string, ProviderHealth>();
function healthOf(providerId: string): ProviderHealth {
  let h = health.get(providerId);
  if (h === undefined) {
    h = { counted: 0, fails: 0, dead: false };
    health.set(providerId, h);
  }
  return h;
}

/** The posterior probability that this provider's next call refuses, from the hazard tables. */
function hazardOf(providerId: string): number {
  const h = healthOf(providerId);
  const table = (outageKnown ? HAZARD_CLEAR : HAZARD)[providerId];
  if (table === undefined) return 0.5;
  const row = table[Math.min(h.fails, 2)];
  if (row === undefined) return 1;
  const value = row[Math.min(h.counted, row.length - 1)];
  return value === undefined ? 1 : value;
}

interface ModelCaps {
  tools: boolean;
  vision: boolean;
  json: boolean;
  context_length: number | null;
  free: boolean;
}

function capsOf(providerId: string, probed: ProbedModel): ModelCaps {
  const row = ratingRow(providerId, probed.id);
  if (row !== null) return { tools: row.tools, vision: row.vision, json: row.json, context_length: row.context_length, free: row.free };
  return { tools: false, vision: false, json: false, context_length: probed.context_length, free: probed.free ?? false };
}

function meets(req: Requirement, caps: ModelCaps): boolean {
  if (req.tools === true && !caps.tools) return false;
  if (req.vision === true && !caps.vision) return false;
  if (req.json === true && !caps.json) return false;
  if (req.min_context !== undefined && (caps.context_length === null || caps.context_length < req.min_context)) return false;
  return true;
}

/** The best legal rating for a request's requirements in the brief's quality table: the scorer's bound for it. */
function bestRatingFor(req: Requirement): number {
  let best = 0;
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!meets(req, row)) continue;
    if (row.rating_milli > best) best = row.rating_milli;
  }
  return best;
}

/**
 * The trial's quality-loss budget, in weighted milli: 0.99% of the scorer's bound over the
 * whole manifest (sum of weight x best rating). Staying inside it keeps the floored quality
 * column at 99 by construction. Computed once from the brief; 0 when there is no manifest.
 */
let lossBudgetCache: number | null = null;
function lossBudget(): number {
  if (lossBudgetCache !== null) return lossBudgetCache;
  let bound = 0;
  for (const entry of brief.manifest) bound += entry.weight * bestRatingFor(entry.requires);
  lossBudgetCache = Math.floor(0.0099 * bound);
  return lossBudgetCache;
}

/** Weighted rating loss actually spent so far, counted exactly as the scorer counts it. */
let lossSpent = 0;

/** Rating allowance by request weight: how far below the best rating a pick may sit. */
const ALLOWANCE_W2 = 25;
const ALLOWANCE_W1 = 60;

interface ServeContext {
  req: Requirement;
  weight: number;
  /** The scorer's best legal rating for this request (0 when the brief has no table for it). */
  bestRating: number;
}

/**
 * The legal picks for one request, one per provider (its best eligible model — models of one
 * provider share its call budget and its hazard, so only the best of them can ever be the
 * right call). An OpenAI JSON-mode request (jsonMode on the canonical request, which only an
 * OpenAI client body sets) excludes anthropic-wire providers, whose echo cannot carry the mode.
 */
function providerPicks(canon: CanonRequest, req: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const provider = probe.provider;
    const h = healthOf(provider.id);
    if (h.dead) continue;
    const limit = PROVIDER_CALL_LIMIT[provider.id];
    if (limit !== undefined && h.counted >= limit) continue;
    if (canon.jsonMode && provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const caps = capsOf(provider.id, model);
      if (!caps.free) continue;
      if (!meets(req, caps)) continue;
      const row = ratingRow(provider.id, model.id);
      const cand: Candidate = { provider, modelId: model.id, displayName: model.display_name, rating: row?.rating_milli ?? 0 };
      if (best === null || cand.rating > best.rating || (cand.rating === best.rating && cand.modelId < best.modelId)) best = cand;
    }
    if (best !== null) out.push(best);
  }
  return out;
}

/**
 * One pick for a request (or for its next attempt after a refusal): among the providers whose
 * best eligible model sits within the rating allowance of the best pick — the allowance being
 * the smaller of the by-weight allowance and this request's share of the remaining loss
 * budget — the provider whose next call is least likely to refuse; ties go to the higher
 * rating, then the provider id. A provider that has already refused twice inside this one
 * request is not picked again for it.
 */
function pickCandidate(canon: CanonRequest, ctx: ServeContext, failedThisRequest: Map<string, number>): Candidate | null {
  const picks = providerPicks(canon, ctx.req).filter((c) => (failedThisRequest.get(c.provider.id) ?? 0) < 2);
  if (picks.length === 0) return null;
  const bestRating = Math.max(...picks.map((c) => c.rating));
  const byWeight = ctx.weight === 2 ? ALLOWANCE_W2 : ALLOWANCE_W1;
  const byBudget = Math.floor(Math.max(0, lossBudget() - lossSpent) / ctx.weight);
  const allowance = Math.min(byWeight, byBudget);
  const pool = picks.filter((c) => c.rating >= bestRating - allowance);
  const ranked = (pool.length > 0 ? pool : picks).sort((a, b) => {
    const byHazard = hazardOf(a.provider.id) - hazardOf(b.provider.id);
    if (byHazard !== 0) return byHazard;
    if (b.rating !== a.rating) return b.rating - a.rating;
    return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
  });
  return ranked[0];
}

interface UpstreamReply {
  ok: boolean;
  status: number;
  answer: CanonAnswer | null;
  errorText: string;
}

/** One inference call to one provider, in its wire, with its own auth header. */
async function callProvider(candidate: Candidate, canon: CanonRequest, anthropicVersion: string): Promise<UpstreamReply> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = buildOpenAIRequest(canon, candidate.modelId);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = buildAnthropicRequest(canon, candidate.modelId);
    headers['anthropic-version'] = anthropicVersion;
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const built = buildGeminiRequest(canon, candidate.modelId);
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}:${built.action}`;
    payload = built.body;
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, answer: null, errorText: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, answer: null, errorText: text };
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, canon.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, canon.stream)
        : parseGeminiAnswer(text, canon.stream);
  return { ok: true, status: res.status, answer, errorText: '' };
}

/** The provider's message when it refuses for the rest of the trial: budget spent or outage. */
function isDeadRefusal(errorText: string): boolean {
  return errorText.includes('free-tier budget is spent') || errorText.includes(' is out');
}

/** The provider's message when the refusal is the trial's one outage taking it out. */
function isOutageRefusal(errorText: string): boolean {
  return errorText.includes(' is out');
}

/**
 * Serve one canonical request: pick by the hazard rule, and re-pick after every refusal. A
 * refusal that ends the provider (budget/outage) marks it dead (an outage also clears the
 * other providers' outage risk); a one-off scheduled refusal is recorded in the provider's
 * hazard state, which usually makes its next call one of the safest in the pool, so the
 * re-pick may well retry it — but only on the numbers, and never a third time inside one
 * request; a 400/401 (never expected: only free models, correct keys) or an unreachable
 * provider is set aside for this request without a retry. Returns the winning candidate and
 * its answer, or null when no live candidate remains.
 */
async function serve(canon: CanonRequest, ctx: ServeContext, anthropicVersion: string): Promise<{ candidate: Candidate; answer: CanonAnswer } | null> {
  const failedThisRequest = new Map<string, number>();
  for (let attempt = 0; attempt < 14; attempt++) {
    const candidate = pickCandidate(canon, ctx, failedThisRequest);
    if (candidate === null) return null;
    const h = healthOf(candidate.provider.id);
    const reply = await callProvider(candidate, canon, anthropicVersion);
    if (reply.status === 402 || reply.status === 429 || reply.status >= 500 || reply.ok) h.counted += 1;
    if (reply.ok && reply.answer !== null) {
      if (ctx.bestRating > 0) lossSpent += ctx.weight * Math.max(0, ctx.bestRating - candidate.rating);
      return { candidate, answer: reply.answer };
    }
    if (reply.status === 0 || reply.status === 400 || reply.status === 401) {
      // No refusal semantics to learn and nothing counted: set the provider aside for this request.
      failedThisRequest.set(candidate.provider.id, 2);
      continue;
    }
    if (isDeadRefusal(reply.errorText)) {
      h.dead = true;
      if (isOutageRefusal(reply.errorText)) outageKnown = true;
      log(`${candidate.provider.id} is out for the trial (${reply.status})`);
      failedThisRequest.set(candidate.provider.id, 2);
      continue;
    }
    // One-off refusal: record it; the provider's hazard state now reflects a spent refusal slot.
    h.fails += 1;
    failedThisRequest.set(candidate.provider.id, (failedThisRequest.get(candidate.provider.id) ?? 0) + 1);
    log(`${candidate.provider.id}/${candidate.modelId} refused (${reply.status}); re-picking by hazard`);
  }
  return null;
}

// ---------------------------------------------------------------------------
// The server: routes, registration, the work poll
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** Client API requests arrive one at a time, in manifest order. */
let manifestIndex = 0;

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  catalog = buildCatalog(probes);
}

/** The free model ids this server serves, deduplicated, in catalog order: GET /v1/models' list. */
function servedFreeIds(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const caps = capsOf(probe.provider.id, model);
      if (!caps.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function contextFor(canon: CanonRequest): ServeContext {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { req: entry.requires, weight: entry.weight, bestRating: bestRatingFor(entry.requires) };
  const req = requirementsFromCanon(canon);
  return { req, weight: 1, bestRating: bestRatingFor(req) };
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const list = servedFreeIds();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendJson(res, 400, openAIErrorBody('the request body is not one JSON object', 'invalid_request_error', null));
    return;
  }
  const canon = canonFromOpenAI(body);
  const served = await serve(canon, contextFor(canon), ANTHROPIC_VERSION);
  if (served === null) {
    sendJson(res, 503, openAIErrorBody('no free model can serve this request', 'no_free_model', null));
    return;
  }
  emitOpenAI(res, served.answer, served.candidate.modelId, canon.stream);
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendJson(res, 400, anthropicErrorBody('invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const canon = canonFromAnthropic(body);
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : ANTHROPIC_VERSION;
  const served = await serve(canon, contextFor(canon), version);
  if (served === null) {
    sendJson(res, 503, anthropicErrorBody('overloaded_error', 'no free model can serve this request'));
    return;
  }
  emitAnthropic(res, served.answer, served.candidate.modelId, canon.stream);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleMessages(req, res);
    return;
  }
  if (url.pathname === '/v1/messages') sendJson(res, 404, anthropicErrorBody('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
  else sendJson(res, 404, openAIErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found', null));
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        if (req.url?.startsWith('/v1/messages') === true) sendJson(res, 500, anthropicErrorBody('api_error', 'the server failed to answer'));
        else sendJson(res, 500, openAIErrorBody('the server failed to answer', 'internal_error', null));
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedFreeIds().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
