// gta6-twitch-ccu-forecast-01 — variant A, re-typed from the prose spec.
// k-NN nearest-analog forecast, k = 6, over the kit's comparable launches:
// each launch contributes a log-ratio curve ln(v_h / v_1) for hours 2..24,
// the case's median curve is the analog-weighted median of those curves,
// level-corrected with a rolling-family OLS slope on ln(p1), widened by a
// calibrated spread, and capped by a family cap. Deterministic, no clock,
// no randomness, Node stdlib only. I/O contract per the kit starter:
// read <id>.in from MAKEFUN_INPUT_DIR (else /input), write <id>.out to
// MAKEFUN_OUTPUT_DIR (else /output).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Number of nearest analogs. */
const K = 6;
/** Analog-weight offset: weight = 1 / (distance + OFFSET). */
const OFFSET = 0.29;
/** Feature weights for the weighted-RMS distance. */
const FW = [1.0, 0.7, 0.7, 0.4];
/** House persistence spread, used only by the <3-analog fallback. */
const SIGMA = 0.15;

interface Point {
  from?: string;
  at?: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: Point[];
  unlock: string;
}

interface CaseFile {
  id: string;
  day_start: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels?: number | null }>;
  forecast: Array<{ hour: number }>;
  history: Point[];
  comparables: Comparable[];
}

interface Launch {
  name: string;
  p1: number;
  ch1: number;
  /** ln(hour-1 print), ln(48h pre-launch peak), ln(7d median baseline), ln(ch1) - ln(p1). */
  feat: number[];
  /** Hour (1..24) -> { viewers, channels } on the launch's UTC day. */
  curve: Map<number, { viewers: number; channels: number | null }>;
  /** Largest hour-2..24 ratio v_h / v_1 (Infinity when hour 1 is missing). */
  maxRatio: number;
  tier: string;
  rolling: boolean;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** Plain median (even count: mean of the two middle values). */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return NaN;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Step-function weighted median: the smallest value whose cumulative weight reaches half the total. */
function weightedMedian(pairs: Array<[number, number]>): number {
  const s = [...pairs].sort((a, b) => a[0] - b[0]);
  const total = s.reduce((acc, p) => acc + p[1], 0);
  let cum = 0;
  for (const [v, w] of s) {
    cum += w;
    if (cum >= 0.5 * total) return v;
  }
  return s[s.length - 1][0];
}

function weightedMean(pairs: Array<[number, number]>): number {
  let sw = 0;
  let sx = 0;
  for (const [v, w] of pairs) {
    sw += w;
    sx += w * v;
  }
  return sx / sw;
}

/** Weighted population variance around the weighted mean. */
function weightedPopVar(pairs: Array<[number, number]>): number {
  const m = weightedMean(pairs);
  let sw = 0;
  let s = 0;
  for (const [v, w] of pairs) {
    sw += w;
    s += w * (v - m) * (v - m);
  }
  return s / sw;
}

/** Unweighted population variance. */
function popVar(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const m = values.reduce((a, b) => a + b, 0) / n;
  return values.reduce((a, v) => a + (v - m) * (v - m), 0) / n;
}

/** Unweighted OLS slope of y on x (0 when the slope is undefined). */
function olsSlope(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  return sxx === 0 ? 0 : sxy / sxx;
}

/** The "Tier:" value parsed from a comparable's unlock text. */
function parseTier(unlock: string): string {
  const m = /Tier:\s*([A-Za-z-]+)/.exec(unlock ?? '');
  return m ? m[1] : '';
}

/** Rolling-midnight family: midnight-local unlock, not simultaneous / single-global. */
function isRolling(unlock: string): boolean {
  return /midnight local/i.test(unlock ?? '') && !/simultaneous|single global/i.test(unlock ?? '');
}

/** A launch's UTC-day curve: hour 1..24 -> that hour's point (last point wins a tie). */
function dayCurve(dayStartMs: number, points: Point[]): Map<number, { viewers: number; channels: number | null }> {
  const byHour = new Map<number, { viewers: number; channels: number | null }>();
  for (const pt of points ?? []) {
    const t = new Date(pt.from ?? pt.at ?? '').getTime();
    if (Number.isNaN(t)) continue;
    const h = Math.round((t - dayStartMs) / 3600000) + 1;
    if (h >= 1 && h <= 24 && pt.viewers != null && pt.viewers > 0) {
      byHour.set(h, { viewers: pt.viewers, channels: pt.channels ?? null });
    }
  }
  return byHour;
}

/** 48h pre-launch peak and 7d pre-launch median baseline, from points before day start. */
function preFeatures(dayStartMs: number, points: Point[]): { peak48: number; med7d: number } {
  const p48: number[] = [];
  const p7d: number[] = [];
  for (const pt of points ?? []) {
    const t = new Date(pt.from ?? pt.at ?? '').getTime();
    if (Number.isNaN(t) || t >= dayStartMs || pt.viewers == null || pt.viewers <= 0) continue;
    if (t >= dayStartMs - 48 * 3600000) p48.push(pt.viewers);
    if (t >= dayStartMs - 7 * 86400000) p7d.push(pt.viewers);
  }
  return {
    peak48: p48.length ? Math.max(...p48) : NaN,
    med7d: p7d.length ? median(p7d) : NaN,
  };
}

/** Feature vector + curve for one launch (comparable or case). */
function buildLaunch(
  name: string,
  dayStartMs: number,
  points: Point[],
  p1: number,
  ch1: number,
  unlock: string,
): Launch {
  const curve = dayCurve(dayStartMs, points);
  const { peak48, med7d } = preFeatures(dayStartMs, points);
  const lp1 = Math.log(Math.max(1, p1));
  const feat = [
    lp1,
    Math.log(Math.max(1, Number.isNaN(peak48) ? p1 : peak48)),
    Math.log(Math.max(1, Number.isNaN(med7d) ? p1 : med7d)),
    Math.log(Math.max(1, ch1)) - lp1,
  ];
  let maxRatio = 0;
  const h1 = curve.get(1);
  if (h1) {
    for (const [h, v] of curve) {
      if (h >= 2) maxRatio = Math.max(maxRatio, v.viewers / h1.viewers);
    }
  } else {
    maxRatio = Infinity;
  }
  return { name, p1: Math.max(1, p1), ch1: Math.max(1, ch1), feat, curve, maxRatio, tier: parseTier(unlock), rolling: isRolling(unlock) };
}

/** Standardize each feature by the eligible comparables' mean + population SD. */
function standardize(feats: number[][]): { mean: number[]; sd: number[] } {
  const mean: number[] = [];
  const sd: number[] = [];
  for (let j = 0; j < 4; j++) {
    const col = feats.map((f) => f[j]);
    const m = col.reduce((a, b) => a + b, 0) / col.length;
    const v = col.reduce((a, x) => a + (x - m) * (x - m), 0) / col.length;
    mean.push(m);
    sd.push(Math.sqrt(v) || 1);
  }
  return { mean, sd };
}

/** Weighted-RMS distance between two standardized feature vectors. */
function distance(a: number[], b: number[], mean: number[], sd: number[]): number {
  const wSum = FW.reduce((x, y) => x + y, 0);
  let s = 0;
  for (let j = 0; j < 4; j++) {
    const d = (a[j] - mean[j]) / sd[j] - (b[j] - mean[j]) / sd[j];
    s += FW[j] * d * d;
  }
  return Math.sqrt(s / wSum);
}

/** The five quantiles for one forecast hour. */
function hourQuantiles(p1: number, med: number, spread: number, capRatio: number): number[] {
  return Z.map((z) => Math.max(0.01, round2(Math.min(p1 * Math.exp(med + z * spread), p1 * capRatio))));
}

/** House persistence shape (fallback when fewer than 3 analogs). */
function persistenceQuantiles(p1: number, h: number): number[] {
  return Z.map((z) => Math.max(0.01, round2(p1 * Math.exp(z * SIGMA * Math.sqrt(h - 1)))));
}

/** Forecast one case: 23 rows, hours 2..24, five quantiles each. */
function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  const ds = new Date(c.day_start).getTime();
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const ch1 = Math.max(1, first.channels ?? 1);
  const caseLaunch = buildLaunch(c.id, ds, c.history ?? [], p1, ch1, '');

  // Eligible comparables: tier "comparable" only, with an hour-1 print.
  const eligible: Launch[] = [];
  for (const comp of c.comparables ?? []) {
    const cds = new Date(comp.day_start).getTime();
    const h1pt = (comp.hourly ?? []).find((pt) => {
      const t = new Date(pt.from ?? '').getTime();
      return !Number.isNaN(t) && Math.round((t - cds) / 3600000) + 1 === 1 && pt.viewers != null && pt.viewers > 0;
    });
    const launch = buildLaunch(comp.name, cds, comp.hourly ?? [], h1pt ? h1pt.viewers : 1, h1pt && h1pt.channels != null ? h1pt.channels : 1, comp.unlock ?? '');
    if (launch.tier === 'comparable' && launch.curve.has(1)) eligible.push(launch);
  }

  const { mean, sd } = standardize(eligible.map((l) => l.feat));
  const caseZ = caseLaunch.feat.map((f, j) => (f - mean[j]) / sd[j]);

  // k nearest analogs by weighted-RMS distance (stable order on ties).
  const scored = eligible
    .map((l, i) => ({ launch: l, d: distance(caseLaunch.feat, l.feat, mean, sd), i }))
    .sort((a, b) => (a.d - b.d) || (a.i - b.i))
    .slice(0, K);
  const analogs = scored.map((s) => ({ launch: s.launch, w: 1 / (s.d + OFFSET) }));

  // Rolling family: eligible comparables with a midnight-local, non-simultaneous unlock.
  const family = eligible.filter((l) => l.rolling);
  // Non-explosive eligible launches (largest hour-2..24 ratio <= 15), for the family cap.
  const nonExplosive = eligible.filter((l) => l.maxRatio <= 15);

  const lp1Case = Math.log(p1);
  const wmedLnp1 = weightedMedian(analogs.map((a) => [Math.log(a.launch.p1), a.w] as [number, number]));

  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const { hour: h } of c.forecast) {
    if (analogs.length < 3) {
      rows.push({ hour: h, q: persistenceQuantiles(p1, h) });
      continue;
    }
    const withH = analogs.filter((a) => a.launch.curve.has(h));
    if (withH.length < 3) {
      rows.push({ hour: h, q: persistenceQuantiles(p1, h) });
      continue;
    }
    const lr = withH.map((a) => {
      const v = a.launch.curve.get(h)!;
      return [Math.log(v.viewers / a.launch.curve.get(1)!.viewers), a.w] as [number, number];
    });
    const med = weightedMedian(lr);

    // Level correction: rolling-family OLS slope of log-ratio on ln(p1).
    const famH = family.filter((l) => l.curve.has(h));
    let beta = 0;
    if (famH.length >= 2) {
      const xs = famH.map((l) => Math.log(l.p1));
      const ys = famH.map((l) => Math.log(l.curve.get(h)!.viewers / l.curve.get(1)!.viewers));
      beta = clamp(olsSlope(xs, ys), -0.6, 0);
    }
    const corrected = med + beta * clamp(lp1Case - wmedLnp1, -1, 1);

    // Spread: analog dispersion plus a share of the rolling family's dispersion.
    const wVar = weightedPopVar(lr);
    const famVar = popVar(famH.map((l) => Math.log(l.curve.get(h)!.viewers / l.curve.get(1)!.viewers)));
    const spread = clamp(1.3 * Math.sqrt(wVar + 0.3 * famVar), 0.25, 2.2);

    // Family cap: no quantile may exceed p1 * max(1.5 * exp(median), 1.25 * the
    // largest ratio that hour among non-explosive launches).
    let largest = 0;
    for (const l of nonExplosive) {
      const v = l.curve.get(h);
      if (v) largest = Math.max(largest, v.viewers / l.curve.get(1)!.viewers);
    }
    const capRatio = Math.max(1.5 * Math.exp(corrected), 1.25 * largest);

    rows.push({ hour: h, q: hourQuantiles(p1, corrected, spread, capRatio) });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
