// gta6-twitch-ccu-forecast-01 — adaptive k-NN nearest-analog forecast (variant A).
//
// For each launch, the comparable launches (tier "comparable" only) are scored by a
// standardized weighted-RMS distance over four log features:
//   [ln(hour-1 print), ln(48h pre-launch peak), ln(7d median baseline),
//    ln(channels at hour 1) - ln(hour-1 print)]
// The six nearest analogs give, per hour h = 2..24, a weighted median of their
// log-ratio curves ln(v_h / v_1), plus a level correction: the median is shifted by
// beta_h * clamp(ln p1_case - weighted-median analog ln p1, -1, +1), where beta_h is
// the OLS slope of log-ratio on ln p1 across the rolling-launch family (midnight
// local launches, excluding simultaneous/single-global ones), clamped to [-0.6, 0].
// The spread is adaptive: clamp(1.3 * sqrt(analog variance + 0.3 * family variance),
// 0.25, 2.2). Every quantile is capped at p1 * max(1.5 * exp(median),
// 1.25 * largest ratio that hour among non-explosive launches, i.e. launches whose
// largest hour-2..24 ratio is at most 15).
// Deterministic, no randomness, no clock, Node 22, standard library only.
// I/O: read every <id>.in in the input directory, write <id>.out to the output
// directory, 23 rows (hours 2..24) of five quantiles each.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Feature weights for the standardized distance. */
const FW = [1.0, 0.7, 0.7, 0.4];
const FWSUM = FW.reduce((a, b) => a + b, 0);
/** Number of nearest analogs. */
const K = 6;
/** Analog weight is 1 / (distance + OFFSET). */
const OFFSET = 0.29;
/** Spread multiplier on sqrt(analog variance + VARF_WEIGHT * family variance). */
const CAL = 1.3;
const VARF_WEIGHT = 0.3;
/** Level-correction slope is clamped to [BETA_LO, 0]; the log-p1 gap to [-1, 1]. */
const BETA_LO = -0.6;
const DELTA_CLAMP = 1;
/** Spread is clamped to [SPREAD_LO, SPREAD_HI]. */
const SPREAD_LO = 0.25;
const SPREAD_HI = 2.2;
/** Family cap: p1 * max(CAP_FLOOR * exp(median), CAP_MULT * max ratio that hour). */
const CAP_FLOOR = 1.5;
const CAP_MULT = 1.25;
/** A launch is non-explosive when its largest hour-2..24 ratio is at most this. */
const CAP_MAX_RATIO = 15;
/** Fallback persistence spread, in log units per square-root hour. */
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
  unlock?: string;
  hourly: Point[];
}

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels?: number | null }>;
  forecast: Array<{ hour: number }>;
  day_start: string;
  history: Point[];
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;
const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const varPop = (xs: number[]): number => {
  const m = mean(xs);
  return mean(xs.map((x) => (x - m) ** 2));
};
const wmean = (xs: number[], ws: number[]): number =>
  xs.reduce((a, x, i) => a + x * ws[i], 0) / ws.reduce((a, b) => a + b, 0);
/** Step weighted median: smallest value whose cumulative weight reaches half. */
const wmedianStep = (xs: number[], ws: number[]): number => {
  const pairs = xs.map((x, i) => [x, ws[i]] as const).sort((a, b) => a[0] - b[0]);
  const tot = ws.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const [x, w] of pairs) {
    acc += w;
    if (acc >= tot / 2) return x;
  }
  return pairs[pairs.length - 1][0];
};
/** The tier of a launch, parsed from the "Tier:" field of its unlock text. */
const tierOf = (unlock?: string): string => {
  const m = /Tier:\s*([a-z-]+)/i.exec(unlock ?? '');
  return m ? m[1].toLowerCase() : 'unknown';
};
/** Rolling family: midnight-local launches, excluding simultaneous/single-global. */
const isRolling = (unlock?: string): boolean =>
  /midnight local/i.test(unlock ?? '') && !/simultaneous|single global/i.test(unlock ?? '');

/** Hour-1..24 curve of one launch, keyed by hour, from hourly points. */
const dayCurve = (hourly: Point[], dayStartMs: number): Map<number, Point> => {
  const byHour = new Map<number, Point>();
  for (const pt of hourly) {
    const t = new Date(pt.from ?? pt.at ?? '').getTime();
    const h = Math.round((t - dayStartMs) / 3600000) + 1;
    if (h >= 1 && h <= 24 && pt.viewers != null && pt.viewers > 0) byHour.set(h, pt);
  }
  return byHour;
};

interface PreFeatures {
  peak: number;
  base: number;
}
/** 48h pre-launch peak and 7d median baseline, from points before day start. */
const preFeatures = (points: Point[], dayStartMs: number, fallback: number): PreFeatures => {
  const pre = points.filter((p) => new Date(p.from ?? p.at ?? '').getTime() < dayStartMs);
  const w48 = pre.filter((p) => new Date(p.from ?? p.at ?? '').getTime() >= dayStartMs - 48 * 3600000);
  const w7d = pre.filter((p) => new Date(p.from ?? p.at ?? '').getTime() >= dayStartMs - 7 * 86400000);
  const peak = w48.length
    ? Math.max(...w48.map((p) => p.viewers))
    : pre.length
      ? Math.max(...pre.map((p) => p.viewers))
      : fallback;
  const base = w7d.length
    ? median(w7d.map((p) => p.viewers))
    : pre.length
      ? median(pre.map((p) => p.viewers))
      : fallback;
  return { peak, base };
};

interface Comp {
  name: string;
  f: number[];
  lnp1: number;
  ratios: Record<number, number>;
  maxRatio: number;
  rolling: boolean;
}

/** The forecast of one case: 23 rows of five quantiles. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const ds = new Date(c.day_start).getTime();
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const ch1 = Math.max(1, first.channels ?? 1);
  const cf = preFeatures(c.history, ds, p1);
  const caseF = [
    Math.log(p1),
    Math.log(Math.max(1, cf.peak)),
    Math.log(Math.max(1, cf.base)),
    Math.log(ch1) - Math.log(p1),
  ];

  const comps: Comp[] = [];
  for (const comp of c.comparables) {
    if (tierOf(comp.unlock) !== 'comparable') continue;
    const cds = new Date(comp.day_start).getTime();
    const byHour = dayCurve(comp.hourly, cds);
    if (!byHour.has(1) || byHour.size < 20) continue;
    const cp1 = byHour.get(1)!.viewers;
    const cch1 = Math.max(1, byHour.get(1)!.channels ?? 1);
    const pf = preFeatures(comp.hourly, cds, cp1);
    const f = [
      Math.log(cp1),
      Math.log(Math.max(1, pf.peak)),
      Math.log(Math.max(1, pf.base)),
      Math.log(cch1) - Math.log(cp1),
    ];
    const ratios: Record<number, number> = {};
    for (let h = 2; h <= 24; h++) {
      if (byHour.has(h)) ratios[h] = Math.log(byHour.get(h)!.viewers / cp1);
    }
    const maxRatio = Math.max(...Object.values(ratios).map((r) => Math.exp(r)));
    comps.push({ name: comp.name, f, lnp1: Math.log(cp1), ratios, maxRatio, rolling: isRolling(comp.unlock) });
  }

  const hours = c.forecast.map(({ hour }) => hour);
  if (comps.length < 3) {
    // Fallback: house persistence shape (never triggered on the LOO/practice sets).
    return hours.map((h) => ({
      hour: h,
      q: Z.map((z) => round2(p1 * Math.exp(z * SIGMA * Math.sqrt(h - 1)))),
    }));
  }

  // Standardize each feature by the eligible comparables' mean and population SD.
  const mu = [0, 1, 2, 3].map((j) => mean(comps.map((x) => x.f[j])));
  const sd = [0, 1, 2, 3].map((j) => Math.sqrt(varPop(comps.map((x) => x.f[j]))) || 1);
  const zc = caseF.map((v, j) => (v - mu[j]) / sd[j]);
  const scored = comps
    .map((x) => {
      const d = Math.sqrt(x.f.reduce((a, v, j) => a + FW[j] * ((v - mu[j]) / sd[j] - zc[j]) ** 2, 0) / FWSUM);
      return { x, d, w: 1 / (d + OFFSET) };
    })
    .sort((a, b) => a.d - b.d);
  const analogs = scored.slice(0, K);
  const fam = comps.filter((x) => x.rolling);
  const medLnP1 = wmedianStep(
    analogs.map((a) => a.x.lnp1),
    analogs.map((a) => a.w),
  );

  return hours.map((h) => {
    const have = analogs.filter((a) => a.x.ratios[h] != null);
    if (have.length === 0) {
      return { hour: h, q: Z.map((z) => round2(p1 * Math.exp(z * SIGMA * Math.sqrt(h - 1)))) };
    }
    const xs = have.map((a) => a.x.ratios[h]);
    const ws = have.map((a) => a.w);
    const med = wmedianStep(xs, ws);
    // Rolling-family OLS slope of log-ratio on ln p1 at this hour.
    const fb = fam.filter((x) => x.ratios[h] != null);
    let beta = 0;
    if (fb.length >= 2) {
      const xsF = fb.map((x) => x.lnp1);
      const ysF = fb.map((x) => x.ratios[h]);
      const mx = mean(xsF);
      const my = mean(ysF);
      const den = xsF.reduce((a, x) => a + (x - mx) ** 2, 0);
      if (den > 0) beta = xsF.reduce((a, x, i) => a + (x - mx) * (ysF[i] - my), 0) / den;
    }
    beta = clamp(beta, BETA_LO, 0);
    const delta = clamp(Math.log(p1) - medLnP1, -DELTA_CLAMP, DELTA_CLAMP);
    const medC = med + beta * delta;
    // Adaptive spread: weighted analog variance + family variance.
    const mA = wmean(xs, ws);
    const varA = xs.reduce((a, x, i) => a + ws[i] * (x - mA) ** 2, 0) / ws.reduce((a, b) => a + b, 0);
    const famVals = fb.map((x) => x.ratios[h]);
    const varF = famVals.length >= 2 ? varPop(famVals) : 0;
    const spread = clamp(CAL * Math.sqrt(varA + VARF_WEIGHT * varF), SPREAD_LO, SPREAD_HI);
    // Family cap: 1.5x the forecast median or 1.25x the largest ratio that hour
    // among non-explosive launches, whichever is larger.
    const maxR = Math.max(
      ...comps.filter((x) => x.maxRatio <= CAP_MAX_RATIO && x.ratios[h] != null).map((x) => Math.exp(x.ratios[h])),
    );
    const cap = p1 * Math.max(CAP_FLOOR * Math.exp(medC), CAP_MULT * maxR);
    const q = Z.map((z) => {
      const v = round2(Math.min(p1 * Math.exp(medC + z * spread), cap));
      return clamp(v, 1, 1e9);
    });
    return { hour: h, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j]))
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
