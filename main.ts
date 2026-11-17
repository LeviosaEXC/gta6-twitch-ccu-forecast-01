// gta6-twitch-ccu-forecast-01 — nearest-analog forecast with a family-capped
// upper tail.
//
// Idea: no two launches share one shape, but launches that LOOK alike before
// release day behave alike on it. This entry is instance-based, not pooled:
// for the target launch it finds the k = 6 most similar past launches and
// forecasts the weighted median of THEIR ratio curves (viewers in each hour
// divided by the hour-1 print), with the analogs' dispersion as its spread.
// Similarity is measured on four features every case file carries for both
// the target and each comparable:
//
//   f1 = ln(hour-1 print)                        (how much has already unlocked)
//   f2 = ln(peak viewers in the 48 h before day_start)   (pre-launch hype peak)
//   f3 = ln(median viewers in the 7 d before day_start)  (baseline audience)
//   f4 = ln(channels at hour 1) - ln(hour-1 print)       (streamer share)
//
// Distances are standardized Euclidean with weights (1.0, 0.7, 0.7, 0.4);
// analog weights are 1 / (d + 0.5). Early-access-contaminated and weak-tier
// launches are excluded from the pool. The spread blends the analogs'
// dispersion with the pooled dispersion (70/30) so six agreeing analogs can
// never claim false certainty. Where the target has no pre-day history the
// hype features fall back to ln(p1); where a launch lacks channels, f4 is
// dropped from that pair's distance.
//
// The upper tail is truncated to the final's own launch family: every
// quantile is capped at 1.25x the largest ratio that hour among the
// non-explosive launches (launches whose late-day ratio never exceeds 15x —
// every rolling-midnight launch qualifies; the simultaneous-global ones, whose
// curves explode 30-2000x from a tiny hour-1, do not). The 25% headroom lets
// GTA VI exceed any single training launch. The cap is computed from the case
// file's own comparables, never from a hardcoded number, so the program stays
// data-driven and deterministic.
//
// Why this shape, and not a regression on level: the hour-1 print is a
// confounded signal. The two largest training launches (Elden Ring, Cyberpunk)
// have huge hour-1 prints AND flat curves — but their hour-1 came from a
// simultaneous PC/global unlock the night before. GTA VI's hour-1 will come
// only from early timezones, with the US unlocking at 05:00/08:00 UTC, so its
// audience should RISE into US prime time like every rolling-midnight launch
// (family median 2.47x at hour 19), not collapse. A level regression learns
// the confound and forecasts a collapse at a 1,000,000-viewer hour-1; the
// analogs, picked by level AND hype AND baseline, do not.
//
//   q_tau(h) = min(p1 * exp(wmed_h + z_tau * sd_h), p1 * 1.25 * maxFamilyRatio_h),
//   with the cap never set below 1.5x the forecast median.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Smallest log-spread allowed for any hour. */
const SD_FLOOR = 0.4;
/** A comparable needs this many release-day hourly points to count. */
const MIN_DAY_HOURS = 20;
/** Number of nearest analogs whose curves are blended. */
const K = 6;
/** Feature weights for the standardized distance: (level, hype peak, baseline, channel share). */
const FEATURE_W = [1.0, 0.7, 0.7, 0.4];
/** A launch counts as non-explosive when its late-day ratio never exceeds this. */
const EXPLOSIVE_RATIO = 15;
/** Headroom over the family's largest ratio for the cap. */
const CAP_HEADROOM = 1.25;
/** The cap never bites closer than this multiple of the forecast median. */
const CAP_MEDIAN_FLOOR = 1.5;

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels?: number }>;
  forecast: Array<{ hour: number }>;
  history: Array<{ at: string; viewers: number | null; channels: number | null }>;
  history_hourly: Array<{ from: string; viewers: number | null; channels: number | null; reads: number }>;
  comparables: Comparable[];
}

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

const sd = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
};

/** x rounded to 2 decimals (half up), matching the starter's output style. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The "Tier:" marker in a comparable's unlock text, when it carries one. */
const tierOf = (unlock: string | undefined): string => {
  const m = /Tier:\s*([A-Za-z-]+)/.exec(unlock ?? '');
  return m ? m[1] : 'comparable';
};

/**
 * A launch's release-day curve: hour (1-24, UTC day of its day_start)
 * to viewers. Points outside the day or without viewers are ignored.
 */
const dayCurve = (dayStart: string, points: HourlyPoint[]): Map<number, number> => {
  const ds = Date.parse(dayStart);
  const byHour = new Map<number, number>();
  for (const pt of points ?? []) {
    if (pt.viewers == null || pt.viewers <= 0) continue;
    const h = Math.round((Date.parse(pt.from) - ds) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, pt.viewers);
  }
  return byHour;
};

/**
 * Pre-launch features from a launch's hourly points before its day_start:
 * [ln(48h peak), ln(7d median)]. Falls back to ln(p1) when there is no
 * usable pre-day history.
 */
const preFeatures = (dayStart: string, points: HourlyPoint[], logP1: number): [number, number] => {
  const ds = Date.parse(dayStart);
  const pre = (points ?? []).filter((p) => p.viewers != null && p.viewers > 0 && Date.parse(p.from) < ds);
  if (pre.length === 0) return [logP1, logP1];
  const peak48 = pre.filter((p) => Date.parse(p.from) >= ds - 48 * 3600000);
  const peak = peak48.length > 0 ? Math.max(...peak48.map((p) => p.viewers as number)) : Math.max(...pre.map((p) => p.viewers as number));
  const last7 = pre.filter((p) => Date.parse(p.from) >= ds - 7 * 24 * 3600000);
  const base = last7.length > 0 ? median(last7.map((p) => p.viewers as number)) : median(pre.map((p) => p.viewers as number));
  return [Math.log(Math.max(1, peak)), Math.log(Math.max(1, base))];
};

/** Weighted median: the value where cumulative weight first reaches half. */
const weightedMedian = (vals: number[], wts: number[]): number => {
  const pairs = vals.map((v, i) => [v, wts[i]] as [number, number]).sort((a, b) => a[0] - b[0]);
  const total = wts.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const [v, w] of pairs) {
    acc += w;
    if (acc >= total / 2) return v;
  }
  return pairs[pairs.length - 1][0];
};

/** Weighted standard deviation around the weighted mean. */
const weightedSd = (vals: number[], wts: number[]): number => {
  const total = wts.reduce((a, b) => a + b, 0);
  const m = vals.reduce((s, v, i) => s + v * wts[i], 0) / total;
  return Math.sqrt(vals.reduce((s, v, i) => s + wts[i] * (v - m) * (v - m), 0) / total);
};

interface Launch {
  /** [ln p1, ln 48h peak, ln 7d median, channel share (or null)]. */
  feats: [number, number, number, number | null];
  /** Log-ratios ln(v_h / v_1) for hours 2-24, null where the hour is missing. */
  ratios: Array<number | null>;
  /** Log hour-1 print (kept for the fallback path). */
  logP1: number;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const logP1 = Math.log(p1);

  // Target's own pre-launch features, from its hourly history (minute-level
  // history is the same points at finer spacing; hourly is what we need).
  const targetHourly: HourlyPoint[] = (c.history_hourly ?? []).map((p) => ({
    from: p.from,
    viewers: p.viewers,
    channels: p.channels,
  }));
  const dayStart = (c as unknown as { day_start: string }).day_start;
  const [tPeak, tBase] = preFeatures(dayStart, targetHourly, logP1);
  const tShare = first.channels != null && first.channels > 0
    ? Math.log(first.channels) - logP1
    : null;
  const targetFeats: [number, number, number, number | null] = [logP1, tPeak, tBase, tShare];

  // Analog pool: usable release-day curves, minus contaminated / weak tiers.
  const pool: Launch[] = [];
  for (const comp of c.comparables ?? []) {
    const tier = tierOf(comp.unlock);
    if (tier === 'early-access-contaminated' || tier === 'weak') continue;
    const byHour = dayCurve(comp.day_start, comp.hourly);
    if (!byHour.has(1) || byHour.size < MIN_DAY_HOURS) continue;
    const p = byHour.get(1) as number;
    const lp = Math.log(p);
    const [pk, bs] = preFeatures(comp.day_start, comp.hourly, lp);
    const h1 = comp.hourly.find((pt) => {
      const h = Math.round((Date.parse(pt.from) - Date.parse(comp.day_start)) / 3600000) + 1;
      return h === 1;
    });
    const share = h1?.channels != null && (h1.channels as number) > 0
      ? Math.log(h1.channels as number) - lp
      : null;
    const ratios: Array<number | null> = [];
    for (let h = 2; h <= 24; h++) {
      const v = byHour.get(h);
      ratios.push(v === undefined ? null : Math.log(v / p));
    }
    pool.push({ feats: [lp, pk, bs, share], ratios, logP1: lp });
  }

  // Family cap: per hour, 1.25x the largest ratio among the non-explosive
  // launches — launches whose late-day (hours 13-24) ratio never exceeds 15x.
  // Every rolling-midnight launch qualifies; the simultaneous-global ones,
  // whose curves explode from a tiny hour-1, do not.
  const familyCap: Array<number | null> = [];
  for (let h = 2; h <= 24; h++) {
    let m: number | null = null;
    for (const l of pool) {
      const late = l.ratios.slice(11).filter((v): v is number => v !== null);
      if (late.length > 0 && Math.max(...late.map(Math.exp)) >= EXPLOSIVE_RATIO) continue;
      const v = l.ratios[h - 2];
      if (v !== null) m = m === null ? Math.exp(v) : Math.max(m, Math.exp(v));
    }
    familyCap.push(m === null ? null : m * CAP_HEADROOM);
  }

  // Standardize features on the pool (guard against a zero spread).
  const scales = [0, 1, 2].map((j) => Math.max(0.25, sd(pool.map((l) => l.feats[j] as number))));
  const scaleShare = Math.max(0.25, sd(pool.map((l) => l.feats[3]).filter((v): v is number => v !== null)));
  const dist = (a: Launch['feats'], b: Launch['feats']): number => {
    let s = 0;
    let wsum = 0;
    for (let j = 0; j < 3; j++) {
      const sc = scales[j];
      s += FEATURE_W[j] * (((a[j] as number) - (b[j] as number)) / sc) ** 2;
      wsum += FEATURE_W[j];
    }
    if (a[3] !== null && b[3] !== null) {
      s += FEATURE_W[3] * (((a[3] as number) - (b[3] as number)) / scaleShare) ** 2;
      wsum += FEATURE_W[3];
    }
    return Math.sqrt(s / wsum);
  };

  const ranked = pool
    .map((l) => ({ l, d: dist(targetFeats, l.feats) }))
    .sort((a, b) => a.d - b.d || a.l.logP1 - b.l.logP1);
  const analogs = ranked.slice(0, Math.min(K, ranked.length));
  const analogW = analogs.map((a) => 1 / (a.d + 0.5));

  const hours = c.forecast.map(({ hour }) => {
    let qs: number[];
    if (analogs.length < 3) {
      // Too little training data: fall back to the house persistence shape.
      qs = Z.map((z) => round2(p1 * Math.exp(z * 0.15 * Math.sqrt(hour - 1))));
    } else {
      const vals: number[] = [];
      const wts: number[] = [];
      for (let i = 0; i < analogs.length; i++) {
        const v = analogs[i].l.ratios[hour - 2];
        if (v !== null) { vals.push(v); wts.push(analogW[i]); }
      }
      if (vals.length < 3) {
        qs = Z.map((z) => round2(p1 * Math.exp(z * 0.15 * Math.sqrt(hour - 1))));
      } else {
        const med = weightedMedian(vals, wts);
        const aSd = weightedSd(vals, wts);
        // Pooled dispersion for the same hour, as a floor against false certainty.
        const pooled = pool.map((l) => l.ratios[hour - 2]).filter((v): v is number => v !== null);
        const pSd = pooled.length > 1 ? sd(pooled) : aSd;
        const spread = Math.max(SD_FLOOR, Math.sqrt(0.7 * aSd * aSd + 0.3 * pSd * pSd));
        qs = Z.map((z) => round2(p1 * Math.exp(med + z * spread)));
      }
    }
    // Family cap: limit every quantile to the family's largest ratio for this
    // hour (with headroom), never below 1.5x the forecast median.
    const cap = familyCap[hour - 2];
    if (cap !== null) {
      const floor = qs[2] * CAP_MEDIAN_FLOOR;
      const limit = Math.max(cap * p1, floor);
      qs = qs.map((q) => Math.min(q, limit));
    }
    // Enforce non-decreasing quantiles and the contest's bounds.
    for (let j = 1; j < qs.length; j++) if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
    qs = qs.map((q) => Math.min(1e9, Math.max(0.01, q)));
    return { hour, q: qs };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
