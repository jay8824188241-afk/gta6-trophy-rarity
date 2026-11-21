"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target time.

Reads every ``<id>.in`` case in the input directory (``$MAKEFUN_INPUT_DIR``,
else ``/input``) and writes one ``<id>.out`` forecast per case to the output
directory (``$MAKEFUN_OUTPUT_DIR``, else ``/output``).

Model (per trophy, standard library only, fully deterministic, no network):
  * Work on the logit of the earn rate against ``x = ln(1 + hours since the
    2026-11-19T00:00:00Z release)``.  Log-time is the natural clock for a
    launch week: early progress is fast and then tapers, and the house curve
    baseline uses the same axis.
  * Fit 1 -- recency-weighted robust linear fit of logit(rate) on x over the
    whole history (weights decay with a 36 h half-life; two Huber passes so
    Exophase refresh jumps, which can drag every rate down at once when the
    tracked-player count jumps, do not bend the trend).  The slope is shrunk
    toward zero by its own t-statistic, so a noisy slope cannot extrapolate
    wildly over a multi-day horizon, and it is capped.
  * Fit 2 -- saturating curve: for a grid of ceilings A, fit logit(rate / A)
    linearly on x and keep the ceiling with the lowest weighted squared
    error.  This stops a fast mover from being extrapolated past any
    plausible ceiling (a story trophy cannot exceed the share of players who
    reach that point in the story).
  * The median is the logit-average of the two fits, blended toward the
    robust trend's fitted value at the last read (not the raw last read:
    one refresh jump would otherwise drag a short-horizon forecast with
    it, and the anchor is gated on the leave-last-out trend so a jumped
    last read is replaced by the trend without it while a genuine last
    read is kept); the persistence weight decays with the horizon (about
    0.10 at 4.5 days, about 0.97 at 90 minutes), so short-horizon
    forecasts stay close to the last read while week-ahead forecasts
    follow the trend.
  * The spread is honest about the horizon: a random-walk term
    (0.5 * sqrt(days), the house baseline's width) plus the slope's standard
    error times the extrapolation distance, plus the disagreement between
    the two fits, plus the pooled size of past refresh jumps -- all combined
    in quadrature, floored at the random-walk width and capped at 2 logit
    units.  The 5%-95% band therefore widens the further ahead we forecast.

Invalid input fails loudly (a bad case raises instead of silently writing a
plausible-looking forecast); a trophy with no usable read at all falls back
to a near-zero rate with the widest spread rather than failing the case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; nothing beside this file
# is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per square-root day (the house baseline's).
RW_SIGMA = 0.5
# Recency half-life for history weights, in hours.
HALF_LIFE_H = 36.0
# Spread floor / cap, in logit units.
SIGMA_CAP = 2.0
# Persistence weight decays as exp(-horizon_days / TAU).
PERSIST_TAU_DAYS = 2.0
# Candidate ceilings (percent) for the saturating fit.
CEILINGS = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 99.0]


def clamp_pct(x):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_linfit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, sse, xbar)."""
    sw = sum(ws)
    if sw <= 0:
        raise ValueError("no weights")
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        # All x identical: flat fit at the weighted mean.
        sse = sum(w * (y - ybar) ** 2 for y, w in zip(ys, ws))
        return ybar, 0.0, 0.0, sse, xbar
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    n = len(xs)
    dof = max(n - 2, 1)
    s2 = sum(w * r * r for r, w in zip(resid, ws)) / (sw * dof / max(n, 1)) if n > 2 else 0.0
    # Variance of the slope under the weighted fit.
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    sse = sum(w * r * r for r, w in zip(resid, ws))
    return a, b, se_b, sse, xbar


def robust_linfit(xs, ys, ws):
    """WLS with two Huber reweighting passes against large residuals."""
    a, b, se_b, sse, xbar = weighted_linfit(xs, ys, ws)
    for _ in range(2):
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        scale = sorted(abs(r) for r in resid)[len(resid) // 2] if resid else 0.0
        scale = max(scale * 1.4826, 0.05)
        cut = 2.0 * scale
        ws2 = [w * (1.0 if abs(r) <= cut else cut / abs(r))
               for r, w in zip(resid, ws)]
        a, b, se_b, sse, xbar = weighted_linfit(xs, ys, ws2)
    return a, b, se_b, sse, xbar


def forecast_trophy(obs, x_target, horizon_days, jump_std):
    """obs: list of (x, logit_rate, weight). Returns (median_logit, sigma)."""
    xs = [o[0] for o in obs]
    ys = [o[1] for o in obs]
    ws = [o[2] for o in obs]
    if not obs:
        return logit_pct(0.01), SIGMA_CAP

    # --- Fit 1: shrunk robust log-time trend on the logit scale. ---
    a, b, se_b, _sse1, xbar = robust_linfit(xs, ys, ws)
    if se_b > 0:
        t_stat = abs(b) / se_b
        shrink = (t_stat * t_stat) / (t_stat * t_stat + 4.0)
    else:
        shrink = 1.0 if b != 0.0 else 0.0
    b_adj = max(-4.0, min(4.0, b * shrink))
    a_adj = (sum(w * y for y, w in zip(ys, ws)) / sum(ws)) - b_adj * xbar
    trend_logit = a_adj + b_adj * x_target

    # --- Fit 2: saturating curve, ceiling picked by weighted SSE. ---
    rates = [100.0 / (1.0 + math.exp(-y)) for y in ys]
    max_rate = max(rates)
    best = None  # (sse, ceiling, a, b)
    if len(set(round(x, 9) for x in xs)) >= 2:
        for ceiling in CEILINGS:
            if ceiling <= max_rate * 1.02:
                continue
            zs = []
            ok = True
            for r in rates:
                frac = r / ceiling
                if frac <= 0.0 or frac >= 1.0:
                    ok = False
                    break
                zs.append(math.log(frac / (1.0 - frac)))
            if not ok:
                continue
            fa, fb, _se, sse, _xb = weighted_linfit(xs, zs, ws)
            if best is None or sse < best[0]:
                best = (sse, ceiling, fa, fb)
    if best is not None:
        _sse, ceiling, fa, fb = best
        fb = max(-4.0, min(4.0, fb))
        sat_frac = 1.0 / (1.0 + math.exp(-(fa + fb * x_target)))
        sat_rate = ceiling * sat_frac
        sat_logit = logit_pct(sat_rate)
    else:
        sat_logit = trend_logit

    # --- Median: average the two fits, blended toward the last read. ---
    ensemble = 0.5 * (trend_logit + sat_logit)
    # Anchor on the robust trend's fitted value at the last read, not the
    # raw last read: one Exophase refresh jump can drag every rate down at
    # once, and at short horizons the persistence weight is near 1, so the
    # raw read would carry the whole forecast with it.
    last_anchor = a_adj + b_adj * xs[-1]
    # Outlier gate: if the last read disagrees with the trend fitted
    # without it by more than the robust cut, it is a refresh jump, so
    # anchor on the leave-last-out trend instead; a genuine last read
    # (inside the cut) keeps the full-fit anchor.  The jump leaks into
    # the full fit's intercept and inflates the slope's standard error
    # (which shrinks the slope toward zero), so testing against the
    # leave-last-out fit is what actually separates a jump from a move.
    if len(obs) >= 3:
        xs_loo, ys_loo, ws_loo = xs[:-1], ys[:-1], ws[:-1]
        a_loo, b_loo, se_loo, _sse_loo, xbar_loo = robust_linfit(
            xs_loo, ys_loo, ws_loo)
        loo_at_last = a_loo + b_loo * xs[-1]
        resid_loo = [y - (a_loo + b_loo * x)
                     for x, y in zip(xs_loo, ys_loo)]
        scale_loo = sorted(abs(r) for r in resid_loo)[len(resid_loo) // 2]
        scale_loo = max(scale_loo * 1.4826, 0.05)
        if abs(ys[-1] - loo_at_last) > 2.0 * scale_loo:
            if se_loo > 0:
                t_loo = abs(b_loo) / se_loo
                shrink_loo = (t_loo * t_loo) / (t_loo * t_loo + 4.0)
            else:
                shrink_loo = 1.0 if b_loo != 0.0 else 0.0
            b_loo_adj = max(-4.0, min(4.0, b_loo * shrink_loo))
            a_loo_adj = (sum(w * y for y, w in zip(ys_loo, ws_loo))
                         / sum(ws_loo)) - b_loo_adj * xbar_loo
            last_anchor = a_loo_adj + b_loo_adj * xs[-1]
    w_persist = math.exp(-horizon_days / PERSIST_TAU_DAYS)
    median_logit = (1.0 - w_persist) * ensemble + w_persist * last_anchor

    # --- Spread: random walk + slope uncertainty + model disagreement
    #     + past refresh-jump size, combined in quadrature. ---
    sigma_rw = RW_SIGMA * math.sqrt(horizon_days)
    sigma_slope = min(1.5, se_b * max(0.0, x_target - xbar))
    sigma_model = min(0.5, 0.5 * abs(trend_logit - sat_logit))
    sigma_jump = min(1.0, jump_std * math.sqrt(max(horizon_days, 0.0)))
    sigma = math.sqrt(sigma_rw ** 2 + sigma_slope ** 2
                      + sigma_model ** 2 + sigma_jump ** 2)
    sigma = max(sigma_rw, min(SIGMA_CAP, sigma))
    return median_logit, sigma


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case.get("trophies")
    history = case.get("history")
    if not trophies:
        raise ValueError(f"{name} has no trophies")
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))
    last_at = parse_time(history[-1]["at"])

    # Pooled size of one-step logit jumps (refresh moves), across trophies.
    jumps = []
    prev_rates = None
    for read in history:
        rates = read.get("rates") or []
        if prev_rates is not None:
            for r0, r1 in zip(prev_rates, rates):
                if r0 is not None and r1 is not None:
                    jumps.append(logit_pct(r1) - logit_pct(r0))
        prev_rates = rates
    if len(jumps) >= 2:
        mean_jump = sum(jumps) / len(jumps)
        jump_std = math.sqrt(sum((j - mean_jump) ** 2 for j in jumps)
                             / (len(jumps) - 1))
    else:
        jump_std = 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        obs = []
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            at = parse_time(read["at"])
            hours = max(0.0, (at - RELEASE).total_seconds() / 3600.0)
            x = math.log1p(hours)
            age_h = max(0.0, (last_at - at).total_seconds() / 3600.0)
            w = 0.5 ** (age_h / HALF_LIFE_H)
            obs.append((x, logit_pct(rates[i]), w))
        median_logit, sigma = forecast_trophy(obs, x_target, horizon_days,
                                              jump_std)
        qs = []
        for z in Z:
            q = sigmoid_logit(median_logit + z * sigma)
            q = min(99.9999, max(0.0001, q))
            qs.append(round4(q))
        # Rounding can only tie neighbours; enforce non-decreasing order.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_files = sorted(INPUT_DIR.glob("*.in"))
    if not in_files:
        raise FileNotFoundError(f"no case files in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in in_files:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("schema") != "makefun/gta6-trophies-case/v1":
            raise ValueError(f"{path.name} has schema {case.get('schema')!r}")
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
