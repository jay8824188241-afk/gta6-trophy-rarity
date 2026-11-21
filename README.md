# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jay8824188241-afk/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jay8824188241-afk/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) on the logit scale. Each trophy gets an ensemble of two fits on log-time since release — a recency-weighted robust logit trend with t-shrunk slope, and a saturating curve whose ceiling is picked by fit — blended toward the last observed rate, with a spread that grows with the horizon (random walk plus slope uncertainty, model disagreement and past refresh-jump size).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=d1n6bjq8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
