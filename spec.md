# Ichimoku Intraday Bot — SPEC (MVP)

## Conventions temps
- Timezone: UTC
- Une bougie est "close" à la fin de son intervalle.
- Le bot ne doit utiliser que des bougies closes (jamais la bougie en cours).

## Timeframes
- Signal TF: 15m
- Trend TF: 1h
- Regime global: BTC 4h

## Ichimoku (9/26/52)
On calcule sur OHLC du timeframe considéré:

- Tenkan = (HH(9) + LL(9)) / 2
- Kijun  = (HH(26) + LL(26)) / 2
- SpanA  = (Tenkan + Kijun) / 2, projeté +26
- SpanB  = (HH(52) + LL(52)) / 2, projeté +26
- Chikou = Close projeté -26

Kumo top/bottom:
- kumo_top = max(SpanA, SpanB)
- kumo_bottom = min(SpanA, SpanB)

## Gates

### BTC Regime (4h) — pour alts uniquement
Regime ON si:
- close_4h > kumo_top_4h
- kijun_4h[t] - kijun_4h[t-10] > 0

Si regime OFF => pas de signaux LONG sur alts.

### Trend (1h) — pour chaque symbole
Trend ON si:
A) close_1h > kumo_top_1h
OU
B) (tenkan_1h > kijun_1h) ET (kijun_1h[t] - kijun_1h[t-10] > 0)

## Setup A+ LONG (Trend Continuation) — 15m

Préconditions à la bougie close t:
1) close_15m > kumo_top_15m
2) tenkan_15m > kijun_15m
3) spanA_15m > spanB_15m
4) chikou_15m > kumo_top_15m  (strict)
5) kijun_slope_15m: kijun[t] - kijun[t-10] > 0

Retest Kijun (fenêtre N=12 bougies, eps=0.0015):
- il existe i dans [t-12, t-1] tel que low[i] <= kijun[i] * (1 + 0.0015)

Trigger (bougie t):
- close[t] > tenkan[t]

ENTRY MODEL:
- entry = open[t+1]

STOP INITIAL:
- ATR14 (Wilder) sur 15m
- stop = min(kijun[t], kumo_bottom[t]) - 0.5 * ATR14[t]

EXITS (ordre):
1) Stop intrabar: si low <= stop => exit STOP (au stop)
2) Kijun break: si close < kijun => exit next open
3) TK cross confirmé: si (tenkan < kijun) ET (close < kumo_top) => exit next open

## Scoring (0–100)
Gates: Trend 1h ON, et BTC Regime 4h ON (pour alts). Sinon pas d’alerte.

Points:
+20 close>kumo_top (15m)
+10 tenkan>kijun (15m)
+10 spanA>spanB (15m)
+10 chikou>kumo_top (15m)
+10 kijun_slope>0 (15m)
+10 retest kijun (3h)
+10 distance(close,kijun)/close <= 0.008
+10 kumo_thickness_pct >= 0.0035

A+ threshold: score >= 80

## Anti-spam / Dedup
- Cooldown après ENTRY: 12 bougies 15m (3h)
- Pas de nouvel ENTRY tant qu’il n’y a pas eu invalidation (close repasse sous kijun ou sous kumo).

## Coûts (pour backtest)
- fee_rate, slippage_rate appliqués à l’entrée et la sortie.
