# Broken Pipe Test -- Resultado de Corrida (Datos Reales)

> Regla de Oro: ruido blanco puro en el 50% de los activos (4-5 de los 9 tickers reales), motor real (HERC walk-forward de `pipeline_optimizacion.py`) + CPCV + DSR. **Corrida sobre MERCADO REAL** (`05_Datos_y_Resultados/precios_ajustados.csv`, SPY/QQQ/EEM/VNQ/TLT/IEF/LQD/GLD/DBC, 2007-2026) -- a diferencia del repo hermano herc-portfolio-optimizer, este proyecto SI tiene datos de precios reales comiteados.

**Configuración:** `{"frac_noise": 0.5, "n_trials": 40, "dias_ventana": 756, "dias_rebalanceo": 21, "noise_mean_mode": "zero", "n_activos_universo": 9, "n_obs_universo": 4925}`

## Hallazgo metodológico: `noise_mean_mode='zero'` vs `'matched'`

Mismo hallazgo que en el repo hermano: calibrar el ruido a la misma media del activo real que reemplaza (`matched`, lectura literal del pedido) le da al ruido la prima de riesgo real del universo (SPY/QQQ 2007-2026 tuvieron drift positivo) — no es fuga, es la prima que el propio generador de ruido regaló. `zero` (mu=0 exacto) es el default correcto.

| Modo | Contribución media diaria sleeve ruido | p-valor | Nivel de evidencia |
|---|---|---|---|
| `zero` (default) | 0.000008 | 0.17648 | sin_evidencia |
| `matched` (lectura literal) | 0.000100 | 0.00000 | fuga_clara |

## Veredicto (modo `zero`, el correcto)

**SIN EVIDENCIA**

SIN EVIDENCIA DE FUGA: contribucion media diaria del sleeve de ruido (0.000008) no es significativamente distinta de cero (p=0.17648) a traves de 40 trials con semillas independientes.

## Métricas agregadas (modo `zero`)

| Métrica | Valor |
|---|---|
| Sharpe medio del portafolio (entre trials) | 0.4434 |
| Sharpe p-value (vs 0) | 0.00000 |
| Contribución media diaria sleeve ruido | 0.000008 |
| Contribución p-value | 0.17648 |
| Sharpe anualizado (ultimo trial) | 0.2549467037681314 |
| Deflated Sharpe Ratio | 0.22134795518947636 |
| N modelos (trials) | 40 |

## CPCV sobre universo mixto (modo `zero`)

| Modelo | Sharpe medio OOS | Sharpe std | N combinaciones |
|---|---|---|---|
| Equiponderado | 0.2416 | 0.1379 | 15 |
| Referencia_60_40 | 0.1894 | 0.3655 | 15 |
| Markowitz | 0.1087 | 0.2688 | 15 |
| Min_CVaR | 0.0390 | 0.1806 | 15 |
| Min_CDaR | 0.0454 | 0.1846 | 15 |
| HERC | 0.2418 | 0.1470 | 15 |

## Detalle por trial (modo `zero`)

| Trial | Seed | N ruido | N real | Sharpe portafolio | Peso medio sleeve ruido | Contribución media diaria |
|---|---|---|---|---|---|---|
| 0 | 1608637542 | 4 | 5 | 0.2043 | 0.6589 | -0.000021 |
| 1 | 1273642419 | 4 | 5 | 0.2225 | 0.6462 | -0.000009 |
| 2 | 1935803228 | 4 | 5 | 0.3414 | 0.5282 | -0.000007 |
| 3 | 787846414 | 4 | 5 | 0.3491 | 0.4120 | -0.000009 |
| 4 | 996406378 | 4 | 5 | 0.4061 | 0.3158 | -0.000004 |
| 5 | 1201263687 | 4 | 5 | 0.4157 | 0.3267 | -0.000020 |
| 6 | 423734972 | 4 | 5 | 0.2110 | 0.3365 | -0.000037 |
| 7 | 415968276 | 4 | 5 | 0.6087 | 0.4966 | 0.000023 |
| 8 | 670094950 | 4 | 5 | 0.4152 | 0.4578 | 0.000039 |
| 9 | 1914837113 | 4 | 5 | 0.9292 | 0.2862 | 0.000075 |
| 10 | 669991378 | 4 | 5 | 0.3864 | 0.6573 | 0.000020 |
| 11 | 429389014 | 4 | 5 | 1.0410 | 0.5740 | 0.000121 |
| 12 | 249467210 | 4 | 5 | 0.3198 | 0.4912 | 0.000007 |
| 13 | 1972458954 | 4 | 5 | 0.7186 | 0.3828 | 0.000029 |
| 14 | 1572714583 | 4 | 5 | 0.5370 | 0.2931 | 0.000034 |
| 15 | 1433267572 | 4 | 5 | 0.5587 | 0.4451 | 0.000007 |
| 16 | 434285667 | 4 | 5 | 0.4703 | 0.4739 | 0.000020 |
| 17 | 613608295 | 4 | 5 | 0.5485 | 0.6237 | 0.000055 |
| 18 | 893664919 | 4 | 5 | 0.3690 | 0.5725 | 0.000015 |
| 19 | 648061058 | 4 | 5 | 0.9257 | 0.3890 | 0.000091 |
| 20 | 88409749 | 4 | 5 | 0.5455 | 0.3273 | 0.000011 |
| 21 | 242285876 | 4 | 5 | 0.3680 | 0.3185 | -0.000007 |
| 22 | 2018247425 | 4 | 5 | 0.4280 | 0.4363 | 0.000008 |
| 23 | 953477463 | 4 | 5 | 0.4392 | 0.3881 | -0.000012 |
| 24 | 1427830251 | 4 | 5 | 0.2189 | 0.5850 | -0.000032 |
| 25 | 1883569565 | 4 | 5 | 0.5471 | 0.3352 | 0.000028 |
| 26 | 911989541 | 4 | 5 | 0.3616 | 0.4749 | -0.000007 |
| 27 | 3344769 | 4 | 5 | 0.1227 | 0.3289 | -0.000048 |
| 28 | 780932287 | 4 | 5 | 0.2507 | 0.4239 | -0.000010 |
| 29 | 2114032571 | 4 | 5 | 0.2441 | 0.3311 | -0.000005 |
| 30 | 787716372 | 4 | 5 | 0.5495 | 0.4624 | 0.000033 |
| 31 | 504579232 | 4 | 5 | 0.4444 | 0.3286 | -0.000016 |
| 32 | 1306710475 | 4 | 5 | 0.5367 | 0.4029 | -0.000023 |
| 33 | 479546681 | 4 | 5 | 0.3847 | 0.3897 | -0.000007 |
| 34 | 106328085 | 4 | 5 | 0.1268 | 0.5722 | -0.000083 |
| 35 | 30349564 | 4 | 5 | 0.4824 | 0.4867 | 0.000042 |
| 36 | 1855189739 | 4 | 5 | 0.4566 | 0.4516 | 0.000003 |
| 37 | 99052376 | 4 | 5 | 0.5403 | 0.4682 | 0.000013 |
| 38 | 1250819632 | 4 | 5 | 0.4544 | 0.4738 | 0.000012 |
| 39 | 106406362 | 4 | 5 | 0.2549 | 0.4952 | -0.000009 |
