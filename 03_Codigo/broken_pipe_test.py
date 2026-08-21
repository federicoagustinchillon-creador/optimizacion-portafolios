"""
================================================================================
BROKEN PIPE TEST -- "Regla de Oro" de auditoría de laboratorio
================================================================================
Pedido explícito del usuario (auditoría extendida, 2026-08-21): "Introduce
ruido blanco puro (datos aleatorios falsos con la misma media y varianza) en
el 50% de tus activos históricos y corre todo tu pipeline completo (GARCH-EVT
+ HERC + CPCV + DSR). Si tus métricas finales... te siguen diciendo que la
estrategia es rentable en esos activos falsos, tu pipeline tiene una fuga de
información matemática o un error de código interno."

A diferencia del proyecto hermano (herc-portfolio-optimizer, motor en vivo
con walk-forward de producción), este repo es un pipeline académico de un
solo script (pipeline_optimizacion.py) sobre un universo FIJO de 9 activos.
Este módulo reutiliza sus piezas reales -- estimar_covarianza_y_correlacion,
calcular_distancia_angular, obtener_agrupamiento_jerarquico, optimizar_herc,
ejecutar_cpcv, calcular_ratio_sharpe_deflactado -- para reemplazar 50% del
universo (aquí, 4-5 de los 9 tickers) por ruido blanco y correr el mismo
walk-forward + CPCV + DSR sobre el universo mixto, exactamente como lo pide
el usuario.

IMPORTANTE -- a diferencia de herc-portfolio-optimizer (donde el dataset de
retornos no está comiteado y este entorno no tiene acceso de red a ningún
proveedor de precios), este repo SÍ tiene datos reales comiteados
(05_Datos_y_Resultados/precios_ajustados.csv, SPY/QQQ/EEM/VNQ/TLT/IEF/LQD/
GLD/DBC, 2007-2026, ~4926 ruedas) -- esta corrida es sobre MERCADO REAL, no
un universo sintético de validación.

Uso: python 03_Codigo/broken_pipe_test.py
"""
from __future__ import annotations
import os
import sys
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_optimizacion import (  # noqa: E402
    calcular_rendimientos, estimar_covarianza_y_correlacion, calcular_distancia_angular,
    obtener_agrupamiento_jerarquico, optimizar_herc, calcular_ratio_sharpe_deflactado,
    calcular_ratio_sharpe_probabilistico, ejecutar_cpcv, TICKERS, DIR_DATOS,
)

RF_ANUAL = 0.02  # misma tasa libre de riesgo que calcular_tabla_metricas/ejecutar_cpcv en el pipeline original


def calcular_distancia_angular_abs_corr(matriz_corr: np.ndarray) -> np.ndarray:
    """
    Auditoría extendida, Sección 4 ("Tratamiento de Correlaciones Negativas"):
    D = 1 - |rho| en vez de D = sqrt(0.5*(1-rho)) (calcular_distancia_angular
    del pipeline original). Mismo hallazgo que en el repo hermano
    (herc-portfolio-optimizer/src/backtest_engine.py::compute_dist_matrix_abs_corr):
    con la métrica clásica, un activo con rho=-0.9 respecto de otro queda a
    D=0.975, MÁS LEJOS que un activo sin ninguna relación (rho=0 -> D=0.707)
    -- el linkage de Ward deja la cobertura para el final y casi siempre la
    aísla en su propio cluster. D=1-|rho| invierte esto: rho=+1 y rho=-1
    quedan ambos a distancia 0 (mismo vecindario inmediato del árbol).

    No reemplaza calcular_distancia_angular en pipeline_optimizacion.py (ver
    nota de alcance en AUDITORIA_EXTENDIDA_MICROESTRUCTURA_MATEMATICA.md) --
    disponible para quien quiera correr el pipeline con esta métrica
    alternativa sobre este universo.
    """
    distancias = np.clip(1.0 - np.abs(matriz_corr), 0.0, 1.0)
    np.fill_diagonal(distancias, 0.0)
    return distancias


def inject_white_noise(precios: pd.DataFrame, frac_noise: float = 0.5, seed: int = 42,
                        noise_mean_mode: str = "zero") -> dict:
    """
    Reemplaza `frac_noise` de las columnas de `precios` por una serie de
    precios reconstruida a partir de log-retornos i.i.d. Normal(mu, sigma) --
    sigma calibrado al propio activo real reemplazado, mu=0 por defecto.

    [DECISION AUTONOMA -- mismo hallazgo que en herc-portfolio-optimizer,
    ver src/broken_pipe_test.py de ese repo]: noise_mean_mode='matched' (mu
    TAMBIÉN calibrado al activo real, la lectura literal del pedido) le da
    al ruido, sobre un universo con drift positivo real (como este --
    SPY/QQQ/etc. con >15 años de historia y prima de riesgo positiva), la
    misma esperanza que el activo real -- una estrategia que gana dinero
    sosteniéndolo no demuestra fuga, cobra correctamente esa prima. mu=0
    ('zero') es la única calibración consistente con la premisa lógica del
    propio test ("ningún activo de ruido i.i.d. puede tener retorno
    esperado positivo"), y es el default.
    """
    if noise_mean_mode not in ("zero", "matched"):
        raise ValueError(f"noise_mean_mode debe ser 'zero' o 'matched', recibido: {noise_mean_mode!r}")
    cols = list(precios.columns)
    n = len(cols)
    if n < 2:
        raise ValueError("broken_pipe_test necesita al menos 2 activos")

    n_noise = max(1, int(np.floor(n * frac_noise))) if frac_noise > 0 else 0
    n_noise = min(n_noise, n - 1)

    _, rendimientos_log = calcular_rendimientos(precios)
    rng = np.random.RandomState(seed)
    noise_idx = rng.choice(n, size=n_noise, replace=False)
    noise_cols = [cols[i] for i in noise_idx]
    real_cols = [c for c in cols if c not in noise_cols]

    precios_mixtos = precios.copy()
    for col in noise_cols:
        r = rendimientos_log[col].dropna().to_numpy()
        mu = float(np.mean(r)) if noise_mean_mode == "matched" else 0.0
        sigma = max(float(np.std(r, ddof=0)), 1e-12)
        synthetic_log_ret = rng.normal(mu, sigma, size=len(precios))
        synthetic_log_ret[0] = 0.0  # ancla el primer nivel al precio inicial real
        precio_inicial = float(precios[col].iloc[0])
        precios_mixtos[col] = precio_inicial * np.exp(np.cumsum(synthetic_log_ret))

    return {"precios_mixtos": precios_mixtos, "noise_cols": noise_cols, "real_cols": real_cols,
            "n_noise": n_noise, "n_real": n - n_noise}


def _herc_walk_forward_daily_weights(precios_mixtos: pd.DataFrame, dias_ventana: int = 756,
                                      dias_rebalanceo: int = 21, tasa_friccion_bps: float = 5.0
                                      ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Replica EXACTAMENTE la rama HERC de ejecutar_walk_forward (mismo costo,
    misma deriva geométrica de pesos entre rebalanceos) pero devuelve el
    HISTORIAL DIARIO de pesos (no sólo en fechas de rebalanceo, como
    dfs_ponderaciones de ejecutar_walk_forward) -- necesario para poder
    calcular la contribución diaria EXACTA del sleeve de ruido al retorno
    del portafolio, el diagnóstico central de este test.
    """
    tickers = list(precios_mixtos.columns)
    n_activos = len(tickers)
    rendimientos_simples, rendimientos_log = calcular_rendimientos(precios_mixtos)
    n_dias = len(precios_mixtos)
    factor_costo = tasa_friccion_bps / 10000.0
    fechas = rendimientos_simples.index

    pesos_vigentes = np.zeros(n_activos)
    port_returns = []
    daily_weights = []

    for t in range(dias_ventana, n_dias - 1):
        es_dia_rebalanceo = ((t - dias_ventana) % dias_rebalanceo == 0)
        rendimientos_dia = rendimientos_simples.iloc[t].values

        if es_dia_rebalanceo:
            ventana_log = rendimientos_log.iloc[t - dias_ventana:t]
            cov_estimada, corr_estimada = estimar_covarianza_y_correlacion(ventana_log)
            distancias = calcular_distancia_angular(corr_estimada)
            arbol_enlace, orden_cuasidiagonal = obtener_agrupamiento_jerarquico(distancias, metodo_enlace="ward")
            pesos_objetivo = optimizar_herc(cov_estimada, arbol_enlace, orden_cuasidiagonal)
            rotacion = np.sum(np.abs(pesos_objetivo - pesos_vigentes))
            costo_operativo = rotacion * factor_costo
            pesos_vigentes = pesos_objetivo
        else:
            costo_operativo = 0.0

        r_cartera = float(np.dot(pesos_vigentes, rendimientos_dia)) - costo_operativo
        port_returns.append(r_cartera)
        daily_weights.append(pesos_vigentes.copy())

        nuevos_pesos = pesos_vigentes * (1.0 + rendimientos_dia)
        suma_pesos = np.sum(nuevos_pesos)
        if suma_pesos > 0:
            pesos_vigentes = nuevos_pesos / suma_pesos

    fechas_oos = fechas[dias_ventana:n_dias - 1]
    port_returns = pd.Series(port_returns, index=fechas_oos, name="HERC")
    daily_weights_df = pd.DataFrame(daily_weights, index=fechas_oos, columns=tickers)
    return port_returns, daily_weights_df


def _sharpe(returns: pd.Series, rf_anual: float = RF_ANUAL) -> float:
    vol = float(returns.std(ddof=1)) * np.sqrt(252.0)
    if vol < 1e-12:
        return 0.0
    ret_anual = float(returns.mean()) * 252.0
    return (ret_anual - rf_anual) / vol


def broken_pipe_test(precios: pd.DataFrame, frac_noise: float = 0.5, n_trials: int = 20, seed: int = 42,
                      dias_ventana: int = 756, dias_rebalanceo: int = 21, tasa_friccion_bps: float = 5.0,
                      noise_mean_mode: str = "zero", run_cpcv: bool = True) -> dict:
    """Ver docstring del módulo. Devuelve veredicto de 3 niveles (mismo
    criterio que herc-portfolio-optimizer/src/broken_pipe_test.py):
    'fuga_clara' (p<0.01), 'senal_debil' (0.01<=p<0.05, no concluyente),
    'sin_evidencia' (p>=0.05) -- SOLO sobre la contribución del sleeve de
    ruido, nunca sobre el Sharpe del portafolio completo (que será positivo
    y significativo casi siempre que el sleeve REAL tenga prima de riesgo
    real, sin que eso diga nada sobre fuga -- mismo hallazgo metodológico
    documentado en el repo hermano)."""
    rng_master = np.random.RandomState(seed)
    trial_seeds = rng_master.randint(0, 2**31 - 1, size=n_trials)

    trials = []
    for i, s in enumerate(trial_seeds):
        injected = inject_white_noise(precios, frac_noise=frac_noise, seed=int(s), noise_mean_mode=noise_mean_mode)
        port_returns, daily_weights = _herc_walk_forward_daily_weights(
            injected["precios_mixtos"], dias_ventana=dias_ventana,
            dias_rebalanceo=dias_rebalanceo, tasa_friccion_bps=tasa_friccion_bps,
        )
        _, rendimientos_simples_mixto = calcular_rendimientos(injected["precios_mixtos"])
        noise_cols = injected["noise_cols"]
        aligned = rendimientos_simples_mixto.reindex(daily_weights.index)[noise_cols]
        noise_contrib = (daily_weights[noise_cols] * aligned).sum(axis=1)

        trials.append({
            "seed": int(s), "n_noise": injected["n_noise"], "n_real": injected["n_real"],
            "portfolio_sharpe": _sharpe(port_returns),
            "noise_sleeve_mean_daily_contribution": float(noise_contrib.mean()),
            "noise_sleeve_avg_weight": float(daily_weights[noise_cols].sum(axis=1).mean()),
            "n_obs": int(len(port_returns)),
        })

    sharpes = np.array([t["portfolio_sharpe"] for t in trials])
    contribs = np.array([t["noise_sleeve_mean_daily_contribution"] for t in trials])
    sharpe_t, sharpe_p = ttest_1samp(sharpes, 0.0)
    contrib_t, contrib_p = ttest_1samp(contribs, 0.0)
    mean_sharpe, mean_contrib = float(np.mean(sharpes)), float(np.mean(contribs))

    contrib_significativo = mean_contrib > 0
    if contrib_significativo and contrib_p < 0.01:
        nivel, fuga = "fuga_clara", True
        veredicto = (f"FUGA DE INFORMACION DETECTADA (p<0.01): contribucion media diaria del sleeve de "
                     f"ruido = {mean_contrib:.6f}, t={contrib_t:.3f}, p={contrib_p:.5f} a traves de "
                     f"{n_trials} trials. Revisar antes de arriesgar capital real.")
    elif contrib_significativo and contrib_p < 0.05:
        nivel, fuga = "senal_debil", False
        veredicto = (f"SENAL DEBIL, NO CONCLUYENTE (0.01<=p<0.05): contribucion media diaria = "
                     f"{mean_contrib:.6f}, p={contrib_p:.5f}. Se recomienda mas trials antes de escalar.")
    else:
        nivel, fuga = "sin_evidencia", False
        veredicto = (f"SIN EVIDENCIA DE FUGA: contribucion media diaria del sleeve de ruido "
                     f"({mean_contrib:.6f}) no es significativamente distinta de cero (p={contrib_p:.5f}) "
                     f"a traves de {n_trials} trials con semillas independientes.")

    cpcv_result = None
    dsr_result = None
    if run_cpcv:
        base = inject_white_noise(precios, frac_noise=frac_noise, seed=seed, noise_mean_mode=noise_mean_mode)
        df_cpcv = ejecutar_cpcv(base["precios_mixtos"], TICKERS, n_bloques=6, k_bloques_test=2, dias_embargo=10)
        cpcv_result = {
            col: {"sharpe_mean": float(df_cpcv[col].mean()), "sharpe_std": float(df_cpcv[col].std()),
                  "n_combos": int(df_cpcv[col].count())}
            for col in df_cpcv.columns
        }

        last_precios_mixtos = inject_white_noise(
            precios, frac_noise=frac_noise, seed=int(trial_seeds[-1]), noise_mean_mode=noise_mean_mode
        )["precios_mixtos"]
        last_returns, _ = _herc_walk_forward_daily_weights(
            last_precios_mixtos, dias_ventana=dias_ventana, dias_rebalanceo=dias_rebalanceo,
            tasa_friccion_bps=tasa_friccion_bps,
        )
        sr_anual = _sharpe(last_returns)  # _sharpe ya devuelve el ratio anualizado
        dsr = calcular_ratio_sharpe_deflactado(
            sharpe_anual=sr_anual, asimetria=float(last_returns.skew()),
            curtosis_exceso=float(last_returns.kurtosis()), n_observaciones=len(last_returns),
            n_modelos_evaluados=n_trials, varianza_sharpes=float(np.var(sharpes, ddof=1)),
        )
        dsr_result = {"Sharpe anualizado (ultimo trial)": sr_anual, "Deflated Sharpe Ratio": dsr,
                      "N modelos (trials)": n_trials}

    return {
        "config": {"frac_noise": frac_noise, "n_trials": n_trials, "dias_ventana": dias_ventana,
                    "dias_rebalanceo": dias_rebalanceo, "noise_mean_mode": noise_mean_mode,
                    "n_activos_universo": precios.shape[1], "n_obs_universo": len(precios)},
        "sharpe_medio_portafolio": mean_sharpe, "sharpe_t_stat": float(sharpe_t), "sharpe_p_value": float(sharpe_p),
        "contribucion_media_sleeve_ruido": mean_contrib,
        "contribucion_t_stat": float(contrib_t), "contribucion_p_value": float(contrib_p),
        "nivel_evidencia": nivel, "fuga_de_informacion_detectada": fuga, "veredicto": veredicto,
        "cpcv": cpcv_result, "dsr": dsr_result, "trials": trials,
    }


def main():
    precios = pd.read_csv(os.path.join(DIR_DATOS, "precios_ajustados.csv"), index_col=0, parse_dates=True)
    precios = precios[TICKERS]

    n_trials = 40  # 20 dio un p-valor marginal (0.026) -- se sube la potencia antes de reportar un veredicto final
    result_zero = broken_pipe_test(precios, frac_noise=0.5, n_trials=n_trials, seed=42, noise_mean_mode="zero")
    result_matched = broken_pipe_test(precios, frac_noise=0.5, n_trials=n_trials, seed=42,
                                       noise_mean_mode="matched", run_cpcv=False)

    lines = ["# Broken Pipe Test -- Resultado de Corrida (Datos Reales)", ""]
    lines.append(
        "> Regla de Oro: ruido blanco puro en el 50% de los activos (4-5 de los 9 tickers reales), "
        "motor real (HERC walk-forward de `pipeline_optimizacion.py`) + CPCV + DSR. **Corrida sobre "
        "MERCADO REAL** (`05_Datos_y_Resultados/precios_ajustados.csv`, SPY/QQQ/EEM/VNQ/TLT/IEF/LQD/"
        "GLD/DBC, 2007-2026) -- a diferencia del repo hermano herc-portfolio-optimizer, este proyecto "
        "SI tiene datos de precios reales comiteados."
    )
    lines.append("")
    lines.append(f"**Configuración:** `{json.dumps(result_zero['config'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Hallazgo metodológico: `noise_mean_mode='zero'` vs `'matched'`")
    lines.append("")
    lines.append(
        "Mismo hallazgo que en el repo hermano: calibrar el ruido a la misma media del activo real "
        "que reemplaza (`matched`, lectura literal del pedido) le da al ruido la prima de riesgo real "
        "del universo (SPY/QQQ 2007-2026 tuvieron drift positivo) — no es fuga, es la prima que el "
        "propio generador de ruido regaló. `zero` (mu=0 exacto) es el default correcto."
    )
    lines.append("")
    lines.append("| Modo | Contribución media diaria sleeve ruido | p-valor | Nivel de evidencia |")
    lines.append("|---|---|---|---|")
    lines.append(f"| `zero` (default) | {result_zero['contribucion_media_sleeve_ruido']:.6f} | "
                  f"{result_zero['contribucion_p_value']:.5f} | {result_zero['nivel_evidencia']} |")
    lines.append(f"| `matched` (lectura literal) | {result_matched['contribucion_media_sleeve_ruido']:.6f} | "
                  f"{result_matched['contribucion_p_value']:.5f} | {result_matched['nivel_evidencia']} |")
    lines.append("")
    lines.append("## Veredicto (modo `zero`, el correcto)")
    lines.append("")
    lines.append(f"**{result_zero['nivel_evidencia'].upper().replace('_', ' ')}**")
    lines.append("")
    lines.append(result_zero["veredicto"])
    lines.append("")
    lines.append("## Métricas agregadas (modo `zero`)")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Sharpe medio del portafolio (entre trials) | {result_zero['sharpe_medio_portafolio']:.4f} |")
    lines.append(f"| Sharpe p-value (vs 0) | {result_zero['sharpe_p_value']:.5f} |")
    lines.append(f"| Contribución media diaria sleeve ruido | {result_zero['contribucion_media_sleeve_ruido']:.6f} |")
    lines.append(f"| Contribución p-value | {result_zero['contribucion_p_value']:.5f} |")
    if result_zero["dsr"]:
        for k, v in result_zero["dsr"].items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

    if result_zero["cpcv"]:
        lines.append("## CPCV sobre universo mixto (modo `zero`)")
        lines.append("")
        lines.append("| Modelo | Sharpe medio OOS | Sharpe std | N combinaciones |")
        lines.append("|---|---|---|---|")
        for m, r in result_zero["cpcv"].items():
            lines.append(f"| {m} | {r['sharpe_mean']:.4f} | {r['sharpe_std']:.4f} | {r['n_combos']} |")
        lines.append("")

    lines.append("## Detalle por trial (modo `zero`)")
    lines.append("")
    lines.append("| Trial | Seed | N ruido | N real | Sharpe portafolio | Peso medio sleeve ruido | Contribución media diaria |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, t in enumerate(result_zero["trials"]):
        lines.append(f"| {i} | {t['seed']} | {t['n_noise']} | {t['n_real']} | {t['portfolio_sharpe']:.4f} | "
                      f"{t['noise_sleeve_avg_weight']:.4f} | {t['noise_sleeve_mean_daily_contribution']:.6f} |")
    lines.append("")

    out_path = os.path.join(DIR_DATOS, "BROKEN_PIPE_TEST_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Reporte escrito en {out_path}")


if __name__ == "__main__":
    main()
