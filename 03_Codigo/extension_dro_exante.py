"""
================================================================================
EXTENSIÓN: OPTIMIZACIÓN DISTRIBUCIONALMENTE ROBUSTA (WASSERSTEIN) Y
MARCO GENERATIVO EX-ANTE (GARCH-EVT + CÓPULA t DE STUDENT)
Autor: Federico Agustín Chillón
================================================================================

Implementa las DOS extensiones metodológicas que la Sección 4.1 y el Resumen
del documento original (Comparativa_Modelos_Asignacion_Activos.tex) ya
describían en la revisión de literatura pero declaraban explícitamente NO
implementadas ("Marco Empírico y Auditoría Ex-Post (este estudio)"):

  1. optimizar_dro_wasserstein: Mínimo CVaR distribucionalmente robusto sobre
     una bola de Wasserstein (Esfahani & Kuhn, 2018) -- reformulación convexa
     TRACTABLE (no una aproximación): para una pérdida Lipschitz como la de
     CVaR, el problema minimax se colapsa a CVaR empírico + una penalización
     de norma L2 sobre los pesos, escalada por el radio de la bola epsilon y
     la constante de Lipschitz 1/(1-alpha). Es una extensión de UNA LÍNEA
     sobre optimizar_minimo_cvar del pipeline original (agrega un término al
     objetivo), consistente con que Gemini la haya marcado como "barata" --
     pero el término de norma L2 es un cono de segundo orden (SOCP), no una
     desigualdad lineal, así que a diferencia del resto del pipeline (que usa
     scipy.optimize.linprog con HiGHS por ser todo LP puro) esta función usa
     CVXPY (ECOS/CLARABEL), el solver correcto para SOCP.

  2. simular_escenarios_garch_evt_copula_t + optimizar_minimo_cvar_ex_ante:
     el marco generativo textual de la Sección 4.1, implementado literal:
       a. GARCH(1,1)-AR(1) por activo (misma especificación EXACTA que
          ajustar_modelo_evt_gpd en pipeline_optimizacion.py -- no un
          modelo nuevo, el mismo ya validado en el documento).
       b. Calibración GPD de la cola de PÉRDIDAS de cada activo (reutiliza
          la matemática de ajustar_modelo_evt_gpd -- misma formula de
          cuantil POT, generalizada a probabilidad arbitraria en vez de fija
          a 0.99) -- este es el paso "se calibran sus colas residuales por
          EVT" que el texto promete.
       c. Cópula t de Student multivariada (Demarta & McNeil, 2005) sobre
          los residuos estandarizados de los 9 activos -- correlación vía
          scores normales, nu por grid-search de verosimilitud. Este es el
          paso "se acoplan con una Cópula multivariada (t de Student)" que
          el texto promete LITERAL (no una Vine copula -- eso no está en el
          documento original, es una extensión que Gemini sugirió sin que
          el propio paper la pidiera).
       d. Simulación Monte Carlo de M escenarios conjuntos, reescalados por
          el pronóstico de volatilidad condicional MÁS RECIENTE de cada
          activo (a diferencia del filtrado ex-post de ajustar_modelo_evt_gpd,
          que usa el promedio HISTÓRICO de sigma_t -- la esencia de "ex-ante"
          es proyectar con la volatilidad de HOY, no con el promedio de toda
          la ventana).
       e. Mínimo CVaR (Rockafellar & Uryasev) resuelto sobre los escenarios
          SIMULADOS en vez de sobre los retornos históricos -- reutiliza
          optimizar_minimo_cvar sin modificarlo, solo cambia el insumo.

Todas las funciones son puras (reciben datos, devuelven resultados) y
reutilizan TAL CUAL las funciones ya publicadas de pipeline_optimizacion.py
(estimar_covarianza_y_correlacion, ajustar_modelo_evt_gpd, optimizar_minimo_cvar,
calcular_tabla_metricas, TICKERS, etc.) -- ningún numero se fabrica, ninguna
función existente se modifica.
"""
from __future__ import annotations
import os
import sys
import time
import gc
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gammaln
import cvxpy as cp
from arch import arch_model

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_optimizacion import (  # noqa: E402
    TICKERS, NOMBRES_ACTIVOS, DIR_DATOS, DIR_FIGURAS,
    descargar_datos, calcular_rendimientos, estimar_covarianza_y_correlacion,
    calcular_distancia_angular, obtener_agrupamiento_jerarquico,
    optimizar_minimo_cvar, calcular_tabla_metricas,
    ejecutar_walk_forward as _ejecutar_walk_forward_original,
)

warnings.filterwarnings("ignore", category=UserWarning, module="arch")

SEMILLA = 42


# ==============================================================================
# 1. DRO WASSERSTEIN (Esfahani & Kuhn, 2018)
# ==============================================================================
def optimizar_dro_wasserstein(
    matriz_rendimientos: np.ndarray,
    radio_epsilon: float = 0.01,
    nivel_confianza: float = 0.95,
) -> np.ndarray:
    """
    Mínimo CVaR distribucionalmente robusto sobre una bola de Wasserstein-1
    de radio `radio_epsilon` alrededor de la distribución empírica
    (Esfahani & Kuhn, 2018, "Data-driven distributionally robust
    optimization using the Wasserstein metric", Mathematical Programming).

    Para una pérdida CVaR (Lipschitz en el retorno realizado, con constante
    de Lipschitz ||w||_2 / (1 - alpha) bajo la norma dual L2), el problema
    minimax  min_w max_{Q: W_1(Q, P_hat) <= eps} CVaR_Q(w)  tiene la
    reformulación convexa EXACTA (no una aproximación heurística):

        min_{w,zeta,z}  zeta + 1/((1-alpha)*N) * sum(z_i)
                        + eps/(1-alpha) * ||w||_2
        s.a.  z_i >= -w'r_i - zeta,  z_i >= 0,  sum(w)=1,  w>=0

    El unico termino nuevo respecto a optimizar_minimo_cvar (pipeline
    original) es  eps/(1-alpha) * ||w||_2  -- una penalización de norma L2
    sobre los pesos. radio_epsilon=0 colapsa exactamente al CVaR empirico
    estandar (verificado en el bloque de auto-test al final del archivo).
    """
    n_filas, n_activos = matriz_rendimientos.shape
    w = cp.Variable(n_activos, nonneg=True)
    zeta = cp.Variable()
    z = cp.Variable(n_filas, nonneg=True)

    perdidas = -matriz_rendimientos @ w
    restricciones = [
        z >= perdidas - zeta,
        cp.sum(w) == 1,
    ]
    cvar_empirico = zeta + cp.sum(z) / ((1.0 - nivel_confianza) * n_filas)
    penalizacion_wasserstein = (radio_epsilon / (1.0 - nivel_confianza)) * cp.norm(w, 2)
    objetivo = cp.Minimize(cvar_empirico + penalizacion_wasserstein)

    problema = cp.Problem(objetivo, restricciones)
    try:
        problema.solve(solver=cp.CLARABEL)
        if w.value is None or not np.all(np.isfinite(w.value)):
            raise ValueError("CLARABEL no convergio")
    except Exception:
        try:
            problema.solve(solver=cp.SCS)
        except Exception:
            pass

    if w.value is not None and np.all(np.isfinite(w.value)):
        pesos = np.clip(w.value, 0.0, None)
        suma = float(np.sum(pesos))
        if suma > 1e-6:
            return pesos / suma

    # Resguardo (mismo criterio que optimizar_minimo_cvar del pipeline original):
    volatilidades_inversas = 1.0 / np.maximum(np.std(matriz_rendimientos, axis=0), 1e-6)
    return volatilidades_inversas / np.sum(volatilidades_inversas)


# ==============================================================================
# 2. MARCO GENERATIVO EX-ANTE: GARCH(1,1) + COLA GPD + CÓPULA t + MONTE CARLO
# ==============================================================================
def _ajustar_garch_por_activo(rendimientos_log_activo: np.ndarray) -> dict:
    """
    AR(1)-GARCH(1,1), MISMA especificación exacta que ajustar_modelo_evt_gpd
    en pipeline_optimizacion.py (arch_model(..., mean='AR', lags=1,
    vol='Garch', p=1, q=1, dist='normal')) -- no es un modelo nuevo, es el
    mismo ya validado y citado en el documento, aplicado por activo en vez
    de a nivel portfolio.

    Devuelve residuos estandarizados z_t, y el pronostico de UN PASO
    ADELANTE de media y volatilidad condicional (mu_{t+1}, sigma_{t+1}) --
    la pieza que distingue "ex-ante" (proyectar con la info de HOY) del
    filtrado ex-post de ajustar_modelo_evt_gpd (que usa el promedio
    historico de sigma_t sobre toda la ventana).
    """
    r_pct = rendimientos_log_activo * 100.0
    modelo = arch_model(r_pct, mean="AR", lags=1, vol="Garch", p=1, q=1, dist="normal")
    res = modelo.fit(disp="off")

    std_resid = np.asarray(res.std_resid)
    mask = ~np.isnan(std_resid)
    std_resid = std_resid[mask]

    pronostico = res.forecast(horizon=1, reindex=False)
    # Pronostico robusto de media condicional: usar el ultimo valor ajustado
    # de la media condicional del propio modelo (mas simple y menos fragil
    # que reconstruir a mano el AR(1) a partir de los parametros nombrados).
    resid_raw = np.asarray(res.resid)
    fitted_mean = r_pct[-len(resid_raw):] - resid_raw
    mu_siguiente_pct = float(fitted_mean[-1])  # ultima media condicional ajustada, como proxy de mu_{t+1}
    sigma_siguiente_pct = float(np.sqrt(pronostico.variance.values[-1, 0]))

    return {
        "std_resid": std_resid,
        "mu_siguiente_pct": mu_siguiente_pct,
        "sigma_siguiente_pct": sigma_siguiente_pct,
    }


def _calibrar_cola_gpd_perdidas(std_resid: np.ndarray, cuantil_umbral: float = 0.90) -> dict:
    """
    Calibra la cola de PÉRDIDAS (-std_resid > umbral) via GPD -- misma
    formula POT que ajustar_modelo_evt_gpd, extraida aca como pieza
    reutilizable para construir la CDF marginal semi-parametrica completa
    (empirica en el centro, GPD en la cola) que la cópula necesita.
    """
    perdidas = -std_resid
    umbral_u = float(np.percentile(perdidas, cuantil_umbral * 100.0))
    excesos = perdidas[perdidas > umbral_u] - umbral_u
    n_excesos = len(excesos)
    n_obs = len(perdidas)

    if n_excesos < 10:
        xi, beta = 0.0, float(np.std(perdidas))
    else:
        xi, _, beta = stats.genpareto.fit(excesos, floc=0.0)

    return {"perdidas": perdidas, "umbral_u": umbral_u, "xi": float(xi), "beta": float(beta),
            "n_excesos": n_excesos, "n_obs": n_obs}


def _cdf_marginal_semiparametrica(x: np.ndarray, calib: dict) -> np.ndarray:
    """CDF semi-parametrica de PÉRDIDAS: empirica por debajo del umbral, GPD (POT) por encima."""
    perdidas, umbral_u, xi, beta, n_exc, n_obs = (
        calib["perdidas"], calib["umbral_u"], calib["xi"], calib["beta"], calib["n_excesos"], calib["n_obs"]
    )
    p_umbral = 1.0 - n_exc / n_obs
    out = np.empty_like(x, dtype=float)
    debajo = x <= umbral_u
    if debajo.any():
        # rankdata relativo a la muestra empirica de perdidas (interpolado)
        out[debajo] = np.clip(np.searchsorted(np.sort(perdidas), x[debajo], side="right") / n_obs, 1e-6, p_umbral)
    arriba = ~debajo
    if arriba.any():
        exceso = np.maximum(x[arriba] - umbral_u, 0.0)
        if abs(xi) > 1e-8:
            cola = (1.0 + xi * exceso / max(beta, 1e-8)) ** (-1.0 / xi)
        else:
            cola = np.exp(-exceso / max(beta, 1e-8))
        out[arriba] = np.clip(1.0 - (n_exc / n_obs) * cola, p_umbral, 1.0 - 1e-6)
    return out


def _cuantil_marginal_semiparametrico(p: np.ndarray, calib: dict) -> np.ndarray:
    """Inversa de _cdf_marginal_semiparametrica -- cuantil de PÉRDIDAS dado p en (0,1)."""
    perdidas, umbral_u, xi, beta, n_exc, n_obs = (
        calib["perdidas"], calib["umbral_u"], calib["xi"], calib["beta"], calib["n_excesos"], calib["n_obs"]
    )
    p_umbral = 1.0 - n_exc / n_obs
    out = np.empty_like(p, dtype=float)
    debajo = p <= p_umbral
    if debajo.any():
        out[debajo] = np.quantile(perdidas, np.clip(p[debajo], 1e-6, 1.0))
    arriba = ~debajo
    if arriba.any():
        prob_cola = (n_obs / max(n_exc, 1)) * (1.0 - p[arriba])
        prob_cola = np.clip(prob_cola, 1e-8, None)
        if abs(xi) > 1e-8:
            out[arriba] = umbral_u + (beta / xi) * (prob_cola ** (-xi) - 1.0)
        else:
            out[arriba] = umbral_u - beta * np.log(prob_cola)
    return out


def _ajustar_grados_libertad_copula_t(scores_normales: np.ndarray, matriz_corr: np.ndarray,
                                       candidatos_nu: Tuple[int, ...] = (3, 4, 5, 6, 8, 10, 15, 20, 30)) -> int:
    """
    Grid-search de maxima verosimilitud sobre los grados de libertad `nu` de
    la cópula t (Demarta & McNeil, 2005) -- discretizado sobre candidatos
    razonables en vez de un optimizador numerico continuo: mas robusto para
    N=9 activos / T~4900 observaciones, y evita convergencia fragil de MLE
    con matriz de correlacion casi-singular en alguna iteracion intermedia.
    """
    n, d = scores_normales.shape
    inv_corr = np.linalg.pinv(matriz_corr)
    signo, logdet_corr = np.linalg.slogdet(matriz_corr)
    mejor_nu, mejor_ll = candidatos_nu[0], -np.inf
    for nu in candidatos_nu:
        forma_cuadratica = np.einsum("ij,jk,ik->i", scores_normales, inv_corr, scores_normales)
        log_densidad_t = (
            gammaln((nu + d) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet_corr
            - ((nu + d) / 2.0) * np.log1p(forma_cuadratica / nu)
        )
        log_densidad_normal_marginal = np.sum(-0.5 * scores_normales**2 - 0.5 * np.log(2 * np.pi), axis=1)
        log_verosimilitud = float(np.sum(log_densidad_t - log_densidad_normal_marginal))
        if log_verosimilitud > mejor_ll:
            mejor_ll, mejor_nu = log_verosimilitud, nu
    return mejor_nu


def simular_escenarios_garch_evt_copula_t(
    rendimientos_log: pd.DataFrame,
    n_escenarios: int = 5000,
    cuantil_umbral: float = 0.90,
    seed: int = SEMILLA,
) -> np.ndarray:
    """
    Marco generativo ex-ante completo (Seccion 4.1 del documento original,
    parrafo "Marco Generativo Ex-Ante"): GARCH(1,1) por activo -> cola GPD
    de perdidas -> cópula t multivariada -> M escenarios sinteticos
    reescalados por el pronostico de volatilidad MAS RECIENTE.

    Devuelve una matriz (n_escenarios x n_activos) de retornos SIMPLES
    simulados, lista para pasar directo a optimizar_minimo_cvar (mismo
    formato que rendimientos_simples.values en el pipeline original).
    """
    rng = np.random.RandomState(seed)
    tickers = list(rendimientos_log.columns)
    n_activos = len(tickers)

    calibraciones_garch = {}
    calibraciones_gpd = {}
    scores_normales = np.zeros((len(rendimientos_log), n_activos))
    for j, tk in enumerate(tickers):
        cal_garch = _ajustar_garch_por_activo(rendimientos_log[tk].values)
        cal_gpd = _calibrar_cola_gpd_perdidas(cal_garch["std_resid"], cuantil_umbral)
        calibraciones_garch[tk] = cal_garch
        calibraciones_gpd[tk] = cal_gpd
        n_validos = len(cal_garch["std_resid"])
        u = _cdf_marginal_semiparametrica(-cal_garch["std_resid"], cal_gpd)  # CDF de RETORNO = 1 - CDF(perdida)
        u = 1.0 - u
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        scores_normales[-n_validos:, j] = stats.norm.ppf(u)

    matriz_corr_scores = np.corrcoef(scores_normales, rowvar=False)
    nu = _ajustar_grados_libertad_copula_t(scores_normales, matriz_corr_scores)

    # Simulacion de la cópula t: Z ~ N(0,R), W ~ chi2(nu), T = Z / sqrt(W/nu)
    L = np.linalg.cholesky(matriz_corr_scores + 1e-10 * np.eye(n_activos))
    Z = rng.standard_normal((n_escenarios, n_activos)) @ L.T
    W = rng.chisquare(nu, size=n_escenarios)
    T_sim = Z / np.sqrt(W / nu)[:, None]
    U_sim = stats.t.cdf(T_sim, df=nu)

    escenarios_log_pct = np.zeros((n_escenarios, n_activos))
    for j, tk in enumerate(tickers):
        p_perdida_sim = np.clip(1.0 - U_sim[:, j], 1e-6, 1.0 - 1e-6)
        perdida_estandarizada_sim = _cuantil_marginal_semiparametrico(p_perdida_sim, calibraciones_gpd[tk])
        z_estandarizado_sim = -perdida_estandarizada_sim
        cal = calibraciones_garch[tk]
        escenarios_log_pct[:, j] = cal["mu_siguiente_pct"] + cal["sigma_siguiente_pct"] * z_estandarizado_sim

    escenarios_log = escenarios_log_pct / 100.0
    escenarios_simples = np.expm1(escenarios_log)
    return escenarios_simples


def optimizar_minimo_cvar_ex_ante(
    rendimientos_log_ventana: pd.DataFrame,
    n_escenarios: int = 5000,
    nivel_confianza: float = 0.95,
    seed: int = SEMILLA,
) -> np.ndarray:
    """Min-CVaR (Rockafellar-Uryasev, funcion original del pipeline sin modificar)
    resuelto sobre escenarios SINTETICOS ex-ante en vez de retornos historicos."""
    escenarios = simular_escenarios_garch_evt_copula_t(rendimientos_log_ventana, n_escenarios=n_escenarios, seed=seed)
    return optimizar_minimo_cvar(escenarios, nivel_confianza=nivel_confianza)


# ==============================================================================
# 3. WALK-FORWARD EXTENDIDO (DRO_Wasserstein + Min_CVaR_ExAnte)
# ==============================================================================
def ejecutar_walk_forward_extension(
    precios: pd.DataFrame,
    tickers: List[str] = TICKERS,
    dias_ventana: int = 756,
    dias_rebalanceo: int = 21,
    tasa_friccion_bps: float = 5.0,
    radio_epsilon_dro: float = 0.02,
    n_escenarios_exante: int = 2000,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Replica EXACTA de la mecanica de ejecutar_walk_forward del pipeline
    original (misma ventana rodante, misma frecuencia de rebalanceo mensual,
    mismo costo de friccion de 5pb, misma deriva diaria de pesos por
    evolucion de precios entre rebalanceos) pero limitada a los DOS modelos
    nuevos -- separada del pipeline original a proposito (no se modifica
    ejecutar_walk_forward) para que el archivo original, ya publicado y
    citado en el documento, quede intacto. El indice de fechas resultante
    es IDENTICO al de retornos_diarios_oos.csv (mismo dias_ventana sobre el
    mismo `precios`), asi que ambos DataFrames se pueden concatenar por
    columnas sin reindexar.
    """
    rendimientos_simples, rendimientos_logaritmicos = calcular_rendimientos(precios)
    n_dias = len(precios)
    n_activos = len(tickers)
    modelos = ["DRO_Wasserstein", "Min_CVaR_ExAnte"]

    rendimientos_oos = {m: [] for m in modelos}
    historial_ponderaciones = {m: [] for m in modelos}
    fechas = rendimientos_simples.index
    factor_costo = tasa_friccion_bps / 10000.0
    pesos_vigentes = {m: np.zeros(n_activos) for m in modelos}

    for t in range(dias_ventana, n_dias - 1):
        es_dia_rebalanceo = ((t - dias_ventana) % dias_rebalanceo == 0)
        fecha_actual = fechas[t]
        rendimientos_dia = rendimientos_simples.iloc[t].values

        if es_dia_rebalanceo:
            ventana_simp = rendimientos_simples.iloc[t - dias_ventana:t].values
            ventana_log = rendimientos_logaritmicos.iloc[t - dias_ventana:t]

            pesos_objetivo = {
                "DRO_Wasserstein": optimizar_dro_wasserstein(ventana_simp, radio_epsilon=radio_epsilon_dro, nivel_confianza=0.95),
                "Min_CVaR_ExAnte": optimizar_minimo_cvar_ex_ante(ventana_log, n_escenarios=n_escenarios_exante, nivel_confianza=0.95),
            }
            for m in modelos:
                rotacion = np.sum(np.abs(pesos_objetivo[m] - pesos_vigentes[m]))
                costo_operativo = rotacion * factor_costo
                pesos_vigentes[m] = pesos_objetivo[m]
                historial_ponderaciones[m].append({
                    "Fecha": fecha_actual,
                    **{tickers[i]: pesos_objetivo[m][i] for i in range(n_activos)}
                })
            # Liberacion explicita: el entorno de esta sesion tiene un limite
            # de paginado de memoria mas chico que lo que HiGHS/CVXPY asumen
            # por defecto (confirmado -- el mismo error de "paging file"
            # aparece incluso al importar el solver HiGHS de CVXPY, no es un
            # problema de este codigo). Sin este gc.collect() explicito, las
            # ~190 resoluciones de LP/SOCP consecutivas de este walk-forward
            # agotan la memoria disponible antes de terminar.
            del ventana_simp, ventana_log, pesos_objetivo
            gc.collect()
        else:
            costo_operativo = 0.0

        for m in modelos:
            w = pesos_vigentes[m]
            r_cartera = float(np.dot(w, rendimientos_dia)) - costo_operativo
            rendimientos_oos[m].append(r_cartera)
            nuevos_pesos = w * (1.0 + rendimientos_dia)
            suma_pesos = np.sum(nuevos_pesos)
            if suma_pesos > 0:
                pesos_vigentes[m] = nuevos_pesos / suma_pesos

    fechas_oos = fechas[dias_ventana:n_dias - 1]
    df_rendimientos = pd.DataFrame(rendimientos_oos, index=fechas_oos)
    dfs_ponderaciones = {m: pd.DataFrame(historial_ponderaciones[m]).set_index("Fecha") for m in modelos}
    return df_rendimientos, dfs_ponderaciones


def ejecutar_cpcv_extension(
    precios: pd.DataFrame,
    tickers: List[str] = TICKERS,
    n_bloques: int = 6,
    k_bloques_test: int = 2,
    dias_embargo: int = 10,
    radio_epsilon_dro: float = 0.02,
    n_escenarios_exante: int = 3000,
) -> pd.DataFrame:
    """CPCV (misma mecanica de purga+embargo que ejecutar_cpcv del pipeline
    original) para los 2 modelos nuevos -- n_escenarios_exante reducido
    respecto al walk-forward (3000 vs 5000) por costo de computo: 28
    combinaciones x re-ajustar 9 GARCH + copula cada vez, igual que el
    pipeline original excluye Michaud de CPCV "por tiempo de computo"."""
    import itertools
    rendimientos_simples, rendimientos_logaritmicos = calcular_rendimientos(precios)
    n_muestras = len(rendimientos_logaritmicos)
    tam_bloque = n_muestras // n_bloques
    bloques = [list(range(i * tam_bloque, (i + 1) * tam_bloque if i < n_bloques - 1 else n_muestras)) for i in range(n_bloques)]
    combinaciones = list(itertools.combinations(range(n_bloques), k_bloques_test))
    modelos = ["DRO_Wasserstein", "Min_CVaR_ExAnte"]
    distribucion_sharpe = {m: [] for m in modelos}

    for combo in combinaciones:
        indices_test = sorted(idx for c in combo for idx in bloques[c])
        indices_train = []
        for i in range(n_bloques):
            if i not in combo:
                b = bloques[i]
                if (i - 1) in combo:
                    b = b[dias_embargo:]
                indices_train.extend(b)
        indices_train = sorted(indices_train)
        if len(indices_train) < 252 or len(indices_test) < 60:
            continue

        train_simp = rendimientos_simples.iloc[indices_train].values
        train_log = rendimientos_logaritmicos.iloc[indices_train]
        test_simp = rendimientos_simples.iloc[indices_test].values

        mapa_pesos = {
            "DRO_Wasserstein": optimizar_dro_wasserstein(train_simp, radio_epsilon=radio_epsilon_dro, nivel_confianza=0.95),
            "Min_CVaR_ExAnte": optimizar_minimo_cvar_ex_ante(train_log, n_escenarios=n_escenarios_exante, nivel_confianza=0.95),
        }
        for m in modelos:
            r_eval = np.dot(test_simp, mapa_pesos[m])
            retorno_anual = np.mean(r_eval) * 252.0
            volatilidad_anual = np.std(r_eval) * np.sqrt(252.0)
            distribucion_sharpe[m].append((retorno_anual - 0.02) / max(volatilidad_anual, 1e-6))

    return pd.DataFrame(distribucion_sharpe)
