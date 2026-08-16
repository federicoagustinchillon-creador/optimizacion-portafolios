"""
================================================================================
OPTIMIZACIÓN DE PORTAFOLIOS: MARKOWITZ, HERC, MODELIZACIÓN DE RIESGOS Y BACKTESTING
Evaluación Cuantitativa y Contrastación Empírica de Asignación de Activos (2007 - 2026)
Autor: Federico Agustín Chillón
================================================================================
"""

from __future__ import annotations
import os
import sys
import itertools
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as sco
import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns

# Fijar semilla pseudoaleatoria para reproducibilidad matemática
np.random.seed(42)

# ==============================================================================
# CONFIGURACIÓN DE DIRECTORIOS Y PARÁMETROS
# ==============================================================================
DIRECTORIO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_FIGURAS = os.path.join(DIRECTORIO_BASE, "04_Figuras")
DIR_DATOS = os.path.join(DIRECTORIO_BASE, "05_Datos_y_Resultados")

os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_DATOS, exist_ok=True)

# Universo de 9 activos líquidos multiactivo
TICKERS: List[str] = [
    "SPY",  # Renta Variable EE.UU. Gran Capitalización (S&P 500)
    "QQQ",  # Renta Variable Tecnológica (Nasdaq 100)
    "EEM",  # Renta Variable Mercados Emergentes
    "VNQ",  # Bienes Raíces e Infraestructura Inmobiliaria (REITs)
    "TLT",  # Bonos del Tesoro EE.UU. Largo Plazo (20+ Años)
    "IEF",  # Bonos del Tesoro EE.UU. Mediano Plazo (7-10 Años)
    "LQD",  # Bonos Corporativos Grado de Inversión
    "GLD",  # Oro Físico (Cobertura Inflacionaria y Geopolítica)
    "DBC",  # Materias Primas Diversificadas (Commodities)
]

NOMBRES_ACTIVOS: Dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq 100 (QQQ)",
    "EEM": "Emergentes (EEM)",
    "VNQ": "Bienes Raíces (VNQ)",
    "TLT": "Tesoro 20Y+ (TLT)",
    "IEF": "Tesoro 7-10Y (IEF)",
    "LQD": "Crédito Corporativo (LQD)",
    "GLD": "Oro (GLD)",
    "DBC": "Materias Primas (DBC)",
}

CLASES_ACTIVOS: Dict[str, str] = {
    "SPY": "Renta Variable",
    "QQQ": "Renta Variable",
    "EEM": "Renta Variable",
    "VNQ": "Bienes Raíces",
    "TLT": "Renta Fija Soberana",
    "IEF": "Renta Fija Soberana",
    "LQD": "Renta Fija Crédito",
    "GLD": "Metales Preciosos",
    "DBC": "Materias Primas",
}

# Paleta armónica sobria
PALETA_MODELOS: Dict[str, str] = {
    "Equiponderado": "#64748B",       # Gris Pizarra
    "Referencia_60_40": "#0F172A",    # Azul Noche Grafito
    "Markowitz": "#991B1B",           # Granate Carmesí
    "Michaud_Remuestreado": "#D97706",# Ámbar Tostado
    "Min_CVaR": "#2563EB",            # Azul Real
    "Min_CDaR": "#7C3AED",            # Violeta Imperial
    "HERC": "#0D9488",                # Verde Esmeralda Teal
}

ETIQUETAS_MODELOS: Dict[str, str] = {
    "Equiponderado": "1/N Equiponderado",
    "Referencia_60_40": "Referencia 60/40",
    "Markowitz": "Markowitz MVO",
    "Michaud_Remuestreado": "Markowitz Remuestreado",
    "Min_CVaR": "Mínimo CVaR (95%)",
    "Min_CDaR": "Mínimo CDaR (95%)",
    "HERC": "HERC (Ledoit-Wolf)",
}

# Configuración tipográfica y de estilo general en Georgia
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman"],
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelpad": 8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#D1D5DB",
    "figure.titlesize": 13.5,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#CBD5E1",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.edgecolor": "#64748B",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
})


# ==============================================================================
# 01. INGESTIÓN Y PREPROCESAMIENTO DE SERIES TEMPORALES
# ==============================================================================
def descargar_datos(
    tickers: List[str],
    fecha_inicio: str = "2007-01-01",
    fecha_fin: str = "2026-08-01",
    ruta_almacenamiento: str = os.path.join(DIR_DATOS, "precios_ajustados.csv")
) -> pd.DataFrame:
    """Descarga precios de cierre ajustados y aplica imputación temporal controlada."""
    if os.path.exists(ruta_almacenamiento):
        df_existente = pd.read_csv(ruta_almacenamiento, index_col=0, parse_dates=True)
        if all(t in df_existente.columns for t in tickers):
            return df_existente[tickers].dropna()

    print(f"Descargando datos históricos ({fecha_inicio} a {fecha_fin}) para {len(tickers)} activos...")
    descarga = yf.download(tickers, start=fecha_inicio, end=fecha_fin, progress=False)

    if isinstance(descarga.columns, pd.MultiIndex):
        if "Adj Close" in descarga.columns.levels[0]:
            precios = descarga["Adj Close"]
        else:
            precios = descarga["Close"]
    else:
        precios = descarga

    precios = precios.ffill(limit=3).bfill(limit=3).dropna()
    precios = precios[tickers]
    precios.to_csv(ruta_almacenamiento)
    return precios


def calcular_rendimientos(precios: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula rendimientos discretos porcentuales y rendimientos logarítmicos continuos."""
    rendimientos_simples = precios.pct_change().dropna()
    rendimientos_logaritmicos = np.log(precios / precios.shift(1)).dropna()
    return rendimientos_simples, rendimientos_logaritmicos


def ejecutar_diagnostico_econometrico(
    precios: pd.DataFrame,
    rendimientos_logaritmicos: pd.DataFrame,
    ruta_almacenamiento: str = os.path.join(DIR_DATOS, "tabla_diagnostico_econometrico.csv")
) -> pd.DataFrame:
    """
    Ejecuta una batería completa de contrastes econométricos y verificación de supuestos:
    1. Momentos empíricos (Media anualizada, Volatilidad anualizada, Asimetría, Curtosis de exceso).
    2. Test de Normalidad Jarque-Bera (JB stat y p-valor).
    3. Test de Autocorrelación Serial Ljung-Box Q(10) sobre rendimientos r_t (contraste de independencia / ruido blanco).
    4. Test de Efectos ARCH / Heterocedasticidad Ljung-Box Q(10) sobre r_t^2 (contraste de varianza condicional constante).
    5. Test de Estacionariedad Augmented Dickey-Fuller (ADF) en niveles de log-precios I(1) y retornos I(0).
    """
    resultados = []
    for col in rendimientos_logaritmicos.columns:
        r = rendimientos_logaritmicos[col].values
        p = precios[col].values
        n = len(r)
        
        # 1. Momentos
        media_anual = float(np.mean(r) * 252.0 * 100.0)
        vol_anual = float(np.std(r, ddof=1) * np.sqrt(252.0) * 100.0)
        asimetria = float(stats.skew(r))
        curtosis_exceso = float(stats.kurtosis(r))
        
        # 2. Normalidad Jarque-Bera
        jb_stat, jb_pval = stats.jarque_bera(r)
        
        # 3. Ljung-Box Q(10) en retornos r_t
        acf_r = [np.corrcoef(r[:-lag], r[lag:])[0, 1] for lag in range(1, 11)]
        q_stat_r = n * (n + 2) * sum((acf**2) / (n - k) for k, acf in enumerate(acf_r, 1))
        q_pval_r = float(1.0 - stats.chi2.cdf(q_stat_r, df=10))
        
        # 4. ARCH-LM Q(10) en residuos al cuadrado r_t^2
        r_dm = r - np.mean(r)
        r2 = r_dm**2
        acf_r2 = [np.corrcoef(r2[:-lag], r2[lag:])[0, 1] for lag in range(1, 11)]
        q_stat_arch = n * (n + 2) * sum((acf**2) / (n - k) for k, acf in enumerate(acf_r2, 1))
        q_pval_arch = float(1.0 - stats.chi2.cdf(q_stat_arch, df=10))
        
        # 5. Estacionariedad ADF
        adf_stat_p, adf_pval_p, _, _, _, _ = adfuller(np.log(p))
        adf_stat_r, adf_pval_r, _, _, _, _ = adfuller(r)
        
        resultados.append({
            "Activo": col,
            "Nombre Activo": NOMBRES_ACTIVOS.get(col, col),
            "Clase": CLASES_ACTIVOS.get(col, "Otros"),
            "Retorno Anual (%)": media_anual,
            "Volatilidad Anual (%)": vol_anual,
            "Asimetría (Skew)": asimetria,
            "Curtosis Exceso": curtosis_exceso,
            "JB Estadístico": float(jb_stat),
            "JB p-valor": float(jb_pval),
            "Q(10) Retornos (p-val)": q_pval_r,
            "ARCH Q(10) (p-val)": q_pval_arch,
            "ADF Log-Precios (p-val)": float(adf_pval_p),
            "ADF Retornos (p-val)": float(adf_pval_r),
        })
        
    df_diagnostico = pd.DataFrame(resultados).set_index("Activo")
    df_diagnostico.to_csv(ruta_almacenamiento)
    return df_diagnostico


# ==============================================================================
# 02. ESTIMACIÓN DE COVARIANZA Y ESTRUCTURA JERÁRQUICA
# ==============================================================================
def estimar_covarianza_y_correlacion(rendimientos: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Estima la matriz de covarianza mediante encogimiento de Ledoit-Wolf."""
    estimador_lw = LedoitWolf()
    estimador_lw.fit(rendimientos.values)
    matriz_cov = estimador_lw.covariance_ * 252.0
    desviaciones = np.sqrt(np.diag(matriz_cov))
    matriz_corr = matriz_cov / np.outer(desviaciones, desviaciones)
    matriz_corr = np.clip(matriz_corr, -1.0, 1.0)
    return matriz_cov, matriz_corr


def calcular_distancia_angular(matriz_corr: np.ndarray) -> np.ndarray:
    """Distancia métrica angular: d_ij = sqrt(0.5 * (1 - rho_ij))."""
    distancias = np.sqrt(np.clip(0.5 * (1.0 - matriz_corr), 0.0, 1.0))
    np.fill_diagonal(distancias, 0.0)
    return distancias


def obtener_agrupamiento_jerarquico(
    matriz_distancias: np.ndarray,
    metodo_enlace: str = "ward"
) -> Tuple[np.ndarray, List[int]]:
    """Construye el árbol de agrupamiento y el orden de quasi-diagonalización."""
    vector_condensado = scd.squareform(matriz_distancias, checks=False)
    arbol_enlace = sch.linkage(vector_condensado, method=metodo_enlace)
    orden_cuasidiagonal = sch.leaves_list(arbol_enlace).tolist()
    return arbol_enlace, orden_cuasidiagonal


# ==============================================================================
# 03. MOTORES NATIVOS DE OPTIMIZACIÓN (SciPy / NumPy)
# ==============================================================================
def optimizar_equiponderado(n_activos: int) -> np.ndarray:
    """Asignación ingenua 1/N."""
    return np.ones(n_activos) / float(n_activos)


def optimizar_referencia_60_40(tickers: List[str]) -> np.ndarray:
    """Cartera clásica de referencia (60% SPY / 40% IEF)."""
    pesos = np.zeros(len(tickers))
    indice_spy = tickers.index("SPY") if "SPY" in tickers else 0
    indice_ief = tickers.index("IEF") if "IEF" in tickers else 1
    pesos[indice_spy] = 0.60
    pesos[indice_ief] = 0.40
    return pesos


def optimizar_markowitz(
    vector_medias: np.ndarray,
    matriz_cov: np.ndarray,
    coeficiente_aversion_riesgo: float = 2.5
) -> np.ndarray:
    """
    Optimización de media-varianza resuelta mediante programación cuadrática SLSQP:
    min 0.5 * gamma * w' * Sigma * w - mu' * w  sujeto a sum(w) = 1, w >= 0.
    """
    n_activos = len(vector_medias)
    cov_regularizada = matriz_cov + 1e-6 * np.eye(n_activos)
    
    def funcion_objetivo(w: np.ndarray) -> float:
        return 0.5 * coeficiente_aversion_riesgo * float(w @ cov_regularizada @ w) - float(vector_medias @ w)
        
    def gradiente_objetivo(w: np.ndarray) -> np.ndarray:
        return coeficiente_aversion_riesgo * (cov_regularizada @ w) - vector_medias
        
    pesos_iniciales = np.ones(n_activos) / float(n_activos)
    limites = [(0.0, 1.0) for _ in range(n_activos)]
    restricciones = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(n_activos)})
    
    solucion = sco.minimize(
        funcion_objetivo,
        pesos_iniciales,
        jac=gradiente_objetivo,
        method='SLSQP',
        bounds=limites,
        constraints=restricciones,
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    if solucion.success and solucion.x is not None:
        pesos_optimos = np.clip(solucion.x, 0.0, 1.0)
        return pesos_optimos / np.sum(pesos_optimos)
        
    varianzas_inversas = 1.0 / np.diag(matriz_cov)
    return varianzas_inversas / np.sum(varianzas_inversas)


def optimizar_michaud_remuestreado(
    matriz_rendimientos: np.ndarray,
    n_simulaciones: int = 25
) -> np.ndarray:
    """Optimización de Markowitz Remuestreado mediante remuestreo bootstrap no paramétrico."""
    n_filas, n_activos = matriz_rendimientos.shape
    pesos_acumulados = np.zeros((n_simulaciones, n_activos))
    
    for i in range(n_simulaciones):
        indices_remuestreo = np.random.choice(n_filas, size=n_filas, replace=True)
        muestra = matriz_rendimientos[indices_remuestreo, :]
        mu_muestra = np.mean(muestra, axis=0) * 252.0
        lw_muestra = LedoitWolf().fit(muestra)
        cov_muestra = lw_muestra.covariance_ * 252.0
        pesos_acumulados[i, :] = optimizar_markowitz(mu_muestra, cov_muestra)
        
    pesos_promedio = np.mean(pesos_acumulados, axis=0)
    return pesos_promedio / np.sum(pesos_promedio)


def optimizar_minimo_cvar(
    matriz_rendimientos: np.ndarray,
    nivel_confianza: float = 0.95
) -> np.ndarray:
    """
    Minimización de Valor en Riesgo Condicional (Rockafellar & Uryasev, 2000)
    resuelta mediante programación lineal HiGHS con matrices dispersas (scipy.sparse).
    """
    n_filas, n_activos = matriz_rendimientos.shape
    coeficientes_costo = np.zeros(n_activos + 1 + n_filas)
    coeficientes_costo[n_activos] = 1.0
    coeficientes_costo[n_activos + 1:] = 1.0 / ((1.0 - nivel_confianza) * n_filas)
    
    A_desigualdad = sp.hstack([
        sp.csr_matrix(-matriz_rendimientos),
        sp.csr_matrix(-np.ones((n_filas, 1))),
        -sp.eye(n_filas, format='csr')
    ], format='csr')
    b_desigualdad = np.zeros(n_filas)
    
    A_igualdad = sp.hstack([
        sp.csr_matrix(np.ones((1, n_activos))),
        sp.csr_matrix(np.zeros((1, 1 + n_filas)))
    ], format='csr')
    b_igualdad = np.array([1.0])
    
    limites_variables = [(0.0, 1.0)] * n_activos + [(None, None)] + [(0.0, None)] * n_filas
    
    resultado = sco.linprog(
        c=coeficientes_costo,
        A_ub=A_desigualdad,
        b_ub=b_desigualdad,
        A_eq=A_igualdad,
        b_eq=b_igualdad,
        bounds=limites_variables,
        method='highs'
    )
    if resultado.success and resultado.x is not None:
        pesos_optimos = np.clip(resultado.x[:n_activos], 0.0, 1.0)
        suma_pesos = float(np.sum(pesos_optimos))
        if suma_pesos > 1e-6:
            return pesos_optimos / suma_pesos
        
    volatilidades_inversas = 1.0 / np.maximum(np.std(matriz_rendimientos, axis=0), 1e-6)
    return volatilidades_inversas / np.sum(volatilidades_inversas)


def optimizar_minimo_cdar(
    matriz_rendimientos: np.ndarray,
    nivel_confianza: float = 0.95
) -> np.ndarray:
    """
    Minimización de Retroceso Condicional Acumulado (Chekhlov et al., 2005)
    resuelta mediante programación lineal HiGHS con matrices dispersas (scipy.sparse).
    """
    if len(matriz_rendimientos) > 252:
        matriz_rendimientos = matriz_rendimientos[-252:]
        
    n_filas, n_activos = matriz_rendimientos.shape
    coeficientes_costo = np.zeros(n_activos + 1 + 2 * n_filas)
    coeficientes_costo[n_activos] = 1.0
    coeficientes_costo[n_activos + 1 : n_activos + 1 + n_filas] = 1.0 / ((1.0 - nivel_confianza) * n_filas)
    
    rendimientos_acumulados = np.cumsum(matriz_rendimientos, axis=0)
    
    bloque_a1 = sp.hstack([
        sp.csr_matrix(-rendimientos_acumulados),
        sp.csr_matrix(-np.ones((n_filas, 1))),
        -sp.eye(n_filas, format='csr'),
        sp.eye(n_filas, format='csr')
    ], format='csr')
    bloque_b1 = np.zeros(n_filas)
    
    bloque_a2 = sp.hstack([
        sp.csr_matrix(rendimientos_acumulados),
        sp.csr_matrix(np.zeros((n_filas, 1))),
        sp.csr_matrix((n_filas, n_filas)),
        -sp.eye(n_filas, format='csr')
    ], format='csr')
    bloque_b2 = np.zeros(n_filas)
    
    matriz_diferencias_u = sp.dok_matrix((n_filas - 1, n_filas))
    for s in range(n_filas - 1):
        matriz_diferencias_u[s, s] = 1.0
        matriz_diferencias_u[s, s + 1] = -1.0
    bloque_a3 = sp.hstack([
        sp.csr_matrix((n_filas - 1, n_activos + 1 + n_filas)),
        matriz_diferencias_u.tocsr()
    ], format='csr')
    bloque_b3 = np.zeros(n_filas - 1)
    
    A_desigualdad = sp.vstack([bloque_a1, bloque_a2, bloque_a3], format='csr')
    b_desigualdad = np.concatenate([bloque_b1, bloque_b2, bloque_b3])
    
    A_igualdad = sp.hstack([
        sp.csr_matrix(np.ones((1, n_activos))),
        sp.csr_matrix(np.zeros((1, 1 + 2 * n_filas)))
    ], format='csr')
    b_igualdad = np.array([1.0])
    
    limites_variables = [(0.0, 1.0)] * n_activos + [(None, None)] + [(0.0, None)] * (2 * n_filas)
    
    resultado = sco.linprog(
        c=coeficientes_costo,
        A_ub=A_desigualdad,
        b_ub=b_desigualdad,
        A_eq=A_igualdad,
        b_eq=b_igualdad,
        bounds=limites_variables,
        method='highs'
    )
    if resultado.success and resultado.x is not None:
        pesos_optimos = np.clip(resultado.x[:n_activos], 0.0, 1.0)
        suma_pesos = float(np.sum(pesos_optimos))
        if suma_pesos > 1e-6:
            return pesos_optimos / suma_pesos
        
    volatilidades_inversas = 1.0 / np.maximum(np.std(matriz_rendimientos, axis=0), 1e-6)
    return volatilidades_inversas / np.sum(volatilidades_inversas)


def optimizar_herc(
    matriz_cov: np.ndarray,
    arbol_enlace: np.ndarray,
    orden_cuasidiagonal: List[int]
) -> np.ndarray:
    """Asignación Jerárquica de Contribución de Riesgo Equitativo (HERC - Raffinot, 2018)."""
    n_activos = len(orden_cuasidiagonal)
    pesos_serie = pd.Series(1.0, index=orden_cuasidiagonal)
    conglomerados = [orden_cuasidiagonal]
    
    while len(conglomerados) > 0:
        conglomerados = [
            c[inicio:fin]
            for c in conglomerados
            for inicio, fin in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        
        for i in range(0, len(conglomerados), 2):
            conglomerado_1 = conglomerados[i]
            conglomerado_2 = conglomerados[i + 1]
            
            cov_1 = matriz_cov[np.ix_(conglomerado_1, conglomerado_1)]
            cov_2 = matriz_cov[np.ix_(conglomerado_2, conglomerado_2)]
            
            inv_diag_1 = 1.0 / np.maximum(np.diag(cov_1), 1e-8)
            w1 = inv_diag_1 / np.sum(inv_diag_1)
            
            inv_diag_2 = 1.0 / np.maximum(np.diag(cov_2), 1e-8)
            w2 = inv_diag_2 / np.sum(inv_diag_2)
            
            var_1 = float(np.dot(w1, np.dot(cov_1, w1)))
            var_2 = float(np.dot(w2, np.dot(cov_2, w2)))
            
            vol_1 = np.sqrt(max(var_1, 1e-8))
            vol_2 = np.sqrt(max(var_2, 1e-8))
            
            peso_conglomerado_1 = (1.0 / vol_1) / ((1.0 / vol_1) + (1.0 / vol_2))
            peso_conglomerado_2 = 1.0 - peso_conglomerado_1
            
            pesos_serie.loc[conglomerado_1] *= peso_conglomerado_1
            pesos_serie.loc[conglomerado_2] *= peso_conglomerado_2
            
    pesos_finales = np.zeros(n_activos)
    for pos_original in orden_cuasidiagonal:
        pesos_finales[pos_original] = pesos_serie.loc[pos_original]
        
    return pesos_finales / np.sum(pesos_finales)


# ==============================================================================
# 04. TEORÍA DE VALORES EXTREMOS (EVT / PARETO GENERALIZADA)
# ==============================================================================
def ajustar_modelo_evt_gpd(
    serie_rendimientos: np.ndarray,
    cuantil_umbral: float = 0.90,
    usar_filtrado_garch: bool = True
) -> Dict[str, Any]:
    """
    Teoría de Valores Extremos (EVT) mediante el Enfoque en Dos Etapas de McNeil & Frey (2000):
    1. Ajuste AR(1)-GARCH(1,1) para aislar la volatilidad condicional sigma_t y media condicional mu_t.
    2. Extracción de residuos estandarizados z_t = (r_t - mu_t) / sigma_t, purgados de efectos ARCH (i.i.d.).
    3. Verificación de la hipótesis nula de no-autocorrelación en z_t^2 (ARCH-LM Q(10) p-valor > 0.05).
    4. Calibración de la Ley de Pareto Generalizada (GPD) sobre los excesos de pérdida estandarizados (-z_t > u).
    5. Deducción analítica de VaR_99% y CVaR_99% condicionales e incondicionales.
    """
    r_pct = np.array(serie_rendimientos)
    if np.max(np.abs(r_pct)) <= 1.0:
        r_pct = r_pct * 100.0  # Convertir a porcentaje si viene en fracciones
        
    n_obs = len(r_pct)
    p_val_arch_residuos = 1.0
    sigma_promedio = float(np.std(r_pct, ddof=1))
    mu_promedio = float(np.mean(r_pct))
    
    std_resid = None
    if usar_filtrado_garch and n_obs >= 252:
        try:
            modelo_garch = arch_model(r_pct, mean='AR', lags=1, vol='Garch', p=1, q=1, dist='normal')
            res_garch = modelo_garch.fit(disp='off')
            
            std_resid_raw = np.asarray(res_garch.std_resid)
            std_resid = std_resid_raw[~np.isnan(std_resid_raw)]
            
            sigma_raw = np.asarray(res_garch.conditional_volatility)
            sigma_t = sigma_raw[~np.isnan(sigma_raw)]
            
            resid_raw = np.asarray(res_garch.resid)
            resid_clean = resid_raw[~np.isnan(resid_raw)]
            mu_t = r_pct[-len(std_resid):] - resid_clean
            
            # Contraste de heterocedasticidad Ljung-Box sobre z_t^2
            e2 = std_resid ** 2
            n_z = len(e2)
            e2_dm = e2 - np.mean(e2)
            acf_z2 = [np.corrcoef(e2_dm[:-lag], e2_dm[lag:])[0, 1] for lag in range(1, 11)]
            q_stat = n_z * (n_z + 2) * sum((acf**2) / (n_z - k) for k, acf in enumerate(acf_z2, 1))
            p_val_arch_residuos = float(1.0 - stats.chi2.cdf(q_stat, df=10))
            
            perdidas = -std_resid
            factor_escala = float(np.mean(sigma_t))
            desplazamiento = float(np.mean(mu_t))
        except Exception as e:
            sigma_promedio = float(np.std(r_pct, ddof=1))
            mu_promedio = float(np.mean(r_pct))
            std_resid = (r_pct - mu_promedio) / max(sigma_promedio, 1e-6)
            perdidas = -std_resid
            factor_escala = sigma_promedio
            desplazamiento = mu_promedio
            p_val_arch_residuos = 0.0
    else:
        perdidas = -r_pct
        factor_escala = 1.0
        desplazamiento = 0.0

    umbral_u = float(np.percentile(perdidas, cuantil_umbral * 100.0))
    excesos = perdidas[perdidas > umbral_u] - umbral_u
    n_excesos = len(excesos)
    
    if n_excesos < 10:
        parametro_xi = 0.0
        parametro_beta = float(np.std(perdidas))
        z_var99 = float(np.percentile(perdidas, 99.0))
        z_cvar99 = float(np.mean(perdidas[perdidas >= z_var99]))
    else:
        parametro_xi, _, parametro_beta = stats.genpareto.fit(excesos, floc=0.0)
        probabilidad_cola = (n_obs / float(n_excesos)) * (1.0 - 0.99)
        
        if parametro_xi != 0.0 and probabilidad_cola > 0:
            z_var99 = umbral_u + (parametro_beta / parametro_xi) * ((probabilidad_cola) ** (-parametro_xi) - 1.0)
            z_cvar99 = (z_var99 + parametro_beta - parametro_xi * umbral_u) / (1.0 - parametro_xi)
        else:
            z_var99 = umbral_u - parametro_beta * np.log(probabilidad_cola)
            z_cvar99 = z_var99 + parametro_beta

    if usar_filtrado_garch:
        var_99_evt = float(desplazamiento - z_var99 * factor_escala)
        cvar_99_evt = float(desplazamiento - z_cvar99 * factor_escala)
    else:
        var_99_evt = float(-z_var99)
        cvar_99_evt = float(-z_cvar99)

    return {
        "umbral_u": umbral_u,
        "parametro_xi": float(parametro_xi),
        "parametro_beta": float(parametro_beta),
        "excesos": excesos,
        "n_excesos": n_excesos,
        "var_99_evt": var_99_evt,
        "cvar_99_evt": cvar_99_evt,
        "p_val_arch": p_val_arch_residuos,
        "residuos_estandarizados": std_resid if std_resid is not None else -perdidas,
        "factor_escala": factor_escala,
        "desplazamiento": desplazamiento,
    }


# ==============================================================================
# 05. SIGNIFICANCIA ESTADÍSTICA (PSR Y DSR DE LÓPEZ DE PRADO)
# ==============================================================================
def calcular_ratio_sharpe_probabilistico(
    sharpe_anual: float,
    asimetria: float,
    curtosis_exceso: float,
    n_observaciones: int,
    sharpe_referencia_anual: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio (PSR) con corrección por asimetría y curtosis."""
    sr_diario = sharpe_anual / np.sqrt(252.0)
    sr_referencia_diario = sharpe_referencia_anual / np.sqrt(252.0)
    gamma_3 = asimetria
    gamma_4 = curtosis_exceso + 3.0
    
    denominador = np.sqrt(max(1.0 - gamma_3 * sr_diario + ((gamma_4 - 1.0) / 4.0) * (sr_diario ** 2), 1e-8))
    estadistico_z = ((sr_diario - sr_referencia_diario) * np.sqrt(n_observaciones - 1)) / denominador
    return float(stats.norm.cdf(estadistico_z))


def calcular_ratio_sharpe_deflactado(
    sharpe_anual: float,
    asimetria: float,
    curtosis_exceso: float,
    n_observaciones: int,
    n_modelos_evaluados: int = 7,
    varianza_sharpes: float = 0.04
) -> float:
    """Deflated Sharpe Ratio (DSR) corregido frente al sesgo de selección múltiple."""
    constante_euler = 0.57721566490153286
    if n_modelos_evaluados > 1:
        cuantil_1 = stats.norm.ppf(1.0 - 1.0 / n_modelos_evaluados)
        cuantil_2 = stats.norm.ppf(1.0 - 1.0 / (n_modelos_evaluados * np.e))
        sharpe_esperado_maximo = np.sqrt(varianza_sharpes) * ((1.0 - constante_euler) * cuantil_1 + constante_euler * cuantil_2)
    else:
        sharpe_esperado_maximo = 0.0
        
    return calcular_ratio_sharpe_probabilistico(
        sharpe_anual=sharpe_anual,
        asimetria=asimetria,
        curtosis_exceso=curtosis_exceso,
        n_observaciones=n_observaciones,
        sharpe_referencia_anual=sharpe_esperado_maximo
    )


# ==============================================================================
# 06. SIMULACIÓN WALK-FORWARD CON DERIVA Y COSTOS
# ==============================================================================
def ejecutar_walk_forward(
    precios: pd.DataFrame,
    tickers: List[str],
    dias_ventana: int = 756,      # 3 años de entrenamiento rodante
    dias_rebalanceo: int = 21,    # Rebalanceo mensual
    tasa_friccion_bps: float = 5.0 # Costo de transacción de 5 puntos básicos
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Simula asignación fuera de muestra con deriva diaria intra-mes y costos."""
    rendimientos_simples, rendimientos_logaritmicos = calcular_rendimientos(precios)
    n_dias = len(precios)
    n_activos = len(tickers)
    
    modelos = [
        "Equiponderado",
        "Referencia_60_40",
        "Markowitz",
        "Michaud_Remuestreado",
        "Min_CVaR",
        "Min_CDaR",
        "HERC"
    ]
    
    rendimientos_fuera_muestra = {m: [] for m in modelos}
    historial_ponderaciones = {m: [] for m in modelos}
    fechas_rebalanceo = []
    
    fechas = rendimientos_simples.index
    factor_costo = tasa_friccion_bps / 10000.0
    pesos_vigentes = {m: np.zeros(n_activos) for m in modelos}
    
    for t in range(dias_ventana, n_dias - 1):
        es_dia_rebalanceo = ((t - dias_ventana) % dias_rebalanceo == 0)
        fecha_actual = fechas[t]
        rendimientos_dia = rendimientos_simples.iloc[t].values
        
        if es_dia_rebalanceo:
            fechas_rebalanceo.append(fecha_actual)
            ventana_log = rendimientos_logaritmicos.iloc[t - dias_ventana : t]
            ventana_simp = rendimientos_simples.iloc[t - dias_ventana : t].values
            
            cov_estimada, corr_estimada = estimar_covarianza_y_correlacion(ventana_log)
            medias_anualizadas = ventana_log.mean().values * 252.0
            distancias = calcular_distancia_angular(corr_estimada)
            arbol_enlace, orden_cuasidiagonal = obtener_agrupamiento_jerarquico(distancias, metodo_enlace="ward")
            
            pesos_objetivo = {
                "Equiponderado": optimizar_equiponderado(n_activos),
                "Referencia_60_40": optimizar_referencia_60_40(tickers),
                "Markowitz": optimizar_markowitz(medias_anualizadas, cov_estimada),
                "Michaud_Remuestreado": optimizar_michaud_remuestreado(ventana_simp, n_simulaciones=25),
                "Min_CVaR": optimizar_minimo_cvar(ventana_simp, nivel_confianza=0.95),
                "Min_CDaR": optimizar_minimo_cdar(ventana_simp, nivel_confianza=0.95),
                "HERC": optimizar_herc(cov_estimada, arbol_enlace, orden_cuasidiagonal),
            }
            
            for m in modelos:
                rotacion = np.sum(np.abs(pesos_objetivo[m] - pesos_vigentes[m]))
                costo_operativo = rotacion * factor_costo
                pesos_vigentes[m] = pesos_objetivo[m]
                historial_ponderaciones[m].append({
                    "Fecha": fecha_actual,
                    **{tickers[i]: pesos_objetivo[m][i] for i in range(n_activos)}
                })
        else:
            costo_operativo = 0.0
            
        for m in modelos:
            w = pesos_vigentes[m]
            r_cartera = float(np.dot(w, rendimientos_dia)) - costo_operativo
            rendimientos_fuera_muestra[m].append(r_cartera)
            
            # Deriva de ponderaciones por evolución de precios
            nuevos_pesos = w * (1.0 + rendimientos_dia)
            suma_pesos = np.sum(nuevos_pesos)
            if suma_pesos > 0:
                pesos_vigentes[m] = nuevos_pesos / suma_pesos
                
    fechas_oos = fechas[dias_ventana : n_dias - 1]
    df_rendimientos = pd.DataFrame(rendimientos_fuera_muestra, index=fechas_oos)
    
    dfs_ponderaciones = {
        m: pd.DataFrame(historial_ponderaciones[m]).set_index("Fecha")
        for m in modelos
    }
    
    return df_rendimientos, pd.DataFrame(index=fechas_rebalanceo), dfs_ponderaciones


# ==============================================================================
# 07. VALIDACIÓN CRUZADA COMBINATORIA CON EMBARGO (CPCV)
# ==============================================================================
def ejecutar_cpcv(
    precios: pd.DataFrame,
    tickers: List[str],
    n_bloques: int = 6,
    k_bloques_test: int = 2,
    dias_embargo: int = 10
) -> pd.DataFrame:
    """Validación cruzada combinatoria con purga y embargo temporal."""
    rendimientos_simples, rendimientos_logaritmicos = calcular_rendimientos(precios)
    n_muestras = len(rendimientos_logaritmicos)
    tam_bloque = n_muestras // n_bloques
    bloques = [list(range(i * tam_bloque, (i + 1) * tam_bloque if i < n_bloques - 1 else n_muestras)) for i in range(n_bloques)]
    
    combinaciones = list(itertools.combinations(range(n_bloques), k_bloques_test))
    modelos = ["Equiponderado", "Referencia_60_40", "Markowitz", "Min_CVaR", "Min_CDaR", "HERC"]
    distribucion_sharpe = {m: [] for m in modelos}
    
    for combo in combinaciones:
        indices_test = []
        for c in combo:
            indices_test.extend(bloques[c])
        indices_test = sorted(indices_test)
        
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
            
        train_log = rendimientos_logaritmicos.iloc[indices_train]
        train_simp = rendimientos_simples.iloc[indices_train].values
        test_simp = rendimientos_simples.iloc[indices_test].values
        
        cov_train, corr_train = estimar_covarianza_y_correlacion(train_log)
        medias_train = train_log.mean().values * 252.0
        distancias = calcular_distancia_angular(corr_train)
        arbol_enlace, orden_cuasidiagonal = obtener_agrupamiento_jerarquico(distancias)
        
        mapa_pesos = {
            "Equiponderado": optimizar_equiponderado(len(tickers)),
            "Referencia_60_40": optimizar_referencia_60_40(tickers),
            "Markowitz": optimizar_markowitz(medias_train, cov_train),
            "Min_CVaR": optimizar_minimo_cvar(train_simp),
            "Min_CDaR": optimizar_minimo_cdar(train_simp),
            "HERC": optimizar_herc(cov_train, arbol_enlace, orden_cuasidiagonal)
        }
        
        for m in modelos:
            w = mapa_pesos[m]
            r_eval = np.dot(test_simp, w)
            retorno_anual = np.mean(r_eval) * 252.0
            volatilidad_anual = np.std(r_eval) * np.sqrt(252.0)
            ratio_sharpe = (retorno_anual - 0.02) / max(volatilidad_anual, 1e-6)
            distribucion_sharpe[m].append(ratio_sharpe)
            
    return pd.DataFrame(distribucion_sharpe)


# ==============================================================================
# 08. TABLA DE MÉTRICAS E INDICADORES ESTADÍSTICOS
# ==============================================================================
def calcular_tabla_metricas(
    df_rendimientos: pd.DataFrame,
    dfs_ponderaciones: Dict[str, pd.DataFrame],
    tasa_libre_riesgo: float = 0.02
) -> pd.DataFrame:
    """Calcula el conjunto completo de indicadores de rendimiento, riesgo, colas y significancia."""
    metricas = {}
    rf_diaria = (1.0 + tasa_libre_riesgo) ** (1.0 / 252.0) - 1.0
    sharpes_registrados = []
    
    for col in df_rendimientos.columns:
        r = df_rendimientos[col].values
        n_dias = len(r)
        
        retorno_acumulado = np.prod(1.0 + r) - 1.0
        cagr = (1.0 + retorno_acumulado) ** (252.0 / n_dias) - 1.0
        
        volatilidad_anual = np.std(r, ddof=1) * np.sqrt(252.0)
        desvios_negativos = r[r < rf_diaria] - rf_diaria
        desviacion_bajista = np.sqrt(np.mean(desvios_negativos ** 2)) * np.sqrt(252.0) if len(desvios_negativos) > 0 else 1e-6
        
        ratio_sharpe = (cagr - tasa_libre_riesgo) / max(volatilidad_anual, 1e-6)
        ratio_sortino = (cagr - tasa_libre_riesgo) / max(desviacion_bajista, 1e-6)
        sharpes_registrados.append(ratio_sharpe)
        
        serie_patrimonio = np.cumprod(1.0 + r)
        picos_historicos = np.maximum.accumulate(serie_patrimonio)
        serie_drawdowns = (serie_patrimonio - picos_historicos) / picos_historicos
        retroceso_maximo = float(np.min(serie_drawdowns))
        ratio_calmar = cagr / abs(retroceso_maximo) if abs(retroceso_maximo) > 0 else 0.0
        
        peores_drawdowns = serie_drawdowns[serie_drawdowns <= np.percentile(serie_drawdowns, 5.0)]
        cdar_95 = float(np.mean(peores_drawdowns)) if len(peores_drawdowns) > 0 else retroceso_maximo
        
        ganancias = r[r > 0]
        perdidas = r[r < 0]
        ratio_ganancia_perdida = np.sum(ganancias) / abs(np.sum(perdidas)) if len(perdidas) > 0 and np.sum(perdidas) != 0 else 0.0
        ratio_omega = np.sum(np.maximum(r - rf_diaria, 0.0)) / max(np.sum(np.maximum(rf_diaria - r, 0.0)), 1e-8)
        
        coef_asimetria = float(stats.skew(r))
        curtosis_exceso = float(stats.kurtosis(r))
        var_95_empirico = float(np.percentile(r, 5.0))
        cvar_95_empirico = float(np.mean(r[r <= var_95_empirico]))
        
        resultado_evt = ajustar_modelo_evt_gpd(r, cuantil_umbral=0.90, usar_filtrado_garch=True)
        
        psr = calcular_ratio_sharpe_probabilistico(ratio_sharpe, coef_asimetria, curtosis_exceso, n_dias, sharpe_referencia_anual=0.0)
        dsr = calcular_ratio_sharpe_deflactado(
            ratio_sharpe, coef_asimetria, curtosis_exceso, n_dias,
            n_modelos_evaluados=len(df_rendimientos.columns),
            varianza_sharpes=float(np.var(sharpes_registrados)) if len(sharpes_registrados) > 1 else 0.02
        )
        
        df_w = dfs_ponderaciones.get(col, pd.DataFrame())
        if not df_w.empty:
            diferencias_pesos = df_w.diff().dropna().abs().values
            rotacion_anual = float(np.mean(np.sum(diferencias_pesos, axis=1)) * 12.0)
            indice_hhi = float(np.mean(np.sum(df_w.values ** 2, axis=1)))
            n_efectivo = 1.0 / max(indice_hhi, 1e-6)
        else:
            rotacion_anual = 0.0
            indice_hhi = 1.0 / len(TICKERS)
            n_efectivo = float(len(TICKERS))
            
        metricas[col] = {
            "CAGR (%)": cagr * 100.0,
            "Retorno Acumulado (%)": retorno_acumulado * 100.0,
            "Volatilidad Anual (%)": volatilidad_anual * 100.0,
            "Desviación a la Baja (%)": desviacion_bajista * 100.0,
            "Ratio de Sharpe": ratio_sharpe,
            "Ratio de Sortino": ratio_sortino,
            "Ratio de Calmar": ratio_calmar,
            "Ratio de Omega": ratio_omega,
            "Ratio Ganancia/Pérdida": ratio_ganancia_perdida,
            "PSR (vs 0) (%)": psr * 100.0,
            "DSR (%)": dsr * 100.0,
            "Retroceso Máximo / Max DD (%)": retroceso_maximo * 100.0,
            "CDaR 95% (%)": cdar_95 * 100.0,
            "VaR 95% Diario (%)": var_95_empirico * 100.0,
            "CVaR 95% Diario (%)": cvar_95_empirico * 100.0,
            "VaR 99% EVT (%)": resultado_evt["var_99_evt"],
            "CVaR 99% EVT (%)": resultado_evt["cvar_99_evt"],
            "Coeficiente de Asimetría": coef_asimetria,
            "Curtosis de Exceso": curtosis_exceso,
            "Rotación Anualizada (%)": rotacion_anual * 100.0,
            "Nº Efectivo de Activos (N_eff)": n_efectivo,
        }
        
    return pd.DataFrame(metricas)


# ==============================================================================
# 09. GENERACIÓN DE FIGURAS
# ==============================================================================
def generar_figura_1_dendrograma_y_correlaciones(
    matriz_corr: np.ndarray,
    arbol_enlace: np.ndarray,
    orden_cuasidiagonal: List[int],
    tickers: List[str]
) -> None:
    """Figura 1: Dendrograma jerárquico y matriz de correlación reordenada."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    
    clases_abrev = {
        "SPY": "RV EE.UU.", "QQQ": "Nasdaq", "EEM": "RV Emerg.",
        "VNQ": "Inmuebles", "TLT": "Tesoro 20Y+", "IEF": "Tesoro 7-10Y",
        "LQD": "Corp. IG", "GLD": "Oro", "DBC": "Commodities"
    }
    
    # Subplot A: Dendrograma
    ax1 = axes[0]
    sch.dendrogram(
        arbol_enlace,
        labels=[f"{t}\n({clases_abrev.get(t, '')})" for t in tickers],
        ax=ax1,
        orientation="top",
        color_threshold=arbol_enlace[-2, 2] if len(arbol_enlace) > 2 else None,
        above_threshold_color="#0F2942"
    )
    ax1.set_title("A. Agrupamiento Jerárquico de Activos (Enlace de Ward)", pad=12, fontweight="bold")
    ax1.set_ylabel("Distancia Métrica Angular  $d(i, j) = \\sqrt{0.5(1 - \\rho_{ij})}$")
    ax1.tick_params(axis="x", rotation=0, labelsize=8.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Subplot B: Matriz reordenada
    ax2 = axes[1]
    etiquetas_ordenadas = [tickers[i] for i in orden_cuasidiagonal]
    corr_ordenada = matriz_corr[np.ix_(orden_cuasidiagonal, orden_cuasidiagonal)]
    
    mapa = sns.heatmap(
        corr_ordenada,
        xticklabels=etiquetas_ordenadas,
        yticklabels=etiquetas_ordenadas,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.0, "weight": "normal", "family": "serif"},
        cbar_kws={"label": "Coeficiente de Correlación ($\\rho$)", "shrink": 0.85},
        ax=ax2,
        linewidths=0.8,
        linecolor="#FFFFFF",
        vmin=-0.4,
        vmax=1.0
    )
    ax2.set_title("B. Matriz de Correlación Quasi-Diagonalizada", pad=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=0)
    ax2.tick_params(axis="y", rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig01_dendrograma_correlaciones.png"), dpi=300)
    plt.close()


def generar_figura_2_frontera_eficiente(
    vector_medias: np.ndarray,
    matriz_cov: np.ndarray,
    pesos_dict: Dict[str, np.ndarray],
    tickers: List[str]
) -> None:
    """
    Figura 2: Frontera eficiente media-varianza analítica, posicionamiento de activos y modelos (Panel A)
    y cono de incertidumbre muestral bajo remuestreo bootstrap de Michaud (Panel B).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6))
    n_activos = len(vector_medias)
    
    # ---------------- PANEL A: Frontera Analítica, Activos y Modelos ----------------
    np.random.seed(42)
    n_simulaciones = 6000
    
    # Simulación de carteras variadas en concentración
    pesos_sim_list = [
        np.random.dirichlet(np.ones(n_activos) * alpha, size=n_simulaciones // 4)
        for alpha in [0.25, 0.60, 1.20, 2.50]
    ]
    pesos_simulados = np.vstack(pesos_sim_list)
    retornos_sim = (pesos_simulados @ vector_medias) * 100.0
    volatilidades_sim = np.sqrt(np.sum((pesos_simulados @ matriz_cov) * pesos_simulados, axis=1)) * 100.0
    sharpes_sim = (retornos_sim - 2.0) / np.maximum(volatilidades_sim, 1e-6)
    
    disp = ax1.scatter(
        volatilidades_sim, retornos_sim,
        c=sharpes_sim, cmap="mako", alpha=0.30, s=12, edgecolors="none", zorder=1
    )
    cbar = plt.colorbar(disp, ax=ax1, pad=0.02, shrink=0.88)
    cbar.set_label("Ratio de Sharpe ($R_f = 2.0\\%$)", fontsize=9.0)
    
    # 1. Activos individuales (los 9 activos del universo completo)
    vols_activos = np.sqrt(np.diag(matriz_cov)) * 100.0
    rets_activos = vector_medias * 100.0
    ax1.scatter(
        vols_activos, rets_activos,
        color="#0F2942", marker="o", s=85, edgecolors="#FFFFFF", linewidths=1.2, zorder=6,
        label="Activos Individuales ($N=9$)"
    )
    
    # Desplazamientos específicos para cada ticker para evitar cualquier solapamiento
    desplazamientos = {
        "SPY": (6, 5), "QQQ": (-30, 5), "EEM": (6, -5),
        "VNQ": (6, 5), "GLD": (-26, 5), "TLT": (6, -5),
        "IEF": (6, 4), "LQD": (6, -5), "DBC": (6, -5)
    }
    for i, t in enumerate(tickers):
        dx, dy = desplazamientos.get(t, (6, 3))
        ax1.annotate(
            t,
            xy=(vols_activos[i], rets_activos[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="#0F2942",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.90, lw=0.7),
            zorder=7
        )
        
    # 2. Frontera eficiente analítica (Long-Only)
    retornos_obj = np.linspace(float(np.min(vector_medias)), float(np.max(vector_medias)) * 0.98, 40)
    f_vol_ana, f_ret_ana = [], []
    for r_obj in retornos_obj:
        w0 = np.ones(n_activos) / n_activos
        res = sco.minimize(
            lambda w: float(w @ matriz_cov @ w),
            w0, method='SLSQP',
            bounds=[(0.0, 1.0) for _ in range(n_activos)],
            constraints=(
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'ineq', 'fun': lambda w: float(vector_medias @ w) - r_obj}
            )
        )
        if res.success:
            f_vol_ana.append(np.sqrt(float(res.x @ matriz_cov @ res.x)) * 100.0)
            f_ret_ana.append(r_obj * 100.0)
            
    if len(f_vol_ana) > 0:
        ax1.plot(f_vol_ana, f_ret_ana, color="#991B1B", lw=2.4, linestyle="-", label="Frontera Eficiente Markowitz", zorder=4)
        
    # 3. Modelos y estrategias evaluadas
    simbolos = {
        "Equiponderado": ("s", PALETA_MODELOS["Equiponderado"]),
        "Referencia_60_40": ("D", PALETA_MODELOS["Referencia_60_40"]),
        "Markowitz": ("X", PALETA_MODELOS["Markowitz"]),
        "Michaud_Remuestreado": ("p", PALETA_MODELOS["Michaud_Remuestreado"]),
        "Min_CVaR": ("^", PALETA_MODELOS["Min_CVaR"]),
        "Min_CDaR": ("v", PALETA_MODELOS["Min_CDaR"]),
        "HERC": ("o", PALETA_MODELOS["HERC"]),
    }
    
    for m, (simb, col) in simbolos.items():
        if m in pesos_dict:
            w_m = pesos_dict[m]
            r_m = float(np.dot(w_m, vector_medias)) * 100.0
            v_m = np.sqrt(float(np.dot(w_m, np.dot(matriz_cov, w_m)))) * 100.0
            tam = 140 if m == "HERC" else 115
            borde = 1.6 if m == "HERC" else 1.0
            ax1.scatter(
                v_m, r_m, color=col, marker=simb, s=tam, zorder=8,
                edgecolors="#FFFFFF", linewidths=borde, label=ETIQUETAS_MODELOS[m]
            )
            
    ax1.set_title("A. Espacio Media-Varianza, Activos y Modelos Evaluados", pad=11, fontweight="bold")
    ax1.set_xlabel("Volatilidad Anualizada (%)", fontsize=9.5)
    ax1.set_ylabel("Rendimiento Anualizado Esperado (%)", fontsize=9.5)
    ax1.set_xlim(3.5, 33.0)
    ax1.set_ylim(0.5, 17.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left", fontsize=7.8, ncol=1)
    
    # ---------------- PANEL B: Remuestreo Bootstrap de Michaud ----------------
    t_muestras = 756
    ret_sint = np.random.multivariate_normal(vector_medias / 252.0, matriz_cov / 252.0, size=t_muestras)
    
    grid_ret = np.linspace(4.0, 15.0, 35)
    curvas_vols_boot = []
    
    for b in range(35):
        idx_b = np.random.choice(t_muestras, size=t_muestras, replace=True)
        mb = np.mean(ret_sint[idx_b], axis=0) * 252.0
        cb = np.cov(ret_sint[idx_b], rowvar=False) * 252.0
        
        vols_b = []
        for ro in grid_ret / 100.0:
            sol_b = sco.minimize(
                lambda w: float(w @ cb @ w),
                np.ones(n_activos) / n_activos,
                method='SLSQP',
                bounds=[(0.0, 1.0) for _ in range(n_activos)],
                constraints=(
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'ineq', 'fun': lambda w: float(mb @ w) - ro}
                )
            )
            if sol_b.success:
                vols_b.append(np.sqrt(float(sol_b.x @ cb @ sol_b.x)) * 100.0)
            else:
                vols_b.append(np.nan)
        curvas_vols_boot.append(vols_b)
        ax2.plot(vols_b, grid_ret, color="#F59E0B", lw=0.6, alpha=0.25, linestyle="-", zorder=2)
        
    matriz_vols = np.array(curvas_vols_boot)
    vol_p05 = np.nanpercentile(matriz_vols, 5, axis=0)
    vol_p10 = np.nanpercentile(matriz_vols, 10, axis=0)
    vol_p90 = np.nanpercentile(matriz_vols, 90, axis=0)
    vol_p95 = np.nanpercentile(matriz_vols, 95, axis=0)
    vol_media_boot = np.nanmean(matriz_vols, axis=0)
    
    # Bandas de confianza de Michaud
    ax2.fill_betweenx(grid_ret, vol_p05, vol_p95, color="#FEF3C7", alpha=0.40, label="Cono de Incertidumbre (90% IC)", zorder=1)
    ax2.fill_betweenx(grid_ret, vol_p10, vol_p90, color="#FDE68A", alpha=0.55, label="Cono de Incertidumbre (80% IC)", zorder=2)
    ax2.plot(vol_media_boot, grid_ret, color="#D97706", lw=2.4, linestyle="-", label="Frontera Promedio Remuestreada (Michaud)", zorder=4)
    if len(f_vol_ana) > 0:
        ax2.plot(f_vol_ana, f_ret_ana, color="#991B1B", lw=2.0, linestyle="--", label="Frontera Markowitz (Muestral)", zorder=5)
        
    # Puntos de contraste Markowitz vs Michaud vs HERC
    if "Markowitz" in pesos_dict:
        w_mvo = pesos_dict["Markowitz"]
        r_mvo = float(np.dot(w_mvo, vector_medias)) * 100.0
        v_mvo = np.sqrt(float(np.dot(w_mvo, np.dot(matriz_cov, w_mvo)))) * 100.0
        ax2.scatter(v_mvo, r_mvo, color=PALETA_MODELOS["Markowitz"], marker="X", s=140, edgecolors="#FFFFFF", lw=1.2, label="Markowitz MVO (Sobreajuste Muestral)", zorder=6)
        
    if "Michaud_Remuestreado" in pesos_dict:
        w_mic = pesos_dict["Michaud_Remuestreado"]
        r_mic = float(np.dot(w_mic, vector_medias)) * 100.0
        v_mic = np.sqrt(float(np.dot(w_mic, np.dot(matriz_cov, w_mic)))) * 100.0
        ax2.scatter(v_mic, r_mic, color=PALETA_MODELOS["Michaud_Remuestreado"], marker="p", s=140, edgecolors="#FFFFFF", lw=1.2, label="Michaud Remuestreado (Estimador Robusto)", zorder=6)
        
    if "HERC" in pesos_dict:
        w_hrc = pesos_dict["HERC"]
        r_hrc = float(np.dot(w_hrc, vector_medias)) * 100.0
        v_hrc = np.sqrt(float(np.dot(w_hrc, np.dot(matriz_cov, w_hrc)))) * 100.0
        ax2.scatter(v_hrc, r_hrc, color=PALETA_MODELOS["HERC"], marker="o", s=140, edgecolors="#FFFFFF", lw=1.6, label="HERC (Paridad Jerárquica)", zorder=7)
        
    ax2.set_title("B. Incertidumbre Muestral y Remuestreo Bootstrap (Michaud)", pad=11, fontweight="bold")
    ax2.set_xlabel("Volatilidad Anualizada (%)", fontsize=9.5)
    ax2.set_ylabel("Rendimiento Anualizado Esperado (%)", fontsize=9.5)
    ax2.set_xlim(3.5, 25.0)
    ax2.set_ylim(2.5, 16.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left", fontsize=7.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig02_frontera_eficiente_modelos.png"), dpi=300)
    plt.close()


def generar_figura_3_evolucion_pesos(
    dfs_ponderaciones: Dict[str, pd.DataFrame],
    tickers: List[str]
) -> None:
    """Figura 3: Evolución temporal de asignación de activos en Walk-Forward."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    modelos_visualizados = [
        ("Markowitz", "A. Markowitz MVO (Concentración e Inestabilidad)", axes[0, 0]),
        ("Min_CVaR", "B. Mínimo CVaR 95% (Énfasis en Baja Volatilidad)", axes[0, 1]),
        ("HERC", "C. HERC (Paridad Jerárquica y Estabilidad)", axes[1, 0]),
        ("Referencia_60_40", "D. Referencia 60/40 (Estática SPY/IEF)", axes[1, 1]),
    ]
    
    paleta_activos = [
        "#1E3A8A", "#3B82F6", "#0D9488", "#10B981",
        "#F59E0B", "#D97706", "#64748B", "#8B5CF6", "#A8A29E"
    ]
    
    for m, titulo, ax in modelos_visualizados:
        df_w = dfs_ponderaciones.get(m, pd.DataFrame())
        if not df_w.empty:
            ax.stackplot(
                df_w.index,
                [df_w[t].values * 100.0 for t in tickers],
                labels=[NOMBRES_ACTIVOS[t] for t in tickers],
                colors=paleta_activos[:len(tickers)],
                alpha=0.90
            )
        ax.set_title(titulo, pad=9, fontweight="bold")
        ax.set_ylabel("Ponderación (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.0
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(DIR_FIGURAS, "fig03_evolucion_pesos_walk_forward.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generar_figura_4_retorno_y_drawdown(
    df_rendimientos: pd.DataFrame
) -> None:
    """Figura 4: Evolución del patrimonio acumulado y curvas de retroceso (drawdown)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.2]})
    
    # Eventos de estrés de mercado
    eventos_estres = [
        ("2011-07-01", "2011-10-31", "Crisis Deuda UE\n(2011)"),
        ("2020-02-15", "2020-04-30", "COVID-19\n(2020)"),
        ("2022-01-01", "2022-10-31", "Shock Inflación\n(2022)"),
    ]
    
    for inicio_ev, fin_ev, texto in eventos_estres:
        ax1.axvspan(pd.to_datetime(inicio_ev), pd.to_datetime(fin_ev), color="#E2E8F0", alpha=0.50, zorder=1)
        ax2.axvspan(pd.to_datetime(inicio_ev), pd.to_datetime(fin_ev), color="#E2E8F0", alpha=0.50, zorder=1)
        
        # Etiqueta horizontal en el margen superior de ax1 para evitar cualquier cruce con las curvas
        mid_date = pd.to_datetime(inicio_ev) + (pd.to_datetime(fin_ev) - pd.to_datetime(inicio_ev)) / 2
        ax1.text(
            mid_date, 660, texto,
            fontsize=7.8, color="#1E293B", horizontalalignment="center", verticalalignment="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.92, lw=0.7),
            zorder=6
        )
        
    for col in df_rendimientos.columns:
        r = df_rendimientos[col]
        patrimonio = (1.0 + r).cumprod() * 100.0
        color = PALETA_MODELOS.get(col, "#333333")
        grosor = 2.2 if col == "HERC" else (1.6 if col in ["Markowitz", "Referencia_60_40"] else 1.2)
        estilo = "-" if col in ["HERC", "Markowitz", "Referencia_60_40"] else "--"
        
        ax1.plot(patrimonio.index, patrimonio.values, label=ETIQUETAS_MODELOS.get(col, col), color=color, lw=grosor, linestyle=estilo, zorder=3)
        
        pico = patrimonio.cummax()
        drawdown = (patrimonio - pico) / pico * 100.0
        ax2.plot(drawdown.index, drawdown.values, color=color, lw=grosor * 0.85, linestyle=estilo, zorder=3)
        
    ax1.set_title("A. Evolución del Patrimonio Fuera de Muestra (Base 100 = 2010)", pad=12, fontweight="bold")
    ax1.set_ylabel("Índice de Valor Acumulado")
    ax1.set_ylim(50, 720)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left", ncol=2, fontsize=8.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    ax2.set_title("B. Curva Submarina de Retroceso Acumulado / Drawdown (%)", pad=8, fontweight="bold")
    ax2.set_ylabel("Caída desde Pico (%)")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylim(-46, 3)
    ax2.axhline(-20, color="#991B1B", linestyle=":", lw=0.9, alpha=0.7, label="Umbral -20%")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig04_retorno_acumulado_y_drawdown.png"), dpi=300)
    plt.close()


def generar_figura_5_evt_colas(
    df_rendimientos: pd.DataFrame
) -> None:
    """Figura 5: Densidad empírica de colas y ajuste de Pareto Generalizada (GPD) bajo GARCH-EVT (McNeil & Frey, 2000)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    
    # Subplot A: Densidad empírica de rendimientos
    ax1 = axes[0]
    r_herc = df_rendimientos["HERC"].values * 100.0
    r_mvo = df_rendimientos["Markowitz"].values * 100.0
    
    sns.kdeplot(r_herc, ax=ax1, color=PALETA_MODELOS["HERC"], lw=2.2, label="HERC", fill=True, alpha=0.18)
    sns.kdeplot(r_mvo, ax=ax1, color=PALETA_MODELOS["Markowitz"], lw=1.8, label="Markowitz MVO", fill=True, alpha=0.10)
    
    var95_herc = np.percentile(r_herc, 5.0)
    var95_mvo = np.percentile(r_mvo, 5.0)
    ax1.axvline(var95_herc, color=PALETA_MODELOS["HERC"], linestyle=":", lw=1.6, label=f"VaR 95% HERC ({var95_herc:.2f}%)")
    ax1.axvline(var95_mvo, color=PALETA_MODELOS["Markowitz"], linestyle=":", lw=1.6, label=f"VaR 95% MVO ({var95_mvo:.2f}%)")
    
    ax1.set_title("A. Densidad Empírica de Rendimientos Diarios", pad=12, fontweight="bold")
    ax1.set_xlabel("Rendimiento Diario (%)")
    ax1.set_ylabel("Densidad de Probabilidad")
    ax1.set_xlim(-6, 4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left")
    
    # Subplot B: Ajuste GARCH-EVT (McNeil & Frey, 2000)
    ax2 = axes[1]
    ajuste_evt = ajustar_modelo_evt_gpd(df_rendimientos["HERC"].values, cuantil_umbral=0.90, usar_filtrado_garch=True)
    
    excesos = ajuste_evt["excesos"]
    u = ajuste_evt["umbral_u"]
    xi = ajuste_evt["parametro_xi"]
    beta = ajuste_evt["parametro_beta"]
    p_arch = ajuste_evt["p_val_arch"]
    
    ax2.hist(excesos, bins=25, density=True, color="#0D9488", alpha=0.45, edgecolor="#0F2942", label="Excesos Estándar ($-z_t > u$)")
    
    eje_x = np.linspace(0, max(excesos) * 1.15, 120)
    pdf_gpd = stats.genpareto.pdf(eje_x, xi, loc=0, scale=beta)
    ax2.plot(eje_x, pdf_gpd, color="#0F2942", lw=2.2, label=f"Ajuste GPD ($\\xi={xi:.2f}, \\beta={beta:.2f}$)")
    
    # Marcadores de VaR y CVaR estandarizados
    z_var99 = u + (beta / xi) * (((len(df_rendimientos) / float(len(excesos))) * 0.01) ** (-xi) - 1.0) if xi != 0 else u - beta * np.log(0.01)
    ax2.axvline(z_var99 - u, color="#991B1B", linestyle="--", lw=1.8, label=f"VaR 99% EVT ({ajuste_evt['var_99_evt']:.2f}%)")
    
    # Badge con información de validación i.i.d.
    ax2.text(
        0.96, 0.48,
        f"Filtrado AR(1)-GARCH(1,1)\nResiduos $z_t \\sim i.i.d.$\nTest ARCH $p={p_arch:.4f}$ (Valida $i.i.d.$)",
        transform=ax2.transAxes,
        ha="right", va="center",
        fontsize=8.0,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.92, lw=0.7),
        zorder=6
    )
    
    ax2.set_title("B. Ajuste GARCH-EVT en Innovaciones Estandarizadas ($i.i.d.$)", pad=12, fontweight="bold")
    ax2.set_xlabel("Exceso de Pérdida Estandarizada sobre Umbral $u$")
    ax2.set_ylabel("Densidad Condicional de Excesos")
    ax2.set_ylim(0, max(pdf_gpd) * 1.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=8.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig05_ajuste_evt_pareto_colas.png"), dpi=300)
    plt.close()


def generar_figura_6_cpcv(df_cpcv: pd.DataFrame) -> None:
    """
    Figura 6: Distribución del Ratio de Sharpe en Validación Cruzada Combinatoria con Embargo (CPCV).
    Diagrama de cajas y puntos individuales optimizado para máxima legibilidad sin solapamientos.
    """
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    
    orden_estrategias = ["Equiponderado", "Referencia_60_40", "Markowitz", "Min_CVaR", "Min_CDaR", "HERC"]
    paleta_colores = [PALETA_MODELOS.get(m, "#333333") for m in orden_estrategias]
    
    etiquetas_mostradas = [
        "1/N\nEquiponderado",
        "Referencia 60/40\n(SPY / IEF)",
        "Markowitz\n(MVO Clásico)",
        "Mínimo CVaR\n(95% Confianza)",
        "Mínimo CDaR\n(95% Confianza)",
        "HERC\n(Paridad Jerárquica)"
    ]
    
    # 1. Boxplot limpio
    sns.boxplot(
        data=df_cpcv[orden_estrategias],
        palette=paleta_colores,
        ax=ax,
        width=0.38,
        boxprops=dict(alpha=0.82, edgecolor="#0F172A", linewidth=1.2),
        whiskerprops=dict(color="#0F172A", linewidth=1.2),
        capprops=dict(color="#0F172A", linewidth=1.2),
        medianprops=dict(color="#FFFFFF", linewidth=2.2),
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="#000000", markerfacecolor="#FFFFFF", markersize=6.5, zorder=4),
        zorder=2
    )
    
    # 2. Stripplot con dispersión controlada
    sns.stripplot(
        data=df_cpcv[orden_estrategias],
        color="#0F2942",
        alpha=0.55,
        size=5.5,
        jitter=0.10,
        edgecolor="#FFFFFF",
        linewidth=0.5,
        ax=ax,
        zorder=3
    )
    
    # 3. Badges con estadísticas resumen arriba de cada columna
    val_max = df_cpcv[orden_estrategias].max().max()
    val_min = df_cpcv[orden_estrategias].min().min()
    y_badge = val_max + 0.08
    
    for i, col_name in enumerate(orden_estrategias):
        med = float(df_cpcv[col_name].median())
        q75, q25 = np.percentile(df_cpcv[col_name].dropna(), [75, 25])
        iqr = q75 - q25
        ax.text(
            i, y_badge,
            f"Med: {med:.2f}\nIQR: {iqr:.2f}",
            ha="center", va="bottom",
            fontsize=8.0,
            color="#1E293B",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.92, lw=0.7),
            zorder=5
        )
        
    ax.set_xticks(range(len(orden_estrategias)))
    ax.set_xticklabels(etiquetas_mostradas, rotation=0, ha="center", fontsize=9.2)
    ax.set_title("Distribución Fuera de Muestra del Ratio de Sharpe (CPCV con 10 Días de Embargo, $N=6, k=2$)", pad=24, fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Ratio de Sharpe Anualizado ($R_f = 2.0\\%$)", fontsize=9.5)
    ax.axhline(0, color="#64748B", linestyle="--", lw=1.0, alpha=0.7, label="Breakeven ($SR=0$)")
    ax.axhline(0.50, color="#1E3A8A", linestyle=":", lw=1.1, alpha=0.7, label="Umbral Institucional ($SR=0.50$)")
    ax.set_ylim(min(val_min - 0.15, -0.20), y_badge + 0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=8.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig06_cpcv_distribucion_sharpe.png"), dpi=300)
    plt.close()


def generar_figura_7_presupuesto_riesgo(
    pesos_dict: Dict[str, np.ndarray],
    matriz_cov: np.ndarray,
    tickers: List[str]
) -> None:
    """Figura 7: Ponderación de capital vs Contribución marginal de riesgo."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    
    estrategias_comparadas = [
        ("Markowitz", "A. Markowitz MVO (Concentración Crítica de Riesgo)", axes[0]),
        ("HERC", "B. HERC (Asignación Jerárquica de Riesgo Equitativo)", axes[1]),
    ]
    
    # Calcular máximo global para acotar el eje Y con holgura
    max_global = 0.0
    for m, _, _ in estrategias_comparadas:
        if m in pesos_dict:
            w_temp = pesos_dict[m]
            var_temp = float(np.dot(w_temp, np.dot(matriz_cov, w_temp)))
            cm_temp = np.dot(matriz_cov, w_temp) / np.sqrt(max(var_temp, 1e-8))
            rc_temp = (w_temp * cm_temp) / np.sqrt(max(var_temp, 1e-8)) * 100.0
            max_global = max(max_global, float(np.max(w_temp * 100.0)), float(np.max(rc_temp)))
            
    limite_y = max(max_global * 1.35, 60.0)
    posiciones_x = np.arange(len(tickers))
    ancho_barra = 0.38
    
    for m, titulo, ax in estrategias_comparadas:
        w = pesos_dict[m]
        varianza_cartera = float(np.dot(w, np.dot(matriz_cov, w)))
        contribucion_marginal = np.dot(matriz_cov, w) / np.sqrt(max(varianza_cartera, 1e-8))
        porcentaje_rc = (w * contribucion_marginal) / np.sqrt(max(varianza_cartera, 1e-8)) * 100.0
        
        ax.bar(
            posiciones_x - ancho_barra/2, w * 100.0, ancho_barra,
            label="Ponderación de Capital (% $w_i$)", color="#64748B", alpha=0.85, edgecolor="#334155", linewidth=0.8
        )
        ax.bar(
            posiciones_x + ancho_barra/2, porcentaje_rc, ancho_barra,
            label="Contribución al Riesgo (% $RC_i$)", color=PALETA_MODELOS[m], alpha=0.85, edgecolor="#1E293B", linewidth=0.8
        )
        
        ax.set_title(titulo, pad=12, fontweight="bold")
        ax.set_xticks(posiciones_x)
        ax.set_xticklabels(tickers, rotation=0, fontsize=9.0)
        ax.set_ylabel("Porcentaje (%)", fontsize=9.5)
        ax.set_ylim(0, limite_y)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=8.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, "fig07_descomposicion_riesgo_cluster.png"), dpi=300)
    plt.close()


# ==============================================================================
# 10. EJECUCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================
def main() -> None:
    print("Iniciando pipeline de optimización de portafolios (Motor Nativo SciPy/NumPy)...")
    
    # 1. Descarga e ingestión
    precios = descargar_datos(TICKERS, fecha_inicio="2007-01-01", fecha_fin="2026-08-01")
    rendimientos_simples, rendimientos_logaritmicos = calcular_rendimientos(precios)
    
    # 1.1 Diagnóstico econométrico y validación formal de supuestos
    print("Ejecutando contraste econométrico y verificación de supuestos sobre los 9 activos...")
    df_diagnostico = ejecutar_diagnostico_econometrico(precios, rendimientos_logaritmicos)
    print("\n--- RESUMEN DE DIAGNÓSTICO ECONOMÉTRICO (ADF, JB, Ljung-Box, ARCH-LM) ---")
    print(df_diagnostico[["Retorno Anual (%)", "Volatilidad Anual (%)", "Asimetría (Skew)", "Curtosis Exceso", "JB p-valor", "Q(10) Retornos (p-val)", "ARCH Q(10) (p-val)", "ADF Retornos (p-val)"]].round(4).to_string())
    
    # 2. Estimación de parámetros
    matriz_cov, matriz_corr = estimar_covarianza_y_correlacion(rendimientos_logaritmicos)
    vector_medias = rendimientos_logaritmicos.mean().values * 252.0
    matriz_distancias = calcular_distancia_angular(matriz_corr)
    arbol_enlace, orden_cuasidiagonal = obtener_agrupamiento_jerarquico(matriz_distancias, metodo_enlace="ward")
    
    # 3. Modelos estáticos
    n_activos = len(TICKERS)
    pesos_estaticos = {
        "Equiponderado": optimizar_equiponderado(n_activos),
        "Referencia_60_40": optimizar_referencia_60_40(TICKERS),
        "Markowitz": optimizar_markowitz(vector_medias, matriz_cov),
        "Michaud_Remuestreado": optimizar_michaud_remuestreado(rendimientos_simples.values, n_simulaciones=25),
        "Min_CVaR": optimizar_minimo_cvar(rendimientos_simples.values, nivel_confianza=0.95),
        "Min_CDaR": optimizar_minimo_cdar(rendimientos_simples.values, nivel_confianza=0.95),
        "HERC": optimizar_herc(matriz_cov, arbol_enlace, orden_cuasidiagonal),
    }
    
    df_pesos_estaticos = pd.DataFrame(pesos_estaticos, index=TICKERS).T
    df_pesos_estaticos.to_csv(os.path.join(DIR_DATOS, "pesos_estaticos_muestra_completa.csv"))
    
    # 4. Figuras estáticas
    generar_figura_1_dendrograma_y_correlaciones(matriz_corr, arbol_enlace, orden_cuasidiagonal, TICKERS)
    generar_figura_2_frontera_eficiente(vector_medias, matriz_cov, pesos_estaticos, TICKERS)
    generar_figura_7_presupuesto_riesgo(pesos_estaticos, matriz_cov, TICKERS)
    
    # 5. Walk-Forward fuera de muestra
    df_rendimientos_oos, _, dfs_ponderaciones = ejecutar_walk_forward(
        precios=precios,
        tickers=TICKERS,
        dias_ventana=756,
        dias_rebalanceo=21,
        tasa_friccion_bps=5.0
    )
    df_rendimientos_oos.to_csv(os.path.join(DIR_DATOS, "retornos_diarios_oos.csv"))
    
    for m, df_w in dfs_ponderaciones.items():
        df_w.to_csv(os.path.join(DIR_DATOS, f"pesos_historicos_{m}.csv"))
        
    # 6. Figuras dinámicas fuera de muestra
    generar_figura_3_evolucion_pesos(dfs_ponderaciones, TICKERS)
    generar_figura_4_retorno_y_drawdown(df_rendimientos_oos)
    generar_figura_5_evt_colas(df_rendimientos_oos)
    
    # 7. Validación Cruzada Combinatoria (CPCV)
    df_cpcv = ejecutar_cpcv(precios, TICKERS, n_bloques=6, k_bloques_test=2, dias_embargo=10)
    df_cpcv.to_csv(os.path.join(DIR_DATOS, "cpcv_sharpe_distribucion.csv"))
    generar_figura_6_cpcv(df_cpcv)
    
    # 8. Tabla de métricas
    df_metricas = calcular_tabla_metricas(df_rendimientos_oos, dfs_ponderaciones, tasa_libre_riesgo=0.02)
    df_metricas.to_csv(os.path.join(DIR_DATOS, "metricas_comparativas.csv"))
    
    print("\nResultados Fuera de Muestra (2010-2026):")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_metricas.round(2).to_string())
    print("\nPipeline completado exitosamente.")


if __name__ == "__main__":
    main()
