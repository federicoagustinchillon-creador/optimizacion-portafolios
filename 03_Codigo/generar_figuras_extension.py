"""
Genera / regenera las figuras que integran DRO-Wasserstein y Marco Generativo
Ex-Ante al cuerpo principal del paper (no como anexo aparte).

No modifica pipeline_optimizacion.py ni extension_dro_exante.py -- solo los
importa y reusa sus funciones y constantes ya publicadas/verificadas.

Salidas en 04_Figuras/:
  - fig02_frontera_eficiente_modelos.png    (REGENERA: Panel A agrega 2 puntos)
  - fig03_evolucion_pesos_walk_forward.png  (REGENERA: 6 paneles en vez de 4;
                                              absorbe lo que antes era fig08)
  - fig04_retorno_acumulado_y_drawdown.png  (REGENERA: agrega 2 series)
  - fig06_cpcv_distribucion_sharpe.png      (REGENERA: agrega 2 columnas)
  - fig07_descomposicion_riesgo_cluster.png (REGENERA: 4 paneles en vez de 2)

fig08_evolucion_pesos_extension.png queda OBSOLETA (absorbida en fig03) y se
elimina del disco al final de este script.
"""
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODIGO = os.path.join(BASE, "03_Codigo")
DATOS = os.path.join(BASE, "05_Datos_y_Resultados")
FIGURAS = os.path.join(BASE, "04_Figuras")
sys.path.insert(0, CODIGO)

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import pipeline_optimizacion as pipe
import extension_dro_exante as ext

TICKERS = pipe.TICKERS
NOMBRES_ACTIVOS = pipe.NOMBRES_ACTIVOS
PALETA_MODELOS = dict(pipe.PALETA_MODELOS)
ETIQUETAS_MODELOS = dict(pipe.ETIQUETAS_MODELOS)

# Paleta y etiquetas extendidas para los 2 modelos nuevos (colores no usados
# por ninguno de los 7 modelos originales, para evitar confusion visual)
PALETA_MODELOS["DRO_Wasserstein"] = "#16A34A"   # Verde Esmeralda
PALETA_MODELOS["Min_CVaR_ExAnte"] = "#DB2777"   # Magenta
ETIQUETAS_MODELOS["DRO_Wasserstein"] = "DRO-Wasserstein"
ETIQUETAS_MODELOS["Min_CVaR_ExAnte"] = "Mínimo CVaR Ex-Ante"

print("=== 1. Cargando datos cacheados (precios, rendimientos) ===")
precios = pipe.descargar_datos(TICKERS, fecha_inicio="2007-01-01", fecha_fin="2026-08-01")
rendimientos_simples, rendimientos_log = pipe.calcular_rendimientos(precios)
matriz_cov, matriz_corr = pipe.estimar_covarianza_y_correlacion(rendimientos_log)
vector_medias = rendimientos_log.mean().values * 252.0
print("precios:", precios.shape)

# ------------------------------------------------------------------
# 2. Pesos estáticos de muestra completa: Markowitz/HERC ya
#    persistidos (se reusan tal cual, sin reintroducir variacion de
#    punto flotante en paneles ya publicados); DRO/Ex-Ante cacheados
#    en una corrida anterior de este mismo script (una sola
#    optimizacion sobre toda la muestra, igual criterio que el resto).
# ------------------------------------------------------------------
print("=== 2. Pesos estaticos (muestra completa 2007-2026) ===")
df_pesos_estaticos = pd.read_csv(os.path.join(DATOS, "pesos_estaticos_muestra_completa.csv"), index_col=0)
pesos_estaticos_originales = {m: df_pesos_estaticos.loc[m, TICKERS].values.astype(float) for m in df_pesos_estaticos.index}
w_markowitz = pesos_estaticos_originales["Markowitz"]
w_herc = pesos_estaticos_originales["HERC"]

ruta_pesos_ext = os.path.join(DATOS, "pesos_estaticos_extension_muestra_completa.csv")
if os.path.exists(ruta_pesos_ext):
    df_pesos_ext = pd.read_csv(ruta_pesos_ext, index_col=0)
    w_dro = df_pesos_ext.loc["DRO_Wasserstein", TICKERS].values.astype(float)
    w_exante = df_pesos_ext.loc["Min_CVaR_ExAnte", TICKERS].values.astype(float)
    print("Pesos estaticos extendidos: cargados desde cache.")
else:
    print("Calculando DRO-Wasserstein estatico (SOCP, una sola resolucion sobre toda la muestra)...")
    w_dro = ext.optimizar_dro_wasserstein(rendimientos_simples.values, radio_epsilon=0.02, nivel_confianza=0.95)
    print("  OK:", dict(zip(TICKERS, np.round(w_dro, 4))))

    print("Calculando Marco Generativo Ex-Ante estatico (GARCH-EVT-copula t + Min-CVaR sobre escenarios, muestra completa)...")
    w_exante = ext.optimizar_minimo_cvar_ex_ante(rendimientos_log, n_escenarios=3000, nivel_confianza=0.95)
    print("  OK:", dict(zip(TICKERS, np.round(w_exante, 4))))

    pd.DataFrame(
        {"DRO_Wasserstein": w_dro, "Min_CVaR_ExAnte": w_exante},
        index=TICKERS
    ).T.to_csv(ruta_pesos_ext)
    print("Guardado:", ruta_pesos_ext)

pesos_estaticos_todos = dict(pesos_estaticos_originales)
pesos_estaticos_todos["DRO_Wasserstein"] = w_dro
pesos_estaticos_todos["Min_CVaR_ExAnte"] = w_exante

# ------------------------------------------------------------------
# 3. FIGURA 2 (regenerada): Panel A agrega DRO-Wasserstein y Ex-Ante
#    Reimplementa exactamente pipe.generar_figura_2_frontera_eficiente,
#    unico cambio: el diccionario `simbolos` del Panel A incluye los
#    2 modelos nuevos. Panel B (cono de Michaud) queda sin cambios de
#    logica -- es una ilustracion de incertidumbre muestral, no compara
#    modelos especificos mas alla de Markowitz/Michaud/HERC ya fijados
#    en el documento.
# ------------------------------------------------------------------
print("=== 3. Regenerando Figura 2 (frontera eficiente, Panel A con 9 modelos) ===")


def generar_figura_2_extendida(vector_medias, matriz_cov, pesos_dict, tickers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6))
    n_activos = len(vector_medias)

    np.random.seed(42)
    n_simulaciones = 6000
    pesos_sim_list = [
        np.random.dirichlet(np.ones(n_activos) * alpha, size=n_simulaciones // 4)
        for alpha in [0.25, 0.60, 1.20, 2.50]
    ]
    pesos_simulados = np.vstack(pesos_sim_list)
    retornos_sim = (pesos_simulados @ vector_medias) * 100.0
    volatilidades_sim = np.sqrt(np.sum((pesos_simulados @ matriz_cov) * pesos_simulados, axis=1)) * 100.0
    sharpes_sim = (retornos_sim - 2.0) / np.maximum(volatilidades_sim, 1e-6)

    disp = ax1.scatter(volatilidades_sim, retornos_sim, c=sharpes_sim, cmap="mako", alpha=0.30, s=12, edgecolors="none", zorder=1)
    cbar = plt.colorbar(disp, ax=ax1, pad=0.02, shrink=0.88)
    cbar.set_label("Ratio de Sharpe ($R_f = 2.0\\%$)", fontsize=9.0)

    vols_activos = np.sqrt(np.diag(matriz_cov)) * 100.0
    rets_activos = vector_medias * 100.0
    ax1.scatter(vols_activos, rets_activos, color="#0F2942", marker="o", s=85, edgecolors="#FFFFFF", linewidths=1.2, zorder=6, label="Activos Individuales ($N=9$)")

    desplazamientos = {
        "SPY": (6, 5), "QQQ": (-30, 5), "EEM": (6, -5),
        "VNQ": (6, 5), "GLD": (-26, 5), "TLT": (6, -5),
        "IEF": (6, 4), "LQD": (6, -5), "DBC": (6, -5)
    }
    for i, t in enumerate(tickers):
        dx, dy = desplazamientos.get(t, (6, 3))
        ax1.annotate(t, xy=(vols_activos[i], rets_activos[i]), xytext=(dx, dy), textcoords="offset points",
                     fontsize=8.2, fontweight="bold", color="#0F2942",
                     bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.90, lw=0.7), zorder=7)

    retornos_obj = np.linspace(float(np.min(vector_medias)), float(np.max(vector_medias)) * 0.98, 40)
    f_vol_ana, f_ret_ana = [], []
    for r_obj in retornos_obj:
        w0 = np.ones(n_activos) / n_activos
        res = sco.minimize(
            lambda w: float(w @ matriz_cov @ w), w0, method='SLSQP',
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

    simbolos = {
        "Equiponderado": ("s", PALETA_MODELOS["Equiponderado"]),
        "Referencia_60_40": ("D", PALETA_MODELOS["Referencia_60_40"]),
        "Markowitz": ("X", PALETA_MODELOS["Markowitz"]),
        "Michaud_Remuestreado": ("p", PALETA_MODELOS["Michaud_Remuestreado"]),
        "Min_CVaR": ("^", PALETA_MODELOS["Min_CVaR"]),
        "Min_CDaR": ("v", PALETA_MODELOS["Min_CDaR"]),
        "HERC": ("o", PALETA_MODELOS["HERC"]),
        "DRO_Wasserstein": ("*", PALETA_MODELOS["DRO_Wasserstein"]),
        "Min_CVaR_ExAnte": ("P", PALETA_MODELOS["Min_CVaR_ExAnte"]),
    }
    for m, (simb, col) in simbolos.items():
        if m in pesos_dict:
            w_m = pesos_dict[m]
            r_m = float(np.dot(w_m, vector_medias)) * 100.0
            v_m = np.sqrt(float(np.dot(w_m, np.dot(matriz_cov, w_m)))) * 100.0
            tam = 160 if m in ("HERC", "DRO_Wasserstein", "Min_CVaR_ExAnte") else 115
            borde = 1.6 if m in ("HERC", "DRO_Wasserstein", "Min_CVaR_ExAnte") else 1.0
            ax1.scatter(v_m, r_m, color=col, marker=simb, s=tam, zorder=8, edgecolors="#FFFFFF", linewidths=borde, label=ETIQUETAS_MODELOS[m])

    ax1.set_title("A. Espacio Media-Varianza, Activos y los Nueve Modelos Evaluados", pad=11, fontweight="bold")
    ax1.set_xlabel("Volatilidad Anualizada (%)", fontsize=9.5)
    ax1.set_ylabel("Rendimiento Anualizado Esperado (%)", fontsize=9.5)
    ax1.set_xlim(3.5, 33.0)
    ax1.set_ylim(0.5, 17.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left", fontsize=6.6, ncol=1, bbox_to_anchor=(1.02, 1.0))

    # ---------------- PANEL B: Remuestreo Bootstrap de Michaud (sin cambios) ----------------
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
                lambda w: float(w @ cb @ w), np.ones(n_activos) / n_activos, method='SLSQP',
                bounds=[(0.0, 1.0) for _ in range(n_activos)],
                constraints=(
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'ineq', 'fun': lambda w: float(mb @ w) - ro}
                )
            )
            vols_b.append(np.sqrt(float(sol_b.x @ cb @ sol_b.x)) * 100.0 if sol_b.success else np.nan)
        curvas_vols_boot.append(vols_b)
        ax2.plot(vols_b, grid_ret, color="#F59E0B", lw=0.6, alpha=0.25, linestyle="-", zorder=2)

    matriz_vols = np.array(curvas_vols_boot)
    vol_p05 = np.nanpercentile(matriz_vols, 5, axis=0)
    vol_p10 = np.nanpercentile(matriz_vols, 10, axis=0)
    vol_p90 = np.nanpercentile(matriz_vols, 90, axis=0)
    vol_p95 = np.nanpercentile(matriz_vols, 95, axis=0)
    vol_media_boot = np.nanmean(matriz_vols, axis=0)

    ax2.fill_betweenx(grid_ret, vol_p05, vol_p95, color="#FEF3C7", alpha=0.40, label="Cono de Incertidumbre (90% IC)", zorder=1)
    ax2.fill_betweenx(grid_ret, vol_p10, vol_p90, color="#FDE68A", alpha=0.55, label="Cono de Incertidumbre (80% IC)", zorder=2)
    ax2.plot(vol_media_boot, grid_ret, color="#D97706", lw=2.4, linestyle="-", label="Frontera Promedio Remuestreada (Michaud)", zorder=4)
    if len(f_vol_ana) > 0:
        ax2.plot(f_vol_ana, f_ret_ana, color="#991B1B", lw=2.0, linestyle="--", label="Frontera Markowitz (Muestral)", zorder=5)

    w_mvo = pesos_dict["Markowitz"]
    r_mvo = float(np.dot(w_mvo, vector_medias)) * 100.0
    v_mvo = np.sqrt(float(np.dot(w_mvo, np.dot(matriz_cov, w_mvo)))) * 100.0
    ax2.scatter(v_mvo, r_mvo, color=PALETA_MODELOS["Markowitz"], marker="X", s=140, edgecolors="#FFFFFF", lw=1.2, label="Markowitz MVO (Sobreajuste Muestral)", zorder=6)

    w_mic = pesos_dict["Michaud_Remuestreado"]
    r_mic = float(np.dot(w_mic, vector_medias)) * 100.0
    v_mic = np.sqrt(float(np.dot(w_mic, np.dot(matriz_cov, w_mic)))) * 100.0
    ax2.scatter(v_mic, r_mic, color=PALETA_MODELOS["Michaud_Remuestreado"], marker="p", s=140, edgecolors="#FFFFFF", lw=1.2, label="Michaud Remuestreado (Estimador Robusto)", zorder=6)

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
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left", fontsize=7.0, bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS, "fig02_frontera_eficiente_modelos.png"), dpi=300)
    plt.close()


generar_figura_2_extendida(vector_medias, matriz_cov, pesos_estaticos_todos, TICKERS)
print("  fig02 regenerada.")

# ------------------------------------------------------------------
# 4. FIGURA 3 (regenerada): evolucion de pesos walk-forward, 6 paneles
#    (absorbe lo que antes era la Figura 8 separada -- un solo lugar
#    para "como cambian los pesos en el tiempo", no dos figuras).
# ------------------------------------------------------------------
print("=== 4. Regenerando Figura 3 (evolucion de pesos, 6 paneles) ===")
df_w_dro = pd.read_csv(os.path.join(DATOS, "pesos_historicos_DRO_Wasserstein.csv"), index_col=0, parse_dates=True)
df_w_exante = pd.read_csv(os.path.join(DATOS, "pesos_historicos_Min_CVaR_ExAnte.csv"), index_col=0, parse_dates=True)
df_w_markowitz = pd.read_csv(os.path.join(DATOS, "pesos_historicos_Markowitz.csv"), index_col=0, parse_dates=True)
df_w_mincvar = pd.read_csv(os.path.join(DATOS, "pesos_historicos_Min_CVaR.csv"), index_col=0, parse_dates=True)
df_w_herc = pd.read_csv(os.path.join(DATOS, "pesos_historicos_HERC.csv"), index_col=0, parse_dates=True)
df_w_6040 = pd.read_csv(os.path.join(DATOS, "pesos_historicos_Referencia_60_40.csv"), index_col=0, parse_dates=True)


def generar_figura_3_extendida(dfs_ponderaciones, tickers):
    fig, axes = plt.subplots(2, 3, figsize=(19, 8.5), sharex=True, sharey=True)
    modelos_visualizados = [
        (dfs_ponderaciones["Markowitz"], "A. Markowitz MVO (Concentración e Inestabilidad)", axes[0, 0]),
        (dfs_ponderaciones["Min_CVaR"], "B. Mínimo CVaR 95% (Énfasis en Baja Volatilidad)", axes[0, 1]),
        (dfs_ponderaciones["HERC"], "C. HERC (Paridad Jerárquica y Estabilidad)", axes[0, 2]),
        (dfs_ponderaciones["Referencia_60_40"], "D. Referencia 60/40 (Estática SPY/IEF)", axes[1, 0]),
        (dfs_ponderaciones["DRO_Wasserstein"], "E. DRO-Wasserstein ($\\varepsilon=0.02$, Robustez sin Concentración)", axes[1, 1]),
        (dfs_ponderaciones["Min_CVaR_ExAnte"], "F. Mínimo CVaR Ex-Ante (GARCH-EVT-Cópula $t$, Rotación Elevada)", axes[1, 2]),
    ]

    paleta_activos = [
        "#1E3A8A", "#3B82F6", "#0D9488", "#10B981",
        "#F59E0B", "#D97706", "#64748B", "#8B5CF6", "#A8A29E"
    ]

    for df_w, titulo, ax in modelos_visualizados:
        ax.stackplot(
            df_w.index,
            [df_w[t].values * 100.0 for t in tickers],
            labels=[NOMBRES_ACTIVOS[t] for t in tickers],
            colors=paleta_activos[:len(tickers)],
            alpha=0.90
        )
        ax.set_title(titulo, pad=9, fontweight="bold", fontsize=10.3)
        ax.set_ylabel("Ponderación (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=9, bbox_to_anchor=(0.5, -0.06),
        frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=7.6
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(os.path.join(FIGURAS, "fig03_evolucion_pesos_walk_forward.png"), dpi=300, bbox_inches="tight")
    plt.close()


generar_figura_3_extendida(
    {
        "Markowitz": df_w_markowitz, "Min_CVaR": df_w_mincvar, "HERC": df_w_herc,
        "Referencia_60_40": df_w_6040, "DRO_Wasserstein": df_w_dro, "Min_CVaR_ExAnte": df_w_exante,
    },
    TICKERS
)
print("  fig03 regenerada (6 paneles).")

ruta_fig08_obsoleta = os.path.join(FIGURAS, "fig08_evolucion_pesos_extension.png")
if os.path.exists(ruta_fig08_obsoleta):
    os.remove(ruta_fig08_obsoleta)
    print("  fig08 (obsoleta, absorbida en fig03) eliminada del disco.")

# ------------------------------------------------------------------
# 5. FIGURA 4 (regenerada): retorno acumulado + drawdown, 9 series
# ------------------------------------------------------------------
print("=== 5. Regenerando Figura 4 (retorno acumulado + drawdown, 9 modelos) ===")
df_ret_main = pd.read_csv(os.path.join(DATOS, "retornos_diarios_oos.csv"), index_col=0, parse_dates=True)
df_ret_ext = pd.read_csv(os.path.join(DATOS, "retornos_diarios_oos_extension.csv"), index_col=0, parse_dates=True)
df_ret_todos = df_ret_main.join(df_ret_ext, how="inner")
assert len(df_ret_todos) == len(df_ret_main), "Fechas de la extension no calzan 1:1 con el panel principal"
print("Series combinadas:", list(df_ret_todos.columns), "filas:", len(df_ret_todos))


def generar_figura_4_extendida(df_rendimientos):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.5, 9.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.2]})

    eventos_estres = [
        ("2011-07-01", "2011-10-31", "Crisis Deuda UE\n(2011)"),
        ("2020-02-15", "2020-04-30", "COVID-19\n(2020)"),
        ("2022-01-01", "2022-10-31", "Shock Inflación\n(2022)"),
    ]
    total_days = (df_rendimientos.index[-1] - df_rendimientos.index[0]).days
    for inicio_ev, fin_ev, texto in eventos_estres:
        t0 = pd.to_datetime(inicio_ev)
        t1 = pd.to_datetime(fin_ev)
        ax1.axvspan(t0, t1, color="#E2E8F0", alpha=0.50, zorder=1)
        ax2.axvspan(t0, t1, color="#E2E8F0", alpha=0.50, zorder=1)
        mid_date = t0 + (t1 - t0) / 2
        x_frac = (mid_date - df_rendimientos.index[0]).days / total_days
        ax1.annotate(
            texto, xy=(x_frac, 1.0), xycoords="axes fraction",
            xytext=(0, 4), textcoords="offset points",
            fontsize=7.5, color="#1E293B", ha="center", va="bottom", weight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.92, lw=0.6),
            annotation_clip=False, zorder=6
        )

    orden_columnas = [
        "Equiponderado", "Referencia_60_40", "Markowitz", "Michaud_Remuestreado",
        "Min_CVaR", "Min_CDaR", "HERC", "DRO_Wasserstein", "Min_CVaR_ExAnte",
    ]
    orden_columnas = [c for c in orden_columnas if c in df_rendimientos.columns]

    for col in orden_columnas:
        r = df_rendimientos[col]
        patrimonio = (1.0 + r).cumprod() * 100.0
        color = PALETA_MODELOS.get(col, "#333333")
        if col in ("DRO_Wasserstein", "Min_CVaR_ExAnte"):
            grosor = 2.0
            estilo = "-." if col == "DRO_Wasserstein" else ":"
        else:
            grosor = 2.2 if col == "HERC" else (1.6 if col in ["Markowitz", "Referencia_60_40"] else 1.2)
            estilo = "-" if col in ["HERC", "Markowitz", "Referencia_60_40"] else "--"

        ax1.plot(patrimonio.index, patrimonio.values, label=ETIQUETAS_MODELOS.get(col, col), color=color, lw=grosor, linestyle=estilo, zorder=3)
        pico = patrimonio.cummax()
        drawdown = (patrimonio - pico) / pico * 100.0
        ax2.plot(drawdown.index, drawdown.values, color=color, lw=grosor * 0.85, linestyle=estilo, zorder=3)

    ax1.set_title("A. Evolución del Patrimonio Fuera de Muestra (Base 100 = 2010) -- Nueve Modelos", pad=28, fontweight="bold")
    ax1.set_ylabel("Índice de Valor Acumulado")
    ax1.set_ylim(50, 720)
    ax1.legend(
        frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1",
        loc="upper left", ncol=1, fontsize=7.0,
        handlelength=1.8, columnspacing=0.8, handletextpad=0.5,
        bbox_to_anchor=(0.01, 0.95)
    )
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
    fig.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(FIGURAS, "fig04_retorno_acumulado_y_drawdown.png"), dpi=300, bbox_inches="tight")
    plt.close()


generar_figura_4_extendida(df_ret_todos)
print("  fig04 regenerada.")

# ------------------------------------------------------------------
# 6. FIGURA 6 (regenerada): CPCV boxplot, 8 estrategias
# ------------------------------------------------------------------
print("=== 6. Regenerando Figura 6 (CPCV, 8 estrategias) ===")
df_cpcv_main = pd.read_csv(os.path.join(DATOS, "cpcv_sharpe_distribucion.csv"), index_col=0)
df_cpcv_ext = pd.read_csv(os.path.join(DATOS, "cpcv_sharpe_distribucion_extension.csv"), index_col=0)
assert len(df_cpcv_main) == len(df_cpcv_ext), "Distinta cantidad de combinaciones CPCV entre panel principal y extension"
df_cpcv_todos = df_cpcv_main.join(df_cpcv_ext)


def generar_figura_6_extendida(df_cpcv):
    fig, ax = plt.subplots(figsize=(13.5, 6.6))

    orden_estrategias = ["Equiponderado", "Referencia_60_40", "Markowitz", "Min_CVaR", "Min_CDaR", "HERC", "DRO_Wasserstein", "Min_CVaR_ExAnte"]
    paleta_colores = [PALETA_MODELOS.get(m, "#333333") for m in orden_estrategias]
    etiquetas_mostradas = [
        "1/N\nEquiponderado",
        "Referencia 60/40\n(SPY / IEF)",
        "Markowitz\n(MVO Clásico)",
        "Mínimo CVaR\n(95% Confianza)",
        "Mínimo CDaR\n(95% Confianza)",
        "HERC\n(Paridad Jerárquica)",
        "DRO-Wasserstein\n($\\varepsilon=0.02$)",
        "Mín. CVaR Ex-Ante\n(GARCH-EVT-Cópula)",
    ]

    sns.boxplot(
        data=df_cpcv[orden_estrategias], palette=paleta_colores, ax=ax, width=0.42,
        boxprops=dict(alpha=0.82, edgecolor="#0F172A", linewidth=1.2),
        whiskerprops=dict(color="#0F172A", linewidth=1.2),
        capprops=dict(color="#0F172A", linewidth=1.2),
        medianprops=dict(color="#FFFFFF", linewidth=2.2),
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="#000000", markerfacecolor="#FFFFFF", markersize=6.5, zorder=4),
        zorder=2
    )
    sns.stripplot(
        data=df_cpcv[orden_estrategias], color="#0F2942", alpha=0.55, size=5.0, jitter=0.10,
        edgecolor="#FFFFFF", linewidth=0.5, ax=ax, zorder=3
    )

    val_max = df_cpcv[orden_estrategias].max().max()
    val_min = df_cpcv[orden_estrategias].min().min()
    y_badge = val_max + 0.08
    for i, col_name in enumerate(orden_estrategias):
        med = float(df_cpcv[col_name].median())
        q75, q25 = np.percentile(df_cpcv[col_name].dropna(), [75, 25])
        iqr = q75 - q25
        ax.text(
            i, y_badge, f"Med: {med:.2f}\nIQR: {iqr:.2f}",
            ha="center", va="bottom", fontsize=7.5, color="#1E293B", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.92, lw=0.7),
            zorder=5
        )

    ax.set_xticks(range(len(orden_estrategias)))
    ax.set_xticklabels(etiquetas_mostradas, rotation=0, ha="center", fontsize=8.4)
    ax.set_title("Distribución Fuera de Muestra del Ratio de Sharpe (CPCV con 10 Días de Embargo, $N=6, k=2$)", pad=24, fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Ratio de Sharpe Anualizado ($R_f = 2.0\\%$)", fontsize=9.5)
    ax.axhline(0, color="#64748B", linestyle="--", lw=1.0, alpha=0.7, label="Breakeven ($SR=0$)")
    ax.axhline(0.50, color="#1E3A8A", linestyle=":", lw=1.1, alpha=0.7, label="Umbral Institucional ($SR=0.50$)")
    ax.set_ylim(min(val_min - 0.15, -0.20), y_badge + 0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS, "fig06_cpcv_distribucion_sharpe.png"), dpi=300)
    plt.close()


generar_figura_6_extendida(df_cpcv_todos)
print("  fig06 regenerada.")

# ------------------------------------------------------------------
# 7. FIGURA 7 (regenerada): descomposicion de riesgo, 4 paneles
# ------------------------------------------------------------------
print("=== 7. Regenerando Figura 7 (descomposicion de riesgo, 4 paneles) ===")


def generar_figura_7_extendida(pesos_dict, matriz_cov, tickers):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharey=True)
    estrategias_comparadas = [
        ("Markowitz", "A. Markowitz MVO (Concentración Crítica de Riesgo)", axes[0, 0]),
        ("HERC", "B. HERC (Asignación Jerárquica de Riesgo Equitativo)", axes[0, 1]),
        ("DRO_Wasserstein", "C. DRO-Wasserstein ($\\varepsilon=0.02$)", axes[1, 0]),
        ("Min_CVaR_ExAnte", "D. Mínimo CVaR Ex-Ante (GARCH-EVT-Cópula $t$)", axes[1, 1]),
    ]

    max_global = 0.0
    for m, _, _ in estrategias_comparadas:
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

        ax.bar(posiciones_x - ancho_barra / 2, w * 100.0, ancho_barra,
               label="Ponderación de Capital (% $w_i$)", color="#64748B", alpha=0.85, edgecolor="#334155", linewidth=0.8)
        ax.bar(posiciones_x + ancho_barra / 2, porcentaje_rc, ancho_barra,
               label="Contribución al Riesgo (% $RC_i$)", color=PALETA_MODELOS.get(m, "#333333"), alpha=0.85, edgecolor="#1E293B", linewidth=0.8)

        ax.set_title(titulo, pad=12, fontweight="bold", fontsize=11.5)
        ax.set_xticks(posiciones_x)
        ax.set_xticklabels(tickers, rotation=0, fontsize=8.6)
        ax.set_ylabel("Porcentaje (%)", fontsize=9.5)
        ax.set_ylim(0, limite_y)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=7.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS, "fig07_descomposicion_riesgo_cluster.png"), dpi=300)
    plt.close()


generar_figura_7_extendida(pesos_estaticos_todos, matriz_cov, TICKERS)
print("  fig07 regenerada.")

print("\n=== TODO OK ===")
