# Optimización de Portafolios: Markowitz, HERC, Modelización de Riesgos y Backtesting

Evaluación cuantitativa, diagnóstico econométrico y contrastación empírica de modelos de asignación de activos sobre un universo multiactivo líquido (2007–2026).

**Autor:** Federico Agustín Chillón  
**Investigación Cuantitativa y Gestión de Portafolios**  
**Fecha:** Agosto 2026  
**Clasificación JEL:** C58, C61, G11, G14  
**MSC (2020):** 91G10, 90C90, 62P05  

---

## 1. Estructura del Proyecto

```text
01_Optimizacion_Portafolios/
├── 01_Reporte_PDF/
│   ├── Comparativa_Modelos_Asignacion_Activos.pdf     # Documento técnico en formato PDF (Georgia, 14 págs.)
│   └── Comparativa_Modelos_Asignacion_Activos.tex     # Código fuente en LaTeX con todas las secciones y tablas
├── 02_Documento_Word/
│   └── Comparativa_Modelos_Asignacion_Activos.docx    # Documento sincronizado en formato Word (.docx)
├── 03_Codigo/
│   └── pipeline_optimizacion.py                       # Pipeline cuantitativo con diagnóstico econométrico y GARCH-EVT
├── 04_Figuras/
│   ├── fig01_dendrograma_correlaciones.png            # Agrupamiento jerárquico y correlación reordenada
│   ├── fig02_frontera_eficiente_modelos.png           # Frontera analítica, activos y cono de incertidumbre bootstrap
│   ├── fig03_evolucion_pesos_walk_forward.png         # Evolución temporal de ponderaciones en Walk-Forward
│   ├── fig04_retorno_acumulado_y_drawdown.png         # Curvas de riqueza y retroceso con regímenes de crisis
│   ├── fig05_ajuste_evt_pareto_colas.png              # Densidad empírica y ajuste GARCH-EVT (McNeil & Frey, 2000)
│   ├── fig06_cpcv_distribucion_sharpe.png             # Validación cruzada combinatoria con embargo (CPCV)
│   └── fig07_descomposicion_riesgo_cluster.png        # Ponderación de capital vs Contribución al riesgo (%RC)
├── 05_Datos_y_Resultados/
│   ├── precios_ajustados.csv                          # Precios limpios y ajustados por dividendos
│   ├── tabla_diagnostico_econometrico.csv             # Contrastes de hipótesis (ADF, JB, Ljung-Box, ARCH-LM)
│   ├── metricas_comparativas.csv                      # Tabla consolidada de métricas fuera de muestra
│   ├── retornos_diarios_oos.csv                       # Rendimientos diarios fuera de muestra
│   ├── cpcv_sharpe_distribucion.csv                   # Distribución de Sharpe en CPCV
│   └── pesos_historicos_*.csv                         # Ponderaciones históricas por estrategia
└── README.md
```

---

## 2. Diagnóstico Econométrico y Validación de Supuestos (2007–2026)

Se aplica una batería de contrastes formales sobre las 4.928 ruedas históricas para verificar la validez de los supuestos clásicos de la teoría financiera:

| Ticker | Clase de Activo | Ret. Anual (%) | Vol. Anual (%) | Asimetría | Exceso Curtosis | Test Jarque-Bera ($p$-val) | Ljung-Box $Q(10)$ ($p$-val $r_t$) | ARCH-LM $Q(10)$ ($p$-val $r_t^2$) | ADF Retornos ($p$-val) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPY** | RV EE.UU. (S&P 500) | 10.35 | 19.63 | -0.30 | +13.92 | $< 0.0001$ | $< 10^{-14}$ | $< 0.0001$ | $< 10^{-27}$ |
| **QQQ** | RV Tecnológica (Nasdaq) | 14.95 | 22.26 | -0.26 | +7.29 | $< 0.0001$ | $< 10^{-13}$ | $< 0.0001$ | $< 10^{-29}$ |
| **EEM** | RV Mercados Emergentes | 4.68 | 27.99 | +0.03 | +16.12 | $< 0.0001$ | $< 10^{-15}$ | $< 0.0001$ | $< 10^{-23}$ |
| **VNQ** | Bienes Raíces (REITs) | 5.42 | 29.58 | -0.52 | +17.87 | $< 0.0001$ | $< 10^{-15}$ | $< 0.0001$ | $< 10^{-25}$ |
| **TLT** | Renta Fija Soberana (20Y+) | 2.71 | 15.05 | +0.01 | +3.20 | $< 0.0001$ | $< 10^{-5}$ | $< 0.0001$ | $< 10^{-24}$ |
| **IEF** | Renta Fija Soberana (7-10Y)| 3.14 | 6.92 | +0.12 | +2.59 | $< 0.0001$ | 0.0048 | $< 0.0001$ | $< 10^{-30}$ |
| **LQD** | Crédito Corporativo IG | 3.94 | 8.79 | -0.45 | +56.88 | $< 0.0001$ | $< 10^{-15}$ | $< 0.0001$ | $< 10^{-24}$ |
| **GLD** | Metales Preciosos (Oro) | 9.14 | 18.15 | -0.43 | +7.08 | $< 0.0001$ | 0.4678 | $< 0.0001$ | $< 10^{-30}$ |
| **DBC** | Materias Primas | 2.15 | 19.35 | -0.49 | +3.35 | $< 0.0001$ | 0.9888 | $< 0.0001$ | $< 10^{-30}$ |

### Conclusiones del Diagnóstico:
1. **Rechazo de Normalidad:** El test de Jarque-Bera rechaza la hipótesis nula gaussiana ($p < 0.0001$). Las colas pesadas (curtosis hasta $+56.88$) y la asimetría negativa invalidan la premisa de que la varianza es una medida completa de riesgo.
2. **Rechazo de $i.i.d.$ en Varianza (Heterocedasticidad / Efectos ARCH):** El test $Q(10)$ en $r_t^2$ da $p < 0.0001$ en todos los activos. Se confirma la existencia de agrupamientos de volatilidad (*volatility clustering*).
3. **Estacionariedad:** Los precios son $I(1)$, mientras que los rendimientos logarítmicos son estrictamente estacionarios $I(0)$ ($p < 10^{-20}$ en ADF).
4. **Cumplimiento de $i.i.d.$ en EVT:** Para cumplir el Teorema de Pickands-Balkema-de Haan en la cuantificación de colas, se aplica la metodología de dos etapas de **McNeil & Frey (2000)** con filtrado AR(1)-GARCH(1,1). Los residuos estandarizados $z_t$ purgan los efectos ARCH ($p = 0.1301 > 0.05$), validando la condición $i.i.d.$

---

## 3. Resultados Empíricos Fuera de Muestra (2010–2026)

Análisis Walk-Forward mensual fuera de muestra (ventana de entrenamiento rodante de 3 años / 756 ruedas) con deriva de ponderaciones (*weight drift*) y costo operativo de 5 puntos básicos por rotación:

| Indicador / Métrica | Equiponderado (1/N) | Referencia 60/40 | Markowitz MVO | Markowitz Remuestreado | Mínimo CVaR (95%) | Mínimo CDaR (95%) | HERC (Ledoit-Wolf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAGR (%)** | 7.85 | 9.64 | 11.39 | 10.48 | 5.11 | 5.15 | **6.81** |
| **Retorno Acumulado (%)** | 249.17 | 358.51 | 495.64 | 419.94 | 127.92 | 129.35 | **197.22** |
| **Volatilidad Anual (%)** | 9.49 | 9.79 | 17.05 | 13.61 | 5.32 | 6.38 | **7.35** |
| **Desviación a la Baja (%)** | 10.10 | 10.34 | 18.60 | 14.76 | 5.54 | 6.73 | **7.82** |
| **Ratio de Sharpe (Rf=2%)** | 0.62 | 0.78 | 0.55 | 0.62 | 0.58 | 0.49 | **0.65** |
| **Ratio de Sortino** | 0.58 | 0.74 | 0.51 | 0.57 | 0.56 | 0.47 | **0.61** |
| **Ratio de Calmar** | 0.38 | 0.46 | 0.29 | 0.37 | 0.31 | 0.27 | **0.32** |
| **Ratio de Omega** | 1.12 | 1.16 | 1.11 | 1.13 | 1.11 | 1.10 | **1.13** |
| **Ratio Ganancia/Pérdida** | 1.16 | 1.20 | 1.14 | 1.15 | 1.18 | 1.16 | **1.18** |
| **PSR (vs 0) (%)** | 99.34 | 99.92 | 98.67 | 99.39 | 99.09 | 97.70 | **99.58** |
| **DSR (%)** | 95.45 | 99.64 | 95.33 | 97.90 | 97.24 | 93.29 | **98.49** |
| **Retroceso Máximo / Max DD (%)** | -20.86 | -21.18 | -39.32 | -28.48 | -16.29 | -19.39 | **-21.11** |
| **CDaR 95% (%)** | -14.77 | -15.52 | -35.61 | -25.49 | -12.77 | -15.81 | **-15.35** |
| **VaR 95% Diario (%)** | -0.90 | -0.94 | -1.74 | -1.41 | -0.52 | -0.61 | **-0.72** |
| **CVaR 95% Diario (%)** | -1.40 | -1.47 | -2.71 | -2.12 | -0.79 | -0.97 | **-1.09** |
| **VaR 99% GARCH-EVT (%)** | -1.54 | -1.56 | -2.87 | -2.28 | -0.83 | -0.99 | **-1.17** |
| **CVaR 99% GARCH-EVT (%)** | -1.96 | -1.98 | -3.56 | -2.83 | -1.05 | -1.25 | **-1.53** |
| **Coeficiente de Asimetría** | -0.55 | -0.31 | -0.53 | -0.52 | -0.17 | -0.29 | **-0.37** |
| **Curtosis de Exceso** | 9.10 | 9.74 | 4.09 | 4.89 | 3.69 | 5.55 | **7.75** |
| **Rotación Anualizada (%)** | 0.00 | 0.00 | 288.75 | 503.76 | 66.37 | 261.02 | **31.43** |
| **Nº Efectivo de Activos ($N_{eff}$)** | 9.00 | 1.92 | 1.31 | 2.65 | 1.84 | 2.16 | **6.46** |

---

## 4. Conclusiones y Análisis de Estrés

1. **Comportamiento en Crisis:** Durante la Crisis de Deuda Europea (2011), el colapso por COVID-19 (2020) y el shock estanflacionario de 2022, HERC preservó el capital conteniendo el retroceso máximo en **-21.11%** (frente al **-39.32%** de Markowitz y **-28.48%** de Michaud).
2. **Eficiencia Operativa:** Markowitz clásico y remuestreado generan una rotación anual de **288.75%** y **503.76%**. HERC opera con solo **31.43%** de rotación anual, limitando el deterioro por comisiones y deslizamiento.
3. **Diversificación Real:** HERC mantiene un número efectivo de activos de **$N_{eff} = 6.46$** y supera los tests de significancia estadística con un **PSR de 99.58%** y un **DSR de 98.49%**, con un $\text{CVaR}_{99\%}^{\text{EVT}}$ de solo **-1.53%** frente al **-3.56%** de Markowitz.

---

## 5. Ejecución del Pipeline

```bash
python 03_Codigo/pipeline_optimizacion.py
```

---

## 6. Bibliografía

1. **Balkema, A. A., & de Haan, L. (1974).** Residual life time at great age. *The Annals of Probability*, 2(5), 792–804.
2. **Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005).** Drawdown measure in portfolio optimization. *International Journal of Theoretical and Applied Finance*, 8(01), 13–58.
3. **Engle, R. F. (1982).** Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
4. **Jarque, C. M., & Bera, A. K. (1980).** Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255–259.
5. **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
6. **López de Prado, M. (2014).** The Deflated Sharpe Ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107.
7. **López de Prado, M. (2016).** Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management*, 42(4), 59–69.
8. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons.
9. **Markowitz, H. (1952).** Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
10. **McNeil, A. J., & Frey, R. (2000).** Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance*, 7(3-4), 271–300.
11. **McNeil, A. J., Frey, R., & Embrechts, P. (2015).** *Quantitative Risk Management: Concepts, Techniques and Tools* (Revised Edition). Princeton University Press.
12. **Michaud, R. O. (1989).** The Markowitz optimization enigma: Is 'optimized' optimal? *Financial Analysts Journal*, 45(1), 31–42.
13. **Patton, A. J. (2012).** A review of copula models for economic time series. *Handbook of Economic Forecasting*, 2, 89–145.
14. **Pickands, J. (1975).** Statistical inference using extreme order statistics. *The Annals of Statistics*, 3(1), 119–131.
15. **Raffinot, T. (2017).** Hierarchical clustering-based asset allocation. *The Journal of Portfolio Management*, 44(2), 89–99.
16. **Raffinot, T. (2018).** The hierarchical equal risk contribution portfolio. *Financial Markets and Portfolio Management*, 32(2), 241–261.
17. **Rockafellar, R. T., & Uryasev, S. (2000).** Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–42.
18. **Rockafellar, R. T., & Uryasev, S. (2002).** Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
