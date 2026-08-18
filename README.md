# Optimización de Portafolios: Markowitz, HERC, GARCH-EVT y Backtesting

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
│   └── Comparativa_Modelos_Asignacion_Activos.docx    # Fuente DOCX 1:1 — el .tex generador ha sido eliminado del repo
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

## 3. Fortalezas y Limitaciones por Modelo

> **Nota metodológica:** No existe un modelo universalmente dominante. Cada estrategia representa un balance explícito entre retorno, riesgo, estabilidad de pesos y requerimientos estadísticos. La comparación honesta exige evaluar cada dimensión por separado.

### 3.1. Equiponderado (1/N)

**Fortalezas:**
- Máxima diversificación ingenua: $N_{eff} = 9.00$, el mayor del universo evaluado.
- Sin parámetros que estimar: inmune al *error de estimación* y al overfitting en muestra.
- DeMiguel, Garlappi & Uppal (2009) demostraron que 1/N supera a Markowitz en retorno ajustado por riesgo en la mayoría de datasets reales con ventanas de estimación inferiores a 50 años.
- Costo operativo nulo (rotación = 0%).

**Limitaciones:**
- Ignora completamente la estructura de correlaciones entre activos.
- No diferencia entre activos de alto y bajo riesgo: asigna igual peso a VNQ (vol. 29.6%) y a IEF (vol. 6.9%).
- Sharpe de 0.62 es el piso de referencia para comparar modelos "sofisticados".

---

### 3.2. Referencia 60/40 (Benchmark Institucional)

**Fortalezas:**
- **Mejor CAGR neto (9.64%) y mejor retorno acumulado (358.51%)** del universo comparado, superando a HERC, 1/N y Min-CVaR/CDaR.
- El mejor Sharpe (0.78), Sortino (0.74) y Calmar (0.46), y el DSR más alto (99.64%).
- Regla simple, transparente, de bajo costo, ampliamente adoptada en el mundo institucional.
- La exposición concentrada en RV (60% en SPY+QQQ) captura la prima de riesgo de equity en el período 2010-2026, que fue inusualmente favorable.

**Limitaciones:**
- Concentración extrema: $N_{eff} = 1.92$. En la práctica, es casi un portafolio de dos activos.
- Max DD de -21.18%, comparable al HERC y al equiponderado.
- **Sus resultados dependen críticamente del período:** el ciclo 2010-2026 estuvo dominado por tasas reales bajas y expansión tecnológica. En un régimen diferente (e.g., estanflación sostenida), el 60/40 pierde su ventaja de retorno sin reducir su riesgo de drawdown.
- No incorpora ningún diagnóstico estadístico ni modelización de colas.

**El 60/40 es el benchmark contra el que todo modelo cuantitativo debe justificar su complejidad.**

---

### 3.3. Markowitz MVO (Media-Varianza)

**Fortalezas:**
- **Mayor retorno absoluto del universo cuantitativo: CAGR 11.39%, retorno acumulado 495.64%.**
- Fundamento teórico riguroso (Markowitz, 1952): maximiza el retorno por unidad de varianza bajo el supuesto de normalidad.
- Útil en contextos con estimaciones de parámetros estables y horizontes cortos.

**Limitaciones críticas:**
- **El mejor retorno viene con el peor perfil de riesgo:** vol. anual de 17.05%, Max DD de -39.32%, CVaR 99% EVT de -3.56%.
- Sharpe de 0.55: el más bajo de todos los modelos. El retorno extra no compensa el riesgo asumido.
- Rotación anual de 288.75%: destruye valor por costos de transacción en implementación real.
- $N_{eff} = 1.31$: concentración extrema. Degenerará hacia activos de máximo retorno histórico.
- Sufrirá la "maldición de la estimación": pequeños errores en $\hat{\mu}$ se amplifican exponencialmente en las ponderaciones (Michaud, 1989). Los parámetros del período 2007-2010 (crisis) contaminan sistemáticamente las ventanas de entrenamiento.

> **Conclusión sobre Markowitz:** el retorno más alto del período analizado no se debe a una ventaja estructural del modelo, sino a que la concentración en RV tecnológica (QQQ/SPY) capturó el bull market post-2012. En un entorno diferente, o con otro universo de activos, esta concentración sería su mayor debilidad.

---

### 3.4. Markowitz Remuestreado / Michaud (1989)

**Fortalezas:**
- Incorpora incertidumbre de estimación mediante simulación de Montecarlo: promedia sobre 500 fronteras simuladas en lugar de optimizar sobre una sola.
- Mayor estabilidad de pesos que el MVO clásico y menor sensibilidad a outliers en $\hat{\mu}$.
- $N_{eff} = 2.65$: mejor diversificación que el Markowitz puro.

**Limitaciones:**
- La mejora de estabilidad tiene costo: Max DD de -28.48% y rotación de 503.76% (la más alta del universo, ya que el proceso de bootstrap genera pesos muy variables entre ventanas).
- No resuelve el problema raíz: sigue dependiendo de estimaciones de retorno esperado, que son notoriamente ruidosas (Merton, 1980).

---

### 3.5. Mínimo CVaR (95%) y Mínimo CDaR (95%)

**Fortalezas:**
- **Perfil de riesgo extremo más bajo del universo:** Min-CVaR logra el menor Max DD (-16.29%), menor VaR diario (-0.52%) y CVaR EVT más bajo (-1.05%).
- CVaR es una medida de riesgo coherente (Artzner et al., 1999): cumple subaditividad, algo que la varianza no garantiza en distribuciones asimétricas.
- Min-CDaR optimiza directamente el drawdown promedio de las peores trayectorias (Chekhlov et al., 2005), que es la métrica más relevante para inversores con restricciones de pérdida máxima (e.g., mandatos de preservación de capital).
- Ambos modelos son apropiados para contextos regulatorios con límites de pérdida absoluta (e.g., fondos de pensiones bajo Solvencia II).

**Limitaciones:**
- **CAGR muy bajo (5.11% / 5.15%):** el costo de minimizar el riesgo de cola es renunciar a retorno. Con Rf = 2%, el exceso de retorno sobre la tasa libre de riesgo es de apenas 3%.
- El retorno acumulado (127-129%) está por debajo del benchmark 60/40 (358%) y del propio equiponderado (249%).
- La cartera concentra capital en los activos de menor volatilidad (IEF, LQD) en detrimento de diversificación real.

---

### 3.6. HERC — Hierarchical Equal Risk Contribution (Raffinot, 2017, 2018)

**Fortalezas:**
- **Balance entre diversificación real y control de riesgo:** $N_{eff} = 6.46$, el más alto dentro de los modelos cuantitativos con estimación de parámetros.
- No requiere estimación de retornos esperados (solo covarianza regularizada Ledoit-Wolf): elimina la fuente de error más ruidosa en la optimización clásica.
- Rotación mínima (31.43%): bajo impacto de costos de transacción.
- Mejor Sharpe ajustado estadísticamente: PSR = 99.58% y DSR = 98.49% son los más sólidos del universo cuantitativo.
- Distribución de riesgo entre clústeres jerárquicos: captura la estructura de correlación del mercado sin forzar concentraciones artificiales.

**Lo que HERC NO es:**
- **No maximiza el retorno absoluto.** HERC (CAGR 6.81%, acumulado 197.22%) está por debajo del 60/40, del equiponderado y de Markowitz MVO en retorno bruto. Quien elija HERC por el retorno se equivoca de modelo.
- **No minimiza el drawdown.** Min-CVaR y Min-CDaR controlan mejor el riesgo de cola.
- La ventaja de HERC es posicionarse en el *espacio de trade-off* de forma eficiente: retorno moderado, drawdown moderado, alta diversificación, baja rotación y robustez estadística. Es el modelo que mejor equilibra todos los objetivos simultáneamente, sin destacar en ninguno individualmente.

---

## 4. ¿Por Qué Optimizar Drawdown y No la Tasa de Ganancia? El Argumento Asimétrico

La intuición popular asocia un "buen" portafolio con alta tasa de aciertos (frecuencia de días positivos). Esta idea es matemáticamente errónea y costosa. El argumento central proviene de dos fuentes:

### 4.1. La Aritmética Asimétrica de las Pérdidas

Una pérdida del X% requiere una ganancia de $\frac{X}{1-X} \times 100\%$ para recuperar el nivel inicial:

| Pérdida | Ganancia necesaria para recuperar |
| :---: | :---: |
| -10% | +11.1% |
| -20% | +25.0% |
| -33% | +50.0% |
| -39% (Markowitz Max DD) | **+64.0%** |
| -50% | +100.0% |

Este efecto multiplicativo es exactamente lo que formaliza **Magdon-Ismail & Atiya (2004)** en su caracterización del drawdown máximo esperado como función del Sharpe ratio y el horizonte temporal. Un portafolio con Sharpe de 0.55 y Max DD de -39% (Markowitz) necesita ~64% de ganancia para volver al high-water mark. En ese tiempo, el capital no genera retorno compuesto: el tiempo de recuperación *destruye* el retorno logarítmico acumulado.

### 4.2. El Cisne Negro y la Asimetría del Período de Recuperación

Ante un evento de cola (COVID-2020, 2022 estanflacionario), los modelos con mayor drawdown quedan "bajo el agua" por períodos más largos, durante los cuales el capital no se compone:

- **Markowitz MVO:** Max DD de -39.32% → el capital necesita ~2-3 años de retornos normales solo para recuperar el high-water mark.
- **HERC:** Max DD de -21.11% → recuperación más corta, el capital vuelve a componer antes.
- **Min-CVaR:** Max DD de -16.29% → la menor interrupción del compounding.

El retorno acumulado final puede ser más alto en Markowitz, pero esta ventaja se debe al período post-2012 de bull market continuo. En presencia de dos o más eventos de cola en el horizonte, el drawdown más profundo destruye más valor mediante la interrupción del compounding que lo que el mayor CAGR en períodos tranquilos puede recuperar.

Este mecanismo está formalizado por **Chekhlov, Uryasev & Zabarankin (2005):** el CDaR captura exactamente esta métrica — el promedio de los peores trayectos bajo el agua — y optimizarla directamente es la forma correcta de proteger la trayectoria de riqueza acumulada, no solo el retorno final.

### 4.3. Drawdown vs. Tasa de Ganancia: El Paper Clave

> **Vince, R. (1992). *The Mathematics of Money Management*.** John Wiley & Sons.

Ralph Vince demostró formalmente, mediante la teoría del *Optimal f* (extensión del criterio de Kelly), que la maximización del crecimiento geométrico del capital depende críticamente de limitar el drawdown, no de maximizar la frecuencia de operaciones ganadoras. Una estrategia con 60% de tasa de ganancia pero pérdidas grandes puede destruir más capital geométrico que una con 40% de ganancia y pérdidas pequeñas. La razón: en el crecimiento geométrico, las pérdidas computan de forma multiplicativa (no aditiva), por lo que asimetría en la magnitud de las pérdidas domina a la frecuencia de aciertos.

El resultado equivalente en finanzas modernas aparece en:
> **Grossman, S. J., & Zhou, Z. (1993).** Optimal investment strategies for controlling drawdowns. *Mathematical Finance*, 3(3), 241–276.

Grossman & Zhou prueban que bajo un mandato de preservación de capital (límite de drawdown), la solución óptima no es la cartera de media-varianza, sino una solución que gestiona activamente la distancia al high-water mark. Este resultado justifica directamente los modelos Min-CDaR y, en menor medida, HERC como alternativas superadoras de Markowitz cuando el mandato incluye restricciones de pérdida.

---

## 5. Comparación vs. Benchmark (60/40)

El benchmark de referencia institucional es la cartera 60/40 (60% renta variable / 40% renta fija). Todo modelo cuantitativo que no supere al 60/40 en alguna dimensión relevante no justifica su complejidad operativa ni sus costos de implementación.

| Dimensión | 60/40 | Markowitz MVO | Mín. CVaR | HERC | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :--- |
| CAGR (%) | **9.64** | 11.39 | 5.11 | 6.81 | 60/40 gana en retorno neto |
| Sharpe | **0.78** | 0.55 | 0.58 | 0.65 | 60/40 domina en eficiencia |
| Max DD (%) | -21.18 | -39.32 | **-16.29** | -21.11 | Min-CVaR gana en protección |
| CVaR 99% EVT | -1.98 | -3.56 | **-1.05** | -1.53 | Min-CVaR/HERC ganan en cola |
| Rotación (%) | **0** | 288.75 | 66.37 | **31.43** | 60/40 y HERC ganan en costo |
| $N_{eff}$ | 1.92 | 1.31 | 1.84 | **6.46** | HERC gana en diversificación |
| DSR (%) | **99.64** | 95.33 | 97.24 | 98.49 | 60/40 lidera en significancia |

### Lectura honesta de la tabla:

1. **El 60/40 no es superable en retorno ajustado por riesgo** en el período 2010-2026. Esto es un resultado del régimen histórico, no una ley general.
2. **HERC supera al 60/40 en diversificación real, CVaR de cola y rotación.** Para mandatos que requieren preservación de capital ante cisnes negros o límites regulatorios de pérdida, HERC domina.
3. **Markowitz domina en retorno absoluto pero con el peor perfil de riesgo.** No supera al 60/40 en Sharpe ni en drawdown. Solo se justifica cuando el mandato es maximizar retorno sin restricciones de riesgo, lo cual es inusual en gestión institucional.
4. **Min-CVaR/CDaR son los únicos modelos que mejoran al 60/40 en protección de colas.** Se justifican cuando el mandato tiene restricciones duras de pérdida máxima (fondos de pensiones, mandatos de capital garantizado).

---

## 6. Resultados Empíricos Fuera de Muestra (2010–2026)

Análisis Walk-Forward mensual fuera de muestra (ventana de entrenamiento rodante de 3 años / 756 ruedas) con deriva de ponderaciones (*weight drift*) y costo operativo de 5 puntos básicos por rotación:

| Indicador / Métrica | Equiponderado (1/N) | Referencia 60/40 | Markowitz MVO | Markowitz Remuestreado | Mínimo CVaR (95%) | Mínimo CDaR (95%) | HERC (Ledoit-Wolf) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAGR (%)** | 7.85 | **9.64** | 11.39 | 10.48 | 5.11 | 5.15 | 6.81 |
| **Retorno Acumulado (%)** | 249.17 | **358.51** | 495.64 | 419.94 | 127.92 | 129.35 | 197.22 |
| **Volatilidad Anual (%)** | 9.49 | 9.79 | 17.05 | 13.61 | **5.32** | 6.38 | 7.35 |
| **Desviación a la Baja (%)** | 10.10 | 10.34 | 18.60 | 14.76 | **5.54** | 6.73 | 7.82 |
| **Ratio de Sharpe (Rf=2%)** | 0.62 | **0.78** | 0.55 | 0.62 | 0.58 | 0.49 | 0.65 |
| **Ratio de Sortino** | 0.58 | **0.74** | 0.51 | 0.57 | 0.56 | 0.47 | 0.61 |
| **Ratio de Calmar** | 0.38 | **0.46** | 0.29 | 0.37 | 0.31 | 0.27 | 0.32 |
| **Ratio de Omega** | 1.12 | **1.16** | 1.11 | 1.13 | 1.11 | 1.10 | 1.13 |
| **Ratio Ganancia/Pérdida** | 1.16 | **1.20** | 1.14 | 1.15 | 1.18 | 1.16 | 1.18 |
| **PSR (vs 0) (%)** | 99.34 | **99.92** | 98.67 | 99.39 | 99.09 | 97.70 | 99.58 |
| **DSR (%)** | 95.45 | **99.64** | 95.33 | 97.90 | 97.24 | 93.29 | 98.49 |
| **Retroceso Máximo / Max DD (%)** | -20.86 | -21.18 | -39.32 | -28.48 | **-16.29** | -19.39 | -21.11 |
| **CDaR 95% (%)** | -14.77 | -15.52 | -35.61 | -25.49 | **-12.77** | -15.81 | -15.35 |
| **VaR 95% Diario (%)** | -0.90 | -0.94 | -1.74 | -1.41 | **-0.52** | -0.61 | -0.72 |
| **CVaR 95% Diario (%)** | -1.40 | -1.47 | -2.71 | -2.12 | **-0.79** | -0.97 | -1.09 |
| **VaR 99% GARCH-EVT (%)** | -1.54 | -1.56 | -2.87 | -2.28 | **-0.83** | -0.99 | -1.17 |
| **CVaR 99% GARCH-EVT (%)** | -1.96 | -1.98 | -3.56 | -2.83 | **-1.05** | -1.25 | -1.53 |
| **Coeficiente de Asimetría** | -0.55 | -0.31 | -0.53 | -0.52 | **-0.17** | -0.29 | -0.37 |
| **Curtosis de Exceso** | 9.10 | 9.74 | 4.09 | 4.89 | **3.69** | 5.55 | 7.75 |
| **Rotación Anualizada (%)** | **0.00** | **0.00** | 288.75 | 503.76 | 66.37 | 261.02 | 31.43 |
| **Nº Efectivo de Activos ($N_{eff}$)** | **9.00** | 1.92 | 1.31 | 2.65 | 1.84 | 2.16 | 6.46 |

> Los valores en **negrita** indican el modelo ganador en esa dimensión específica. No existe un único ganador global.

---

## 7. Análisis de Eventos de Estrés

| Evento | Markowitz MVO | 60/40 | HERC | Min-CVaR |
| :--- | :---: | :---: | :---: | :---: |
| Crisis Deuda Europea (2011) | Mayor impacto por concentración RV | Moderado por diversif. RF | Contenido por clústeres | Mínimo |
| COVID-19 (Feb-Mar 2020) | DD estimado ~-35% local | DD ~-18% local | DD ~-16% local | DD ~-12% local |
| Shock estanflacionario 2022 | Caída RV + caída RF simultánea | Golpe doble por 60/40 | Mejor diversif. entre clases | Menor exposición |
| **Tiempo estimado de recuperación** | **Largo (>18 meses post-DD máximo)** | Moderado | Moderado | **Corto** |

**El argumento del cisne negro en términos de compounding:** con Max DD de -39%, Markowitz necesita generar aproximadamente un 64% de retorno bruto solo para recuperar el capital perdido. Durante ese tiempo, el capital no se compone hacia adelante. HERC, con -21%, necesita solo un 27% para recuperar, acortando drásticamente el período improductivo y preservando la trayectoria de crecimiento compuesto a largo plazo.

---

## 8. Ejecución del Pipeline

```bash
python 03_Codigo/pipeline_optimizacion.py
```

---

## 9. Bibliografía

1. **Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999).** Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
2. **Balkema, A. A., & de Haan, L. (1974).** Residual life time at great age. *The Annals of Probability*, 2(5), 792–804.
3. **Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005).** Drawdown measure in portfolio optimization. *International Journal of Theoretical and Applied Finance*, 8(01), 13–58.
4. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *The Review of Financial Studies*, 22(5), 1915–1953.
5. **Engle, R. F. (1982).** Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
6. **Grossman, S. J., & Zhou, Z. (1993).** Optimal investment strategies for controlling drawdowns. *Mathematical Finance*, 3(3), 241–276.
7. **Jarque, C. M., & Bera, A. K. (1980).** Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters*, 6(3), 255–259.
8. **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
9. **López de Prado, M. (2014).** The Deflated Sharpe Ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107.
10. **López de Prado, M. (2016).** Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management*, 42(4), 59–69.
11. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons.
12. **Magdon-Ismail, M., & Atiya, A. F. (2004).** Maximum drawdown. *Risk Magazine*, 17(10), 99–102.
13. **Markowitz, H. (1952).** Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
14. **McNeil, A. J., & Frey, R. (2000).** Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *Journal of Empirical Finance*, 7(3-4), 271–300.
15. **McNeil, A. J., Frey, R., & Embrechts, P. (2015).** *Quantitative Risk Management: Concepts, Techniques and Tools* (Revised Edition). Princeton University Press.
16. **Merton, R. C. (1980).** On estimating the expected return on the market: An exploratory investigation. *Journal of Financial Economics*, 8(4), 323–361.
17. **Michaud, R. O. (1989).** The Markowitz optimization enigma: Is 'optimized' optimal? *Financial Analysts Journal*, 45(1), 31–42.
18. **Patton, A. J. (2012).** A review of copula models for economic time series. *Handbook of Economic Forecasting*, 2, 89–145.
19. **Pickands, J. (1975).** Statistical inference using extreme order statistics. *The Annals of Statistics*, 3(1), 119–131.
20. **Raffinot, T. (2017).** Hierarchical clustering-based asset allocation. *The Journal of Portfolio Management*, 44(2), 89–99.
21. **Raffinot, T. (2018).** The hierarchical equal risk contribution portfolio. *Financial Markets and Portfolio Management*, 32(2), 241–261.
22. **Rockafellar, R. T., & Uryasev, S. (2000).** Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–42.
23. **Rockafellar, R. T., & Uryasev, S. (2002).** Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance*, 26(7), 1443–1471.
24. **Vince, R. (1992).** *The Mathematics of Money Management: Risk Analysis Techniques for Traders*. John Wiley & Sons.
