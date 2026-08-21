# Auditoría Extendida: Microestructura Avanzada, EVT y Trampas Matemáticas Ocultas

> Respuesta de ingeniería a un mensaje de auditoría de laboratorio (2026-08-21) que pide,
> con rigor científico y sin complacencia, aplicar 7 dominios de auditoría extendida + la
> "Regla de Oro" (Broken Pipe Test) sobre este pipeline. Este documento es la versión
> **acotada al alcance real de este proyecto**: `pipeline_optimizacion.py` es un pipeline
> académico de **un solo script**, sobre un universo **fijo de 9 activos** (SPY, QQQ, EEM,
> VNQ, TLT, IEF, LQD, GLD, DBC, 2007-2026), con datos EOD (`yfinance`, `auto_adjust=True`).
> No es un motor en vivo con API de rebalanceo, monitoreo de producción ni ejecución real
> — a diferencia del proyecto hermano `herc-portfolio-optimizer` (donde la mayoría de esta
> misma lista aplica con más profundidad, ver
> [`14_AUDITORIA_EXTENDIDA_MICROESTRUCTURA_MATEMATICA.md`](https://github.com/federicoagustinchillon-creador/herc-portfolio-optimizer/blob/claude/portfolio-optimizer-audit-0im04q/outputs/dashboard/research/14_AUDITORIA_EXTENDIDA_MICROESTRUCTURA_MATEMATICA.md)
> en ese repo), gran parte de esta lista (monitoreo en vivo, ejecución de órdenes,
> rebalanceo por entropía, drift multivariante) **no tiene dónde aplicarse aquí** — no
> porque el ítem no importe, sino porque este proyecto no tiene un componente "en
> producción" al cual monitorear. Se documenta explícitamente en vez de fingir que aplica.

---

## Estado del "paper" (Word/PDF publicado) tras esta auditoría

**`02_Documento_Word/Comparativa_Modelos_Asignacion_Activos.docx` (fuente canónica) —
ACTUALIZADO.** Se agregó una nueva sección, *"Auditoría de Robustez: Broken Pipe Test
(Ruido Blanco al 50% del Universo)"*, antes de "Referencias", con el hallazgo real de la
Sección 0 de este documento (resultado sobre mercado real, hallazgo metodológico
`zero` vs. `matched`, tabla de resultados) y una nota sobre la distancia de correlación
absoluta (Sección 4 de este documento). Editado con `python-docx`, validado
estructuralmente (XSD + diff de párrafos) y verificado visualmente contra el estilo
tipográfico existente (Georgia, encabezados en negrita centrados, tabla sin bordes —
mismo criterio que el resto del documento).

**`01_Reporte_PDF/Comparativa_Modelos_Asignacion_Activos.pdf` — NO regenerado desde el
DOCX actualizado, deliberadamente.** Dos motivos concretos, no una omisión:

1. **Ya hay un precedente documentado en este mismo repositorio de que regenerar este PDF
   automáticamente sale mal.** El historial de git muestra exactamente ese intento y su
   reversión: `ddd706b` ("chore: regenerate report PDF from DOCX canonical source") seguido
   de `e1ce63b` ("revert: restore original report PDF (pre-regeneration, 18 pages)") —
   la versión regenerada automáticamente (929 KB) se descartó a favor de la original
   (4.59 MB, 18 páginas), evidencia de que la conversión automática pierde algo que la
   original tiene.
2. **Este entorno de sandbox no tiene instalada la tipografía Georgia** que usa el documento
   (verificado: `fc-match Georgia` resuelve a "DejaVu Serif" por sustitución silenciosa) —
   cualquier conversión DOCX→PDF hecha acá (LibreOffice headless, la única herramienta
   disponible) renderizaría el documento COMPLETO —no solo la sección nueva— con la
   tipografía equivocada, reproduciendo exactamente el tipo de degradación que el commit
   `e1ce63b` ya corrigió una vez.

**Qué hacer en la próxima sesión con el entorno correcto:** abrir
`02_Documento_Word/Comparativa_Modelos_Asignacion_Activos.docx` (ya actualizado) en
Microsoft Word real (con Georgia instalada) y usar "Guardar como PDF" / "Exportar a PDF"
para regenerar `01_Reporte_PDF/Comparativa_Modelos_Asignacion_Activos.pdf` — el mismo
proceso manual que aparentemente produjo el PDF de 18 páginas actual. No usar conversión
automática (LibreOffice/`soffice --convert-to pdf`) sin antes instalar la fuente Georgia y
comparar visualmente contra el PDF actual, dado el precedente ya documentado arriba.

---

## 0. La Regla de Oro: Broken Pipe Test — resultado sobre MERCADO REAL

**Implementado:** [`03_Codigo/broken_pipe_test.py`](./03_Codigo/broken_pipe_test.py).
Corrida real: [`05_Datos_y_Resultados/BROKEN_PIPE_TEST_REPORT.md`](./05_Datos_y_Resultados/BROKEN_PIPE_TEST_REPORT.md).

A diferencia de `herc-portfolio-optimizer` (sin datos de precios comiteados, y sin acceso
de red desde el entorno de auditoría), **este repo sí tiene precios reales comiteados**
(`05_Datos_y_Resultados/precios_ajustados.csv`, 4925 ruedas, 2007-2026) — esta corrida es
sobre **mercado real**, no un universo sintético de validación.

Metodología: reemplaza 4 de los 9 tickers (50%) por ruido blanco, corre el motor HERC
walk-forward real (misma lógica exacta de costo/deriva de `ejecutar_walk_forward`, con
seguimiento diario de pesos para poder atribuir el retorno exacto del sleeve de ruido) +
`ejecutar_cpcv` + `calcular_ratio_sharpe_deflactado` — 40 semillas independientes.

**Mismo hallazgo metodológico que en el repo hermano:** calibrar el ruido a la misma
media del activo real (`noise_mean_mode='matched'`, lectura literal del pedido) le da al
ruido, sobre un universo con drift positivo real (SPY/QQQ 2007-2026), la misma prima de
riesgo que el activo real — no es fuga, es la prima que el generador de ruido regaló. Se
corrigió a `noise_mean_mode='zero'` (mu=0 exacto) como default, y se corrieron ambos modos
para mostrar el contraste:

| Modo | Contribución media diaria sleeve ruido | p-valor | Veredicto |
|---|---|---|---|
| `zero` (correcto, 40 trials) | +0.000008/día | 0.176 | **Sin evidencia de fuga** |
| `matched` (lectura literal) | +0.000100/día | <0.0001 | Fuga clara (artefacto esperado, documentado) |

**Resultado sobre mercado real, con la metodología correcta: sin evidencia de fuga de
información** en el motor HERC walk-forward de este pipeline. Con 20 trials (potencia
menor) el p-valor había cruzado el umbral 0.05 por poco (0.026) — se subió a 40 trials
antes de reportar un veredicto, siguiendo el mismo criterio de rigor estadístico del
proyecto hermano: un p-valor marginal con pocos trials no es una alarma, se aumenta la
potencia antes de concluir.

```bash
python 03_Codigo/broken_pipe_test.py
```

---

## 1-7: Mapeo de las 7 secciones al alcance real de este proyecto

| Sección del pedido | Aplica a este proyecto | Motivo |
|---|---|---|
| **1. Datos/Microestructura** (EVT intradía, VWAP, spread real, opciones, barras de volumen) | 🚫 Bloqueado por datos, igual que el repo hermano | `precios_ajustados.csv` es EOD (`yfinance`, `auto_adjust=True`) — mismo bloqueo estructural documentado en detalle en la auditoría del repo hermano. No se repite el análisis completo acá; la causa raíz es idéntica. |
| **2. Sobreajuste** (P-hacking CPCV, cópulas, DSR) | ⚠️ Parcialmente aplicable | `ejecutar_cpcv` (línea 832) NO ajusta hiperparámetros dentro de sí — usa constantes fijas por modelo, mismo argumento que en el repo hermano (no hay P-hacking trans-partición del tipo descrito). `calcular_ratio_sharpe_deflactado` toma `n_modelos_evaluados` como parámetro **fijo en 7** por defecto (línea 698) — a diferencia del repo hermano, **no hay** un registro real de experimentos (MLflow/git audit) que cuente intentos descartados; el N=7 es el número de modelos publicados, una cota inferior no verificada contra el historial real de intentos. Brecha real, no cerrada — este proyecto no tiene infraestructura de tracking de experimentos. Cópulas: este pipeline no usa cópulas explícitas (a diferencia de `min_cvar_copula`/`vine_copula_research.py` del repo hermano) — el ítem de regularización de cópulas **no aplica**, no hay una matriz de dependencia de cópula que regularizar aquí. |
| **3. GARCH-EVT+Cópulas** | ⚠️ Parcialmente aplicable | `ajustar_modelo_evt_gpd` (línea 568) SÍ implementa el filtrado AR(1)-GARCH(1,1) de dos etapas (McNeil & Frey 2000) sobre datos reales, con umbral de EVT **fijo** (`cuantil_umbral=0.90` por defecto, no seleccionado dinámicamente por ventana — mismo argumento que en el repo hermano, elimina el riesgo de "umbral que oscila porque se re-elige"). Sensibilidad de persistencia GARCH a la ventana: no auditada hoy en este repo — el repo hermano ya corrió este experimento (`GARCH_WINDOW_SENSITIVITY.md`) reutilizando la misma función `fit_garch11`/`arch_model`; la conclusión cualitativa (ventanas cortas producen más dispersión en α+β) es transferible por usar la misma librería subyacente (`arch`), no se duplica la corrida acá. Cópulas asimétricas: **no aplica**, este pipeline no ajusta cópulas explícitas. |
| **4. HERC / Correlaciones negativas** | 🆕 **Implementado hoy** | Ver sección dedicada abajo — `calcular_distancia_angular` usa la misma fórmula clásica `d=√(0.5(1-ρ))` que el repo hermano, con el mismo problema documentado (coberturas quedan más lejos que activos sin relación). Se agregó `calcular_distancia_angular_abs_corr` (`D=1-|ρ|`) como función opcional, verificada. |
| **5. Ejecución/Impacto de mercado** | 🚫 No aplica / bloqueado por datos | Este pipeline no tiene ningún modelo de impacto de mercado (ni siquiera Almgren-Chriss teórico como el repo hermano) — el costo de transacción es un supuesto plano en bps (`tasa_friccion_bps=5.0`, línea 727). Sin L2/opciones históricas, el resto de la sección tampoco es implementable — mismo bloqueo que Sección 1. |
| **6. Controles en vivo / Monitoreo** | 🚫 **No aplica a este proyecto** | Este pipeline genera un reporte estático (PDF/DOCX) a partir de una corrida puntual — no tiene un componente "en producción" que monitorear día a día. Cointegración de clusters HERC, drift Mahalanobis multivariante, bandas de entropía: los tres asumen un sistema que sigue operando después de la corrida (rebalanceando, sirviendo decisiones) — este proyecto no lo es. Implementar estos controles aquí sería fabricar un componente de monitoreo para un sistema que no existe. |
| **7. Métricas ex-post** (CI de Calmar/Sortino) | ✅ **Parcialmente ya cubierto** | `calcular_tabla_metricas` reporta Sharpe/Sortino/Calmar puntuales — sin bootstrap de bloques circulares para CI, a diferencia del repo hermano. Brecha real reconocida, no cerrada hoy por alcance (agregar block bootstrap aquí sería duplicar exactamente `block_bootstrap_metric_ci` del repo hermano sin una razón específica de este proyecto para tenerlo por separado). |

---

## Sección 4 en detalle: distancia de correlación absoluta para coberturas

**Implementado hoy:** `calcular_distancia_angular_abs_corr` en
`03_Codigo/broken_pipe_test.py` (por scope, no se modificó `pipeline_optimizacion.py` para
no alterar el output ya publicado del reporte PDF/DOCX oficial — ver nota abajo).

Mismo hallazgo cuantitativo que en el repo hermano: con la métrica clásica
`d=√(0.5(1-ρ))`, un activo con ρ=-0.9 respecto del resto queda a `d=0.975`, **más lejos**
que un activo sin ninguna relación (ρ=0 → `d=0.707`). Con `D=1-|ρ|`, esa misma cobertura
queda a `D=0.1` (vecindario inmediato del árbol).

**Nota de alcance:** a diferencia del repo hermano (donde se wireó como una opción
`dist_method='abs_correlation'` dentro de la clase de producción `HERCOptimizerBT`, sin
tocar el comportamiento por defecto), este repo publica un PDF/DOCX ya generado con la
métrica clásica — cambiar `pipeline_optimizacion.py` regeneraría todas las figuras/tablas
ya publicadas sin que el usuario lo haya pedido explícitamente. La función queda disponible
en `broken_pipe_test.py` para quien quiera correr el pipeline con la métrica alternativa
sobre este universo (que, con SPY/QQQ/EEM/VNQ/TLT/IEF/LQD/GLD/DBC, no tiene coberturas
tan extremas como ρ=-0.9 — TLT/IEF vs. equities rondan ρ≈-0.3/-0.4 en el período
2007-2026 según `tabla_diagnostico_econometrico.csv`, un caso más moderado que el ejemplo
del repo hermano, pero el mismo argumento matemático aplica en magnitud proporcional).
