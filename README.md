# Motor Cuantitativo de Optimización de Portafolios & Gestión de Riesgo

Framework modular en Python para la construcción, optimización y validación fuera de muestra (out-of-sample) de carteras de inversión, diseñado para mitigar la inestabilidad de la frontera media-varianza clásica.

## 📌 Metodologías y Enfoque

- **Asignación Jerárquica de Capital:** Implementación de *Hierarchical Equal Risk Contribution* (HERC) y *Hierarchical Risk Parity* (HRP) mediante clustering aglomerativo sobre distancias de correlación.
- **Estimación Robusta de Covarianza:** Contracción lineal Ledoit-Wolf y filtrado espectral para matrices mal condicionadas.
- **Modelación de Colas Pesadas (Fat Tails):** Ajuste por Teoría de Valores Extremos (EVT - Generalized Pareto Distribution en 2 etapas McNeil & Frey) sobre residuos estandarizados de modelos GARCH(1,1).
- **Métricas de Riesgo Asimétrico:** Optimización orientada a la minimización de Conditional Value-at-Risk (CVaR / Expected Shortfall) y Conditional Drawdown-at-Risk (CDaR).
- **Protocolo de Backtesting Anti-Sobreajuste:**
  - Validación walk-forward continua con rebalanceo periódico y costos de transacción explícitos (5 bps).
  - *Combinatorial Purged Cross-Validation* (CPCV) con embargo para neutralizar fuga de información (lookahead bias).
  - Cálculo del *Deflated Sharpe Ratio* (DSR) de López de Prado para auditar la significancia estadística frente al benchmark 60/40.

## 🛠️ Requisitos e Instalación

`ash
git clone https://github.com/federicoagustinchillon-creador/optimizacion-portafolios.git
cd optimizacion-portafolios
pip install -r requirements.txt
`
