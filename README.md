# Lithium-Battery-Trading-Optimization

Optimización de la operación diaria de una batería de iones de litio que
participa simultáneamente en el mercado spot español (OMIE) y en el mercado
secundario de regulación (aFRR) de Red Eléctrica de España.

Autor: Hugo Raggini Paternain

---

## 1. Objetivo

Construir un modelo de programación lineal entera mixta (MILP) que decida, en
cada uno de los 96 intervalos cuartohorarios de un día, qué cantidad de energía
comprar/vender en el mercado spot y qué banda de potencia (subida y bajada)
ofertar como reserva secundaria, de manera que se maximice el beneficio neto
del día.

A continuación se contrastan los resultados teóricos con un simulador Monte
Carlo que aplica el cronograma optimizado bajo condiciones realistas con
incertidumbre en precios, activación y disponibilidad técnica.

## 2. Contexto del mercado eléctrico español

El modelo considera dos mercados:

- **Mercado spot diario (OMIE)**: subasta horaria/cuartohoraria donde se casan
  oferta y demanda. Se descarga el precio marginal del sistema español en
  €/MWh para cada uno de los 96 intervalos del día.
- **Reserva secundaria (aFRR / banda secundaria)**: producto gestionado por
  REE para mantener el balance frecuencia-potencia. Tiene dos componentes
  retributivas:
  - **Disponibilidad** (€/MWh ofertado): pago por reservar capacidad,
    independientemente de si se llega a activar. Se separa en pago por banda
    a subida y por banda a bajada, con valores distintos.
  - **Activación** (€/MWh entregado): pago por la energía efectivamente
    entregada cuando REE lo solicita.

La batería puede participar simultáneamente en ambos mercados, siempre que
respete el balance energético, los límites de potencia y la disponibilidad de
SOC en cada momento.

## 3. Datos de entrada

Los precios horarios del sistema español se obtienen de los ficheros
`INT_PBC_EV_H_1_*.xls` publicados por OMIE. Cuando un fichero contiene los 24
precios horarios (formato antiguo), se expanden a 96 cuartos mediante un
muestreo lognormal con `σ_intra = 2%` que **conserva la media horaria**
exactamente (el cuarto Q4 se calcula como residuo para que `mean(Q1..Q4) = p_h`).

Estructura esperada en `datos/omie_xls/`:

```
datos/omie_xls/
├── Enero 2026/
│   ├── INT_PBC_EV_H_1_01_01_2026_01_01_2026.XLS
│   ├── ...
├── Febrero 2026/
└── ...
```

Tras el parseo, se genera un CSV mensual por cada combinación mes-año en
`datos/precios/precios_<mes>_<anio>.csv`, con 96 columnas (`H1Q1`, `H1Q2`, …,
`H24Q4`) y una fila por día.

## 4. Formulación del modelo

### 4.1 Variables de decisión

Para cada intervalo `t ∈ {1, …, 96}` (cuartos de hora):

| Variable | Descripción | Unidad |
|---|---|---|
| `x_sell[t]` | Energía vendida en spot | MWh |
| `x_buy[t]` | Energía comprada en spot | MWh |
| `x_ch[t]` | Energía cargada a batería desde la red | MWh |
| `x_dis[t]` | Energía descargada de batería a la red | MWh |
| `r_up[t]` | Banda de reserva ofertada a subida | MWh |
| `r_down[t]` | Banda de reserva ofertada a bajada | MWh |
| `a_up[t]` | Activación esperada al alza (`= α_up · r_up`) | MWh |
| `a_down[t]` | Activación esperada a la baja (`= α_down · r_down`) | MWh |
| `SOC[t]` | Estado de carga al final del intervalo | MWh |
| `y_bat[t]` | Binaria: 1 si carga, 0 si descarga (impide simultaneidad) | — |
| `y_grid[t]` | Binaria: 1 si compra, 0 si vende | — |

### 4.2 Función objetivo

```
max Σ_t [ p[t] · x_sell[t]
        − p_eff[t] · x_buy[t]
        + π_disp_up   · r_up[t]
        + π_disp_down · r_down[t]
        + (π_act_up   − p[t])     · a_up[t]
        + (π_act_down − p_eff[t]) · a_down[t]
        − c_deg · (x_ch[t] + x_dis[t] + a_up[t] + a_down[t]) ]
```

Donde:

- `p[t]` es el precio spot real, **que puede ser negativo**.
- `p_eff[t] = max(p[t], 0)` es el precio efectivo de compra. Al usar `p_eff`
  en lugar de `p` en el coste de compra, **se impide que el modelo "gane
  dinero" comprando a precio negativo** (lo que sería un artefacto del
  optimizador, no una operación real).
- Las dos partidas de disponibilidad reflejan la **asimetría real del pago
  por banda secundaria** en REE: la subida cotiza ligeramente más cara que la
  bajada.
- El término de activación captura el **margen** entre el precio de
  activación y el precio spot: cuando REE activa subida, la batería
  descarga y cobra `π_act_up`, pero "sacrifica" la energía que podría haber
  vendido al spot, de ahí la resta.
- `c_deg` aplica una penalización proporcional a la energía total operada
  (carga + descarga + activación), como proxy del coste de degradación de
  la batería.

### 4.3 Restricciones

- **Balance energético por intervalo**:
  `x_buy + x_dis + a_up = x_sell + x_ch + a_down`

- **Dinámica del SOC** con eficiencia `η`:
  `SOC[t] = SOC[t−1] + η · (x_ch + a_down) − (x_dis + a_up) / η`

- **Límites de SOC**: `SOC_min ≤ SOC[t] ≤ SOC_max` con
  `SOC_min = E_max · (1 − DoD)`.

- **No simultaneidad carga/descarga** (`y_bat`) ni compra/venta (`y_grid`):
  uso de Big-M con variables binarias.

- **Límites de potencia** (incluyendo activación):
  `x_ch + a_down ≤ y_bat · P_ch_max · Δt`
  `x_dis + a_up ≤ (1 − y_bat) · P_dis_max · Δt`

- **Reserva acotada por potencia disponible**:
  `x_dis + r_up ≤ P_dis_max · Δt`
  `x_ch + r_down ≤ P_ch_max · Δt`

- **Activación esperada proporcional a reserva**: `a_up = α_up · r_up`,
  `a_down = α_down · r_down`.

- **Reserva acotada por SOC disponible**: la reserva al alza nunca puede
  comprometer más energía de la que físicamente queda en la batería; la
  reserva a la baja, por encima de lo que cabe.

- **Cierre del ciclo diario**: `SOC[96] = SOC[0]`, lo que garantiza que
  el día no se "vacía" la batería para inflar el beneficio.

## 5. Parámetros del sistema

Centralizados en `parametros.py` para que optimizador, simulador y
visualizaciones compartan la misma configuración.

### 5.1 Batería

| Parámetro | Valor | Comentario |
|---|---|---|
| `E_MAX` | 2.0 MWh | Capacidad nominal |
| `P_CH_MAX` / `P_DIS_MAX` | 0.5 MW | Batería de 4 h (ratio E/P = 4) |
| `ETA` | 0.9381 | Eficiencia round-trip √(η_round) |
| `DOD` | 0.80 | Profundidad de descarga permitida |
| `SOC_INIT` | 1.0 MWh | Estado inicial = 50% E_MAX |
| `C_DEG` | 6.63 €/MWh | Coste de degradación por MWh operado |

### 5.2 Mercado spot

| Parámetro | Valor |
|---|---|
| `DELTA` | 0.25 h |
| `M_BIG` | 10000.0 |

### 5.3 Reserva secundaria aFRR

| Parámetro | Valor | Fuente |
|---|---|---|
| `ALPHA_UP` | 0.236 | Tasa esperada de activación al alza |
| `ALPHA_DOWN` | 0.222 | Tasa esperada de activación a la baja |
| `PI_DISP_UP` | 9.27 €/MWh | Pago disponibilidad subida (REE) |
| `PI_DISP_DOWN` | 8.48 €/MWh | Pago disponibilidad bajada (REE) |
| `PI_ACT_UP` | 118.15 €/MWh | Pago activación al alza |
| `PI_ACT_DOWN` | 53.05 €/MWh | Pago activación a la baja |

## 6. Simulador Monte Carlo

El optimizador asume conocimiento perfecto del día siguiente, lo cual no
ocurre en operación real. Para cuantificar el impacto de la incertidumbre se
implementa un simulador que ejecuta el cronograma optimizado bajo condiciones
estocásticas.

### 6.1 Fuentes de incertidumbre modeladas

| Fuente | Distribución | σ por defecto |
|---|---|---|
| Error de previsión spot | Lognormal multiplicativa sobre el precio previsto | 12% |
| Error en `π_disp_up/down` | Lognormal escalar (ruido independiente up/down) | 5% |
| Error en `π_act_up/down` | Lognormal escalar | 10% |
| Puja perdida | Bernoulli por intervalo (no entra carga) | 5% |
| Activación real ≠ esperada | Factor lognormal sobre `a_up_prev` y `a_down_prev` | 15% |
| Fallo técnico puntual | Bernoulli por intervalo (no opera batería) | 2% |

### 6.2 Penalizaciones realistas

- **Recorte proporcional por SOC insuficiente**: si la descarga planificada
  (arbitraje + activación) excede la energía disponible, se recorta
  proporcionalmente — la batería entrega lo que tiene.
- **Penalización por desviación de SOC final**: cualquier déficit o exceso
  respecto a `SOC_INIT` se penaliza con el precio medio del día, modelando el
  coste de reposición o de oportunidad asociado.

### 6.3 Número de escenarios

Se simulan 200 escenarios por día. La justificación del valor está en §6.4.

### 6.4 Análisis de convergencia

`simulacion/convergencia.py` ejecuta el mismo día con N ∈ {25, 50, 100, 150,
200, 300, 500} simulaciones y mide la varianza del estimador P50/P5 entre
10 repeticiones independientes. Permite justificar formalmente que N=200
escenarios es suficiente: el error estándar relativo del estimador central
(P50) queda por debajo del 5% y el de la cola (P5, más ruidosa por
construcción) por debajo del 15%.

## 7. Estructura del proyecto

```
Modelo VSCode/
├── README.md
├── parametros.py             # Fuente unica de constantes
├── run.py                    # Orquestador del pipeline completo
│
├── parseo/
│   └── omie.py               # XLS de OMIE  ->  CSV mensual cuartohorario
│
├── optimizacion/
│   ├── bateria.py            # MILP de un dia (Pyomo + HiGHS)
│   └── mes.py                # Loop diario sobre un mes
│
├── simulacion/
│   ├── dia.py                # Monte Carlo de un dia
│   ├── mes.py                # Monte Carlo sobre un mes
│   └── convergencia.py       # Analisis de N suficiente
│
├── visualizacion/
│   ├── dia.py                # Dashboard operacion y economia de 1 dia
│   ├── mes.py                # 4 dashboards mensuales (det)
│   ├── sim_dia.py            # Distribucion de beneficios de 1 dia (MC)
│   ├── sim_mes.py            # 3 dashboards mensuales (sim)
│   └── anual.py              # Comparador det/sim/comp sobre rango de meses
│
├── datos/
│   ├── omie_xls/             # XLS originales de OMIE
│   └── precios/              # CSVs cuartohorarios mensuales
│
└── resultados/
    ├── optimizacion/         # CSVs deterministas por dia + resumen mensual
    ├── simulacion/           # CSVs Monte Carlo + resumen estadistico
    └── dias_sueltos/         # Resultados de dias optimizados individualmente
```

## 8. Ejecución

### 8.1 Pipeline completo

```bash
python run.py                # detecta meses con datos y procesa los que falten
python run.py 1 2026         # solo un mes
python run.py 1 2026 --force # rehace pasos aunque existan resultados
```

`run.py` detecta automáticamente los meses presentes en `datos/omie_xls/` y,
para cada uno, encadena:

1. `parseo.omie` — extrae los XLS a un CSV cuartohorario mensual
2. `optimizacion.mes` — resuelve el MILP día a día, guarda 1 CSV por día
   más un resumen del mes
3. `simulacion.mes` — para cada día, optimiza y ejecuta 200 escenarios
   Monte Carlo, guarda CSVs por día y un resumen estadístico

Los pasos ya completados se saltan automáticamente.

### 8.2 Scripts individuales

```bash
# Parseo
python -m parseo.omie 1 2026

# Optimización
python -m optimizacion.bateria         # interactivo, un dia suelto
python -m optimizacion.mes 1 2026

# Simulación
python -m simulacion.dia               # interactivo, un dia
python -m simulacion.mes 1 2026 200    # 200 escenarios normales
python -m simulacion.convergencia 2026-01-01

# Visualización
python -m visualizacion.dia 2026-01-01
python -m visualizacion.mes 1 2026
python -m visualizacion.sim_dia 2026-01-01
python -m visualizacion.sim_mes 1 2026
python -m visualizacion.anual          # interactivo, rango de meses
```

## 9. Resultados experimentales (Mar 2025 → Feb 2026)

Sobre 12 meses consecutivos de datos OMIE reales y una batería de 2 MWh / 4 h:

| Métrica | Valor |
|---|---|
| Beneficio determinista anual (modelo) | 93 685 € |
| Beneficio P50 anual (mediana Monte Carlo) | 88 832 € |
| Coste de incertidumbre | 4 853 € (5.2%) |
| Beneficio mensual medio | ~7 800 € |
| Mes mejor | Mayo 2025 (9 796 €) |
| Mes peor | Diciembre 2025 (5 399 €) |

El **coste de incertidumbre** (diferencia entre el beneficio que promete el
optimizador y la mediana real observada en simulación) oscila entre el 2.6%
(febrero, mercado más estable) y el 11% (diciembre, mayor volatilidad
invernal). Esto refleja el **límite práctico del modelo determinista**:
asumir conocimiento perfecto sobreestima el beneficio en un 5% promedio
anual.

El comportamiento del optimizador con `π_disp_up > π_disp_down` resulta en
una clara preferencia por ofrecer banda a subida — el ratio `r_up : r_down`
observado en los días de ejemplo se acerca a 60:1, consecuencia combinada de
mejor pago por disponibilidad y mayor spread entre `π_act_up` y el precio
spot.

## 10. Decisiones de modelado destacables

1. **Doble precio de compra (`p` y `p_eff`)**: usar `p_eff = max(p, 0)` en
   el coste de compra impide que el modelo explote precios negativos como
   ingreso ficticio. Es una decisión conservadora frente al artefacto
   numérico habitual.

2. **Asimetría de pago por disponibilidad**: separar `π_disp_up` y
   `π_disp_down` permite que el modelo decida la composición óptima de la
   banda en lugar de tratarla como simétrica.

3. **Coste de degradación lineal**: `C_DEG · MWh_operado` es una
   simplificación. Modelos más sofisticados usan funciones cuadráticas de
   DOD o cuentan ciclos equivalentes, pero el lineal es suficiente para
   estudios de rentabilidad agregada y mantiene el modelo MILP.

4. **Cierre diario `SOC_final = SOC_inicial`**: imprescindible para
   comparaciones día a día y para evitar que el optimizador "venda" la
   batería al final del día como ingreso espurio. Convierte el problema en
   recurrente y comparable.

5. **Simulador independiente del optimizador**: el simulador no resuelve un
   problema de control en línea — ejecuta el schedule fijo bajo realidad
   estocástica. Esto permite cuantificar la pérdida por incertidumbre sin
   confundirla con la mejora por re-optimización intra-diaria.

## 11. Limitaciones y trabajo futuro

- **No hay re-optimización intra-diaria**: el schedule del día anterior se
  ejecuta tal cual. Un modelo MPC (Model Predictive Control) que
  re-optimizase cada hora podría reducir el coste de incertidumbre.
- **C_DEG constante**: la degradación real depende del DOD por ciclo y del
  estado térmico. Un modelo de envejecimiento más fiel afinaría la
  rentabilidad esperada.
- **Asimetría de activación**: aunque la disponibilidad está separada
  up/down, las tasas `α_up` y `α_down` son constantes. Podrían modelarse
  como series temporales (más activación al alza en horas de demanda alta).
- **Solo aFRR (secundaria)**: el modelo ignora mercado terciario y
  desvíos. Incorporarlos requiere ampliar la función objetivo y las
  restricciones de no-superposición.
- **Sin acoplamiento con previsión real**: los errores de previsión se
  modelan como lognormales escalares; un modelo más realista usaría
  errores con autocorrelación temporal.

## 12. Dependencias

- Python 3.11+
- `pyomo` (modelo MILP)
- `highspy` o `highs` (solver MILP; alternativas: `glpk`, `cbc`)
- `pandas`, `numpy`
- `matplotlib`
- `xlrd` (lectura de XLS de OMIE)
