"""
Optimización Cuartohoraria de Batería — Arbitraje + Mercado Secundario (aFRR)
Autor: Hugo Raggini Paternain

Modelo de optimización con Pyomo para maximizar beneficios en el mercado eléctrico.
Lee precios reales del CSV mensual generado por parseo_omie.py.

Uso:
    python optimizacion_bateria.py    # pregunta fecha interactivamente
"""

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# =============================================================================
# PARÁMETROS DEL SISTEMA
# =============================================================================

# --- Batería ---
E_MAX     = 2.0
P_CH_MAX  = 1.0
P_DIS_MAX = 1.0
ETA       = 0.90
DOD       = 0.93
SOC_MIN   = E_MAX * (1 - DOD)
SOC_MAX   = E_MAX
SOC_INIT  = E_MAX * 0.5
C_DEG     = 2.0

# --- Mercado ---
DELTA = 0.25
M_BIG = 100.0

# --- Reserva secundaria (aFRR / REE) ---
ALPHA_UP    = 0.2357
ALPHA_DOWN  = 0.2225
PI_DISP     = 10.0
PI_ACT_UP   = 114.30
PI_ACT_DOWN = 50.73

# --- Configuración de carpetas ---
NOMBRES_MES = {
    1: "Enero",      2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",       6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre",  12: "Diciembre"
}
COLUMNAS_Q     = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]
CARPETA_SALIDA  = Path("Resultados Días Sueltos")
CARPETA_PRECIOS = Path("Precios")


# =============================================================================
# CARGA DE PRECIOS DESDE CSV
# =============================================================================

def cargar_precios(fecha_str: str) -> np.ndarray:
    """
    Lee los precios cuartohorarios de una fecha dada desde el CSV mensual.
    fecha_str formato: 'YYYY-MM-DD'
    """
    fecha  = pd.to_datetime(fecha_str)
    mes    = fecha.month
    anio   = fecha.year
    nombre = NOMBRES_MES[mes]
    csv    = CARPETA_PRECIOS / f"precios_{nombre.lower()}_{anio}.csv"

    if not csv.exists():
        raise FileNotFoundError(
            f"No se encuentra '{csv}'. Ejecuta primero: python parseo_omie.py {mes} {anio}"
        )

    df = pd.read_csv(csv)

    # Verificar columnas
    for col in COLUMNAS_Q:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {csv}")

    fila = df[df["fecha"] == fecha_str]
    if fila.empty:
        raise ValueError(
            f"Fecha '{fecha_str}' no encontrada en '{csv}'.\n"
            f"  Fechas disponibles: {df['fecha'].min()} → {df['fecha'].max()}"
        )

    precios = fila[COLUMNAS_Q].values[0].astype(float)
    if len(precios) != 96:
        raise ValueError(f"Se esperaban 96 valores, se encontraron {len(precios)}")

    return precios


# =============================================================================
# CONSTRUCCIÓN DEL MODELO PYOMO
# =============================================================================

def construir_modelo(precios: np.ndarray):
    """
    Construye el modelo MILP de optimización de batería pura.
    Sin generación fotovoltaica — solo arbitraje + servicios de red.

    Mercados modelados:
      · Arbitraje (OMIE): compra barato / vende caro
      · Reserva secundaria (aFRR): disponibilidad + activación up/down

    Notación de precios:
      · p[t]     — precio SPOT del mercado en t (puede ser negativo)
      · p_eff[t] — precio efectivo de compra: max(p[t], 0)
                   Evita que el modelo gane dinero comprando a precio negativo.
    """
    model = pyo.ConcreteModel(name="Arbitraje_Bateria_Pura")

    T_set = range(1, 97)
    model.T = pyo.Set(initialize=T_set)

    # --- Parámetros de precio ---
    # p[t]: precio SPOT en el intervalo t (€/MWh) — puede ser negativo
    model.p     = pyo.Param(model.T, initialize={t: precios[t-1] for t in T_set})
    # p_eff[t]: precio efectivo de compra = max(p[t], 0)
    model.p_eff = pyo.Param(model.T, initialize={t: max(precios[t-1], 0.0) for t in T_set})

    # --- Parámetros de batería ---
    model.E_max    = pyo.Param(initialize=E_MAX)
    model.P_ch_max = pyo.Param(initialize=P_CH_MAX)
    model.P_dis_max= pyo.Param(initialize=P_DIS_MAX)
    model.eta      = pyo.Param(initialize=ETA)
    model.SOC_min  = pyo.Param(initialize=SOC_MIN)
    model.SOC_max  = pyo.Param(initialize=SOC_MAX)
    model.SOC_init = pyo.Param(initialize=SOC_INIT)
    model.c_deg    = pyo.Param(initialize=C_DEG)

    # --- Parámetros de mercado ---
    model.delta       = pyo.Param(initialize=DELTA)
    model.M           = pyo.Param(initialize=M_BIG)
    model.alpha_up    = pyo.Param(initialize=ALPHA_UP)
    model.alpha_down  = pyo.Param(initialize=ALPHA_DOWN)
    model.pi_disp     = pyo.Param(initialize=PI_DISP)
    model.pi_act_up   = pyo.Param(initialize=PI_ACT_UP)
    model.pi_act_down = pyo.Param(initialize=PI_ACT_DOWN)

    # --- Variables ---
    model.x_sell = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.x_buy  = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.x_ch   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.x_dis  = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.r_up   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.r_down = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.a_up   = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.a_down = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.SOC    = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.y_bat  = pyo.Var(model.T, domain=pyo.Binary)
    model.y_grid = pyo.Var(model.T, domain=pyo.Binary)

    # --- Función objetivo ---
    # Ingresos por venta en SPOT:         p[t]     * x_sell[t]
    # Costes por compra en SPOT:          p_eff[t] * x_buy[t]
    # Ingresos por disponibilidad aFRR:   pi_disp  * (r_up[t] + r_down[t])
    # Ingresos netos por activación up:   (pi_act_up   - p[t])     * a_up[t]
    # Ingresos netos por activación down: (pi_act_down - p_eff[t]) * a_down[t]
    # Costes de degradación:              c_deg * (energía total operada)
    def obj_rule(m):
        return sum(
            m.p[t]     * m.x_sell[t]
            - m.p_eff[t] * m.x_buy[t]
            + m.pi_disp  * (m.r_up[t] + m.r_down[t])
            + (m.pi_act_up   - m.p[t])     * m.a_up[t]
            + (m.pi_act_down - m.p_eff[t]) * m.a_down[t]
            - m.c_deg * (m.x_ch[t] + m.x_dis[t] + m.a_up[t] + m.a_down[t])
            for t in m.T
        )
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # --- Restricciones ---

    # Balance energético en cada intervalo
    def balance_rule(m, t):
        return (m.x_buy[t] + m.x_dis[t] + m.a_up[t]
                == m.x_sell[t] + m.x_ch[t] + m.a_down[t])
    model.balance = pyo.Constraint(model.T, rule=balance_rule)

    # Dinámica del SOC
    def soc_rule(m, t):
        soc_prev = m.SOC_init if t == 1 else m.SOC[t-1]
        return (m.SOC[t] == soc_prev
                + m.eta * (m.x_ch[t] + m.a_down[t])
                - (m.x_dis[t] + m.a_up[t]) / m.eta)
    model.soc_dyn   = pyo.Constraint(model.T, rule=soc_rule)
    model.soc_min_c = pyo.Constraint(model.T, rule=lambda m, t: m.SOC[t] >= m.SOC_min)
    model.soc_max_c = pyo.Constraint(model.T, rule=lambda m, t: m.SOC[t] <= m.SOC_max)

    # Límites de potencia de carga y descarga
    model.pmax_ch = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_ch[t] + m.a_down[t]
                          <= m.y_bat[t] * m.P_ch_max * m.delta)
    model.pmax_dis = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_dis[t] + m.a_up[t]
                          <= (1 - m.y_bat[t]) * m.P_dis_max * m.delta)

    # Reserva limitada por potencia máxima
    model.res_up_pow   = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_dis[t] + m.r_up[t]   <= m.P_dis_max * m.delta)
    model.res_down_pow = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_ch[t]  + m.r_down[t] <= m.P_ch_max  * m.delta)

    # Activación limitada por reserva ofertada
    model.act_up_c   = pyo.Constraint(model.T,
        rule=lambda m, t: m.a_up[t]   == m.alpha_up   * m.r_up[t])
    model.act_down_c = pyo.Constraint(model.T,
        rule=lambda m, t: m.a_down[t] == m.alpha_down * m.r_down[t])

    # Reserva limitada por SOC disponible
    def res_up_soc_rule(m, t):
        soc_prev = m.SOC_init if t == 1 else m.SOC[t-1]
        return m.r_up[t] <= m.eta * (soc_prev - m.SOC_min)
    def res_down_soc_rule(m, t):
        soc_prev = m.SOC_init if t == 1 else m.SOC[t-1]
        return m.r_down[t] <= (m.SOC_max - soc_prev) / m.eta
    model.res_up_soc   = pyo.Constraint(model.T, rule=res_up_soc_rule)
    model.res_down_soc = pyo.Constraint(model.T, rule=res_down_soc_rule)

    # No comprar y vender simultáneamente
    model.buy_c  = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_buy[t]  <= m.y_grid[t]       * m.M)
    model.sell_c = pyo.Constraint(model.T,
        rule=lambda m, t: m.x_sell[t] <= (1 - m.y_grid[t]) * m.M)

    # SOC final igual al inicial
    model.soc_final = pyo.Constraint(rule=lambda m: m.SOC[96] == m.SOC_init)

    return model


# =============================================================================
# RESOLUCIÓN Y RESULTADOS
# =============================================================================

def resolver_modelo(model, solver="highs"):
    opt = pyo.SolverFactory(solver)
    opt.options["time_limit"]     = 120
    opt.options["mip_rel_gap"]    = 0.001
    opt.options["output_flag"]    = 0
    opt.options["log_to_console"] = 0
    results = opt.solve(model, tee=False)
    return results


def extraer_resultados(model):
    data = []
    for t in model.T:
        data.append({
            "t":                        t,
            "hora":                     f"{(t-1)//4:02d}:{((t-1)%4)*15:02d}",
            "p [€/MWh]":               round(pyo.value(model.p[t]), 2),
            "p_eff [€/MWh]":           round(pyo.value(model.p_eff[t]), 2),
            "x_sell [MWh]":            round(pyo.value(model.x_sell[t]), 4),
            "x_buy [MWh]":             round(pyo.value(model.x_buy[t]), 4),
            "x_ch [MWh]":              round(pyo.value(model.x_ch[t]), 4),
            "x_dis [MWh]":             round(pyo.value(model.x_dis[t]), 4),
            "r_up [MWh]":              round(pyo.value(model.r_up[t]), 4),
            "r_down [MWh]":            round(pyo.value(model.r_down[t]), 4),
            "a_up [MWh]":              round(pyo.value(model.a_up[t]), 4),
            "a_down [MWh]":            round(pyo.value(model.a_down[t]), 4),
            "SOC [MWh]":               round(pyo.value(model.SOC[t]), 4),
            "y_bat":                   int(pyo.value(model.y_bat[t])),
            "y_grid":                  int(pyo.value(model.y_grid[t])),
            "pi_disp [€/MWh]":         float(pyo.value(model.pi_disp)),
            "pi_act_up [€/MWh]":       float(pyo.value(model.pi_act_up)),
            "pi_act_down [€/MWh]":     float(pyo.value(model.pi_act_down)),
            "E_max [MWh]":             float(pyo.value(model.E_max)),
            "SOC_min [MWh]":           float(pyo.value(model.SOC_min)),
        })
    df = pd.DataFrame(data)
    beneficio_total = pyo.value(model.obj)

    print(f"\n{'='*50}")
    print(f"   BENEFICIO TOTAL DEL DÍA: {beneficio_total:.2f} €")
    print(f"{'='*50}\n")

    return df, beneficio_total


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n=== Optimización de Batería — Día Suelto ===")
    print("Introduce la fecha a optimizar.")
    print("(Los datos deben estar en el CSV mensual generado por parseo_omie.py)\n")

    while True:
        entrada = input("Fecha (DD/MM/YYYY o YYYY-MM-DD): ").strip()
        try:
            if "/" in entrada:
                fecha_dt  = pd.to_datetime(entrada, dayfirst=True)
            else:
                fecha_dt  = pd.to_datetime(entrada)
            fecha_str = fecha_dt.strftime("%Y-%m-%d")
            break
        except Exception:
            print("  [!] Formato no reconocido. Prueba: 13/03/2026 o 2026-03-13")

    try:
        precios = cargar_precios(fecha_str)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[!] Error: {e}")
        sys.exit()

    print(f"\n  Fecha       : {fecha_str}")
    print(f"  Precio máx  : {precios.max():.2f} €/MWh")
    print(f"  Precio mín  : {precios.min():.2f} €/MWh")
    print(f"  Spread      : {precios.max() - precios.min():.2f} €/MWh")
    print(f"  Precio medio: {precios.mean():.2f} €/MWh")
    print(f"  Optimizando...\n")

    model   = construir_modelo(precios)
    results = resolver_modelo(model)

    status = results.solver.termination_condition
    if status not in (pyo.TerminationCondition.optimal,
                      pyo.TerminationCondition.feasible):
        print(f"[!] Solver terminó con estado: {status}")
        sys.exit()

    df, beneficio = extraer_resultados(model)
    df.insert(0, "fecha", fecha_str)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    csv_salida = CARPETA_SALIDA / f"resultado_{fecha_str}.csv"
    df.to_csv(csv_salida, index=False)

    print(f"  CSV guardado: {csv_salida}")
    print(f"  Listo. Puedes visualizar con: python visualizacion_resultados.py {fecha_str}\n")