"""
Simulador de Ejecución Real — Batería Pura (Arbitraje + aFRR)
Autor: Hugo Raggini Paternain
---------------------------------------------------------------
Flujo:
  1. El modelo de optimización decide el schedule la noche anterior
     usando precios previstos (OMIE histórico)
  2. El día real, los precios y eventos difieren de la previsión
  3. El schedule se ejecuta igualmente (ya comprometido en mercado)
     pero el SOC evoluciona con lo que REALMENTE ocurre
  4. El beneficio real puede ser mejor o peor que el previsto

Eventos modelados:
  · Error de previsión de precios spot        (lognormal)
  · Error en precio de reserva aFRR           (lognormal)
  · No ganar la puja de carga                 (Bernoulli)
  · Activación de reserva mayor/menor         (lognormal)
  · Fallo técnico puntual de batería          (Bernoulli)

SOC y penalizaciones:
  · Recorte proporcional si SOC insuficiente para ejecutar schedule
  · Penalización dinámica si SOC final ≠ SOC_INIT:
      - Déficit  → coste de reposición  = |desviación| × precio_medio_día
      - Exceso   → coste de oportunidad = desviación   × precio_medio_día

Casos extremos:
  · Escenarios normales   (distribución general)
  · Escenarios extremos   (P95/P99 — amplificador ×3)
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PARÁMETROS DEL SISTEMA
# =============================================================================

E_MAX     = 2.0
P_CH_MAX  = 1.0
P_DIS_MAX = 1.0
ETA       = 0.90
DOD       = 0.93
SOC_MIN   = E_MAX * (1 - DOD)
SOC_MAX   = E_MAX
SOC_INIT  = E_MAX * 0.5
C_DEG     = 2.0
DELTA     = 0.25
M_BIG     = 100.0

PI_DISP_BASE     = 10.0
PI_ACT_UP_BASE   = 114.30
PI_ACT_DOWN_BASE = 50.73
ALPHA_UP_BASE    = 0.2357
ALPHA_DOWN_BASE  = 0.2225


# =============================================================================
# PASO 1 — OBTENER EL SCHEDULE DEL MODELO DE OPTIMIZACIÓN
# =============================================================================

def obtener_schedule(precios_previstos: np.ndarray, solver: str = "highs") -> dict:
    """
    Ejecuta el modelo de optimización UNA SOLA VEZ con los precios previstos
    y devuelve el schedule comprometido para el día siguiente.
    """
    from optimizacion_bateria import construir_modelo

    print("  Optimizando schedule con precios previstos...")
    model = construir_modelo(precios_previstos)
    opt   = pyo.SolverFactory(solver)
    opt.options["time_limit"]     = 120
    opt.options["mip_rel_gap"]    = 0.001
    opt.options["output_flag"]    = 0
    opt.options["log_to_console"] = 0
    results = opt.solve(model, tee=False)

    status = results.solver.termination_condition
    if status not in (pyo.TerminationCondition.optimal,
                      pyo.TerminationCondition.feasible):
        raise RuntimeError(f"Optimización fallida: {status}")

    schedule = {
        "x_ch":               np.array([pyo.value(model.x_ch[t])   for t in range(1, 97)]),
        "x_dis":              np.array([pyo.value(model.x_dis[t])  for t in range(1, 97)]),
        "x_sell":             np.array([pyo.value(model.x_sell[t]) for t in range(1, 97)]),
        "x_buy":              np.array([pyo.value(model.x_buy[t])  for t in range(1, 97)]),
        "r_up":               np.array([pyo.value(model.r_up[t])   for t in range(1, 97)]),
        "r_down":             np.array([pyo.value(model.r_down[t]) for t in range(1, 97)]),
        "a_up_prev":          np.array([pyo.value(model.a_up[t])   for t in range(1, 97)]),
        "a_down_prev":        np.array([pyo.value(model.a_down[t]) for t in range(1, 97)]),
        "beneficio_previsto": pyo.value(model.obj),
        "precios_previstos":  precios_previstos.copy(),
    }

    print(f"  Schedule obtenido. Beneficio previsto: {schedule['beneficio_previsto']:.2f} €\n")
    return schedule


# =============================================================================
# PASO 2 — GENERAR ESCENARIO DE EJECUCIÓN REAL
# =============================================================================

def generar_escenario_ejecucion(
    schedule: dict,
    rng: np.random.Generator,
    sigma_spot: float       = 0.12,
    sigma_pi_disp: float    = 0.05,
    sigma_pi_act: float     = 0.10,
    p_no_puja: float        = 0.05,
    sigma_activacion: float = 0.15,
    p_fallo_tecnico: float  = 0.02,
    extremo: bool           = False,
) -> dict:
    """
    Genera un escenario de ejecución real dado el schedule fijo.
    En modo extremo amplifica todos los factores de riesgo × 3.
    """
    T            = 96
    precios_prev = schedule["precios_previstos"]
    amp          = 3.0 if extremo else 1.0

    # 1. Precios spot reales — error lognormal multiplicativo
    epsilon        = rng.standard_normal(T)
    factor         = np.exp(sigma_spot * amp * epsilon)
    precios_reales = np.sign(precios_prev) * np.abs(precios_prev) * factor

    # 2. Precios de reserva reales — lognormal escalar
    pi_disp_real     = PI_DISP_BASE     * np.exp(sigma_pi_disp * amp * rng.standard_normal())
    pi_act_up_real   = PI_ACT_UP_BASE   * np.exp(sigma_pi_act  * amp * rng.standard_normal())
    pi_act_down_real = PI_ACT_DOWN_BASE * np.exp(sigma_pi_act  * amp * rng.standard_normal())

    # 3. Pujas perdidas — Bernoulli por intervalo
    p_adj        = min(p_no_puja * amp, 0.95)
    no_gana_puja = rng.random(T) < p_adj

    # 4. Activación real de reserva — factor lognormal sobre lo previsto
    #    Puede ser mayor o menor que lo asumido por el modelo
    factor_act_up   = np.exp(sigma_activacion * amp * rng.standard_normal())
    factor_act_down = np.exp(sigma_activacion * amp * rng.standard_normal())
    a_up_real   = np.clip(schedule["a_up_prev"]   * factor_act_up,   0, schedule["r_up"])
    a_down_real = np.clip(schedule["a_down_prev"] * factor_act_down, 0, schedule["r_down"])

    # 5. Fallos técnicos — Bernoulli por intervalo
    p_fallo_adj   = min(p_fallo_tecnico * amp, 0.5)
    fallo_tecnico = rng.random(T) < p_fallo_adj

    return {
        "precios_reales":    precios_reales,
        "pi_disp_real":      pi_disp_real,
        "pi_act_up_real":    pi_act_up_real,
        "pi_act_down_real":  pi_act_down_real,
        "no_gana_puja":      no_gana_puja,
        "a_up_real":         a_up_real,
        "a_down_real":       a_down_real,
        "fallo_tecnico":     fallo_tecnico,
        "factor_act_up":     factor_act_up,
        "factor_act_down":   factor_act_down,
        "extremo":           extremo,
    }


# =============================================================================
# PASO 3 — SIMULAR EJECUCIÓN REAL (SOC en cascada)
# =============================================================================

def simular_ejecucion(schedule: dict, escenario: dict) -> dict:
    """
    Ejecuta el schedule fijo bajo las condiciones reales del escenario.

    RECORTE PROPORCIONAL:
        Si el SOC no tiene suficiente energía para ejecutar la descarga
        planificada (arbitraje + activación up), se recorta proporcionalmente
        entre ambas. Se entrega lo que hay — no se cancela el intervalo.

    PENALIZACIÓN SOC FINAL:
        Al cerrar el día, si SOC_final ≠ SOC_INIT:
          · Déficit  → penalización = |desviación| × precio_medio_día
          · Exceso   → penalización =  desviación  × precio_medio_día
        Precio medio calculado sobre los precios reales positivos del día.
    """
    soc            = SOC_INIT
    beneficio_real = 0.0

    ingreso_spot_real      = 0.0
    ingreso_disp_real      = 0.0
    ingreso_act_real       = 0.0
    coste_deg_real         = 0.0
    energia_no_cargada     = 0.0
    energia_no_operada     = 0.0
    energia_recortada      = 0.0
    intervalos_soc_critico = 0

    p_reales    = escenario["precios_reales"]
    no_puja     = escenario["no_gana_puja"]
    fallo       = escenario["fallo_tecnico"]
    a_up_r      = escenario["a_up_real"]
    a_down_r    = escenario["a_down_real"]
    pi_disp     = escenario["pi_disp_real"]
    pi_act_up   = escenario["pi_act_up_real"]
    pi_act_down = escenario["pi_act_down_real"]

    # Precio medio del día real — referencia dinámica para penalización SOC
    precio_medio_dia = float(np.mean(np.maximum(p_reales, 0.0)))

    soc_series = []

    for t in range(96):
        p_real = p_reales[t]
        p_eff  = max(p_real, 0.0)

        x_ch_plan   = schedule["x_ch"][t]
        x_dis_plan  = schedule["x_dis"][t]
        x_sell_plan = schedule["x_sell"][t]
        x_buy_plan  = schedule["x_buy"][t]
        r_up_plan   = schedule["r_up"][t]
        r_down_plan = schedule["r_down"][t]

        # ------ Aplicar eventos de mercado/técnicos ------

        if fallo[t]:
            # Fallo técnico: la batería no opera en absoluto
            x_ch_real  = 0.0
            x_dis_real = 0.0
            a_up_t     = 0.0
            a_down_t   = 0.0
            energia_no_operada += x_ch_plan + x_dis_plan

        elif no_puja[t] and x_ch_plan > 0:
            # Puja de carga perdida: no carga, pero puede descargar/vender
            x_ch_real  = 0.0
            x_dis_real = x_dis_plan
            a_up_t     = a_up_r[t]
            a_down_t   = 0.0   # sin carga no puede absorber reserva down
            energia_no_cargada += x_ch_plan

        else:
            # Ejecución normal — activación puede diferir del previsto
            x_ch_real  = x_ch_plan
            x_dis_real = x_dis_plan
            a_up_t     = a_up_r[t]
            a_down_t   = a_down_r[t]

        # ------ Recorte proporcional por SOC insuficiente ------
        descarga_total = x_dis_real + a_up_t

        if descarga_total > 0:
            energia_disponible = max(soc - SOC_MIN, 0.0) * ETA
            if energia_disponible < descarga_total:
                ratio          = energia_disponible / descarga_total
                energia_recortada += descarga_total - energia_disponible
                x_dis_real     = x_dis_real * ratio
                a_up_t         = a_up_t     * ratio
                intervalos_soc_critico += 1

        # ------ Actualizar SOC ------
        soc_nuevo = (soc
                     + ETA * (x_ch_real + a_down_t)
                     - (x_dis_real + a_up_t) / ETA)

        soc = max(min(soc_nuevo, SOC_MAX), SOC_MIN)
        soc_series.append(round(soc, 4))

        # ------ Beneficio del intervalo ------
        ing_spot_t = p_real * x_sell_plan - p_eff * x_buy_plan
        ing_disp_t = pi_disp * (r_up_plan + r_down_plan)
        ing_act_t  = ((pi_act_up   - p_real) * a_up_t
                    + (pi_act_down - p_eff)  * a_down_t)
        deg_t      = C_DEG * (x_ch_real + x_dis_real + a_up_t + a_down_t)

        ben_t           = ing_spot_t + ing_disp_t + ing_act_t - deg_t
        beneficio_real += ben_t
        ingreso_spot_real  += ing_spot_t
        ingreso_disp_real  += ing_disp_t
        ingreso_act_real   += ing_act_t
        coste_deg_real     += deg_t

    # ------ Penalización SOC final ------
    soc_final      = soc
    desviacion_soc = soc_final - SOC_INIT   # + exceso, - déficit

    # Ambos casos penalizan: precio medio del día como referencia dinámica
    penalizacion_soc = abs(desviacion_soc) * precio_medio_dia
    beneficio_real  -= penalizacion_soc

    return {
        "beneficio_real [€]":       round(beneficio_real, 4),
        "beneficio_previsto [€]":   round(schedule["beneficio_previsto"], 4),
        "desviacion [€]":           round(beneficio_real - schedule["beneficio_previsto"], 4),
        "desviacion [%]":           round((beneficio_real - schedule["beneficio_previsto"])
                                          / max(abs(schedule["beneficio_previsto"]), 1) * 100, 2),
        "ingreso_spot_real [€]":    round(ingreso_spot_real, 4),
        "ingreso_disp_real [€]":    round(ingreso_disp_real, 4),
        "ingreso_act_real [€]":     round(ingreso_act_real, 4),
        "coste_deg_real [€]":       round(coste_deg_real, 4),
        "penalizacion_soc [€]":     round(penalizacion_soc, 4),
        "soc_final [MWh]":          round(soc_final, 4),
        "soc_final_desv [MWh]":     round(desviacion_soc, 4),
        "precio_medio_dia [€/MWh]": round(precio_medio_dia, 4),
        "energia_no_cargada [MWh]": round(energia_no_cargada, 4),
        "energia_no_operada [MWh]": round(energia_no_operada, 4),
        "energia_recortada [MWh]":  round(energia_recortada, 4),
        "intervalos_soc_critico":   intervalos_soc_critico,
        "pi_disp_real [€/MWh]":     round(escenario["pi_disp_real"], 4),
        "pi_act_up_real [€/MWh]":   round(escenario["pi_act_up_real"], 4),
        "pi_act_down_real [€/MWh]": round(escenario["pi_act_down_real"], 4),
        "factor_act_up":            round(escenario["factor_act_up"], 4),
        "factor_act_down":          round(escenario["factor_act_down"], 4),
        "n_pujas_perdidas":         int(escenario["no_gana_puja"].sum()),
        "n_fallos_tecnicos":        int(escenario["fallo_tecnico"].sum()),
        "extremo":                  escenario["extremo"],
        "soc_series":               soc_series,
    }


# =============================================================================
# PASO 4 — MONTE CARLO COMPLETO
# =============================================================================

def run_simulacion(
    precios_previstos: np.ndarray,
    n_normal: int           = 500,
    n_extremo: int          = 100,
    seed: int               = 42,
    sigma_spot: float       = 0.12,
    sigma_pi_disp: float    = 0.05,
    sigma_pi_act: float     = 0.10,
    p_no_puja: float        = 0.05,
    sigma_activacion: float = 0.15,
    p_fallo_tecnico: float  = 0.02,
    solver: str             = "highs",
) -> tuple:
    """
    Simulación Monte Carlo completa en dos modos:
      · Normal  (n_normal escenarios)  — distribución general
      · Extremo (n_extremo escenarios) — tail risk (P95/P99)

    Devuelve: (df_normal, df_extremo, schedule)
    """
    schedule = obtener_schedule(precios_previstos, solver=solver)
    rng      = np.random.default_rng(seed)

    kwargs_base = dict(
        sigma_spot=sigma_spot,
        sigma_pi_disp=sigma_pi_disp,
        sigma_pi_act=sigma_pi_act,
        p_no_puja=p_no_puja,
        sigma_activacion=sigma_activacion,
        p_fallo_tecnico=p_fallo_tecnico,
    )

    # Escenarios normales
    print(f"Simulando {n_normal} escenarios normales...")
    resultados_normal = []
    for i in range(n_normal):
        if i % 100 == 0:
            print(f"  {i+1}/{n_normal}", end="\r")
        esc = generar_escenario_ejecucion(schedule, rng, extremo=False, **kwargs_base)
        res = simular_ejecucion(schedule, esc)
        res["sim_id"] = i + 1
        resultados_normal.append({k: v for k, v in res.items() if k != "soc_series"})
    df_normal = pd.DataFrame(resultados_normal)

    # Escenarios extremos
    print(f"\nSimulando {n_extremo} escenarios extremos (tail risk)...")
    resultados_extremo = []
    for i in range(n_extremo):
        if i % 20 == 0:
            print(f"  {i+1}/{n_extremo}", end="\r")
        esc = generar_escenario_ejecucion(schedule, rng, extremo=True, **kwargs_base)
        res = simular_ejecucion(schedule, esc)
        res["sim_id"] = i + 1
        resultados_extremo.append({k: v for k, v in res.items() if k != "soc_series"})
    df_extremo = pd.DataFrame(resultados_extremo)

    _imprimir_resumen(df_normal, df_extremo, schedule)
    return df_normal, df_extremo, schedule


# =============================================================================
# RESUMEN
# =============================================================================

def _imprimir_resumen(df_normal: pd.DataFrame, df_extremo: pd.DataFrame, schedule: dict):
    ben_prev = schedule["beneficio_previsto"]
    ben_n    = df_normal["beneficio_real [€]"]
    ben_e    = df_extremo["beneficio_real [€]"]
    pen_n    = df_normal["penalizacion_soc [€]"]

    print(f"\n\n{'='*65}")
    print(f"  RESULTADOS SIMULACIÓN DE EJECUCIÓN REAL")
    print(f"{'='*65}")
    print(f"  Beneficio PREVISTO (modelo):     {ben_prev:>10.2f} €")
    print(f"\n  --- ESCENARIOS NORMALES ({len(df_normal)} sim.) ---")
    print(f"  Beneficio medio real:            {ben_n.mean():>10.2f} €")
    print(f"  Desviación media vs previsto:    {(ben_n - ben_prev).mean():>10.2f} €")
    print(f"  Penalización SOC media:          {pen_n.mean():>10.2f} €")
    print(f"  P10 (pesimista):                 {ben_n.quantile(0.10):>10.2f} €")
    print(f"  P50 (mediana):                   {ben_n.quantile(0.50):>10.2f} €")
    print(f"  P90 (optimista):                 {ben_n.quantile(0.90):>10.2f} €")
    print(f"  VaR 95% (peor 5%):               {ben_n.quantile(0.05):>10.2f} €")
    print(f"  % sim. con beneficio < 0:        {(ben_n < 0).mean()*100:>9.1f} %")
    print(f"\n  --- ESCENARIOS EXTREMOS ({len(df_extremo)} sim.) ---")
    print(f"  Beneficio medio (extremo):       {ben_e.mean():>10.2f} €")
    print(f"  Peor escenario absoluto:         {ben_e.min():>10.2f} €")
    print(f"  CVaR 95%:                        {ben_e[ben_e <= ben_e.quantile(0.05)].mean():>10.2f} €")
    print(f"  Intervalos SOC crítico (media):  {df_extremo['intervalos_soc_critico'].mean():>10.1f}")
    print(f"  Energía recortada media:         {df_extremo['energia_recortada [MWh]'].mean():>10.3f} MWh")
    print(f"  SOC final medio (extremo):       {df_extremo['soc_final [MWh]'].mean():>10.3f} MWh")
    print(f"  (SOC objetivo: {SOC_INIT:.3f} MWh)")
    print(f"{'='*65}\n")


# =============================================================================
# MAIN — uso directo como script
# =============================================================================

if __name__ == "__main__":

    from pathlib import Path
    import sys
    import pandas as pd

    NOMBRES_MES = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    COLUMNAS_Q = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]

    print("\n=== Simulador de Ejecución Real — Día Suelto ===")
    while True:
        entrada = input("Fecha (DD/MM/YYYY o YYYY-MM-DD): ").strip()
        try:
            fecha_dt  = pd.to_datetime(entrada, dayfirst=("/" in entrada))
            fecha_str = fecha_dt.strftime("%Y-%m-%d")
            break
        except Exception:
            print("  [!] Formato no reconocido.")

    mes, anio = fecha_dt.month, fecha_dt.year
    csv_path  = Path("Precios") / f"precios_{NOMBRES_MES[mes].lower()}_{anio}.csv"

    if not csv_path.exists():
        print(f"[!] No se encuentra '{csv_path}'. Ejecuta primero parseo_omie.py.")
        sys.exit()

    df_p = pd.read_csv(csv_path)
    fila = df_p[df_p["fecha"] == fecha_str]
    if fila.empty:
        print(f"[!] Fecha '{fecha_str}' no encontrada en el CSV.")
        sys.exit()

    precios = fila[COLUMNAS_Q].values[0].astype(float)

    n_sim_str = input("Número de simulaciones normales (Enter = 200): ").strip()
    n_sim     = int(n_sim_str) if n_sim_str else 200

    df_normal, df_extremo, schedule = run_simulacion(
        precios_previstos = precios,
        n_normal          = n_sim,
        n_extremo         = max(n_sim // 5, 20),
        seed              = 42,
    )

    carpeta = Path("Resultados_Sim") / "dias_sueltos"
    carpeta.mkdir(parents=True, exist_ok=True)
    df_normal.to_csv(carpeta  / f"sim_normal_{fecha_str}.csv",  index=False)
    df_extremo.to_csv(carpeta / f"sim_extremo_{fecha_str}.csv", index=False)
    print(f"\nCSVs guardados en '{carpeta}/'")