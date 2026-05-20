"""
Simulador Mensual de Ejecución Real — Batería Pura (Arbitraje + aFRR)
Autor: Hugo Raggini Paternain
----------------------------------------------------------------------
Integración con el ecosistema existente:
    parseo.omie         → datos/precios/precios_{mes}_{año}.csv
    optimizacion.bateria → construir_modelo()
    simulacion.dia       → generar_escenario_ejecucion(), simular_ejecucion()

Flujo:
    Para cada día del mes:
        1. Carga precios reales del CSV mensual (datos/precios/)
        2. Optimiza el schedule con esos precios
        3. Simula N escenarios de ejecución real con aleatoriedad
        4. Guarda CSV diario + acumula resumen mensual

Outputs:
    resultados/simulacion/{YYYY-MM}/simulacion_{fecha}.csv
    resultados/simulacion/{YYYY-MM}/resumen_sim_{YYYY}_{MM}.csv

Uso:
    python -m simulacion.mes              # interactivo
    python -m simulacion.mes 1 2026       # enero 2026
    python -m simulacion.mes 3 2026 200   # marzo 2026, 200 sim. normales
"""

import sys
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimizacion.bateria import construir_modelo
from simulacion.dia import (
    generar_escenario_ejecucion,
    simular_ejecucion,
)

# =============================================================================
# CONFIGURACION
# =============================================================================

NOMBRES_MES = {
    1: "Enero",      2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",       6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre",  12: "Diciembre"
}

COLUMNAS_Q      = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]
CARPETA_PRECIOS = ROOT / "datos" / "precios"

SOLVER = "highs"
SOLVER_OPTIONS = {
    "threads":        4,
    "time_limit":     60,
    "mip_rel_gap":    0.001,
    "output_flag":    0,
    "log_to_console": 0,
}

PARAMS_SIM = {
    "sigma_spot":       0.12,
    "sigma_pi_disp":    0.05,
    "sigma_pi_act":     0.10,
    "p_no_puja":        0.05,
    "sigma_activacion": 0.15,
    "p_fallo_tecnico":  0.02,
}

# =============================================================================
# ARGUMENTOS
# =============================================================================

if len(sys.argv) >= 3:
    MES  = int(sys.argv[1])
    ANIO = int(sys.argv[2])
else:
    MES  = int(input("Mes (numero 1-12): "))
    ANIO = int(input("Anio (ej. 2026): "))

N_NORMAL  = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
N_EXTREMO = max(N_NORMAL // 5, 20)

CSV_PRECIOS    = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[MES].lower()}_{ANIO}.csv"
CARPETA_SALIDA = ROOT / "resultados" / "simulacion" / f"{ANIO}-{MES:02d}"
SEED_BASE      = 42


# =============================================================================
# OBTENER SCHEDULE DE UN DIA
# =============================================================================

def obtener_schedule_dia(precios: np.ndarray, opt) -> dict:
    """
    Optimiza el schedule para un dia dado sus precios previstos.
    Devuelve el schedule fijo o None si falla.
    """
    try:
        model   = construir_modelo(precios)
        results = opt.solve(model, tee=False)
        status  = results.solver.termination_condition

        if status not in (pyo.TerminationCondition.optimal,
                          pyo.TerminationCondition.feasible,
                          pyo.TerminationCondition.maxTimeLimit):
            return None

        return {
            "x_ch":               np.array([pyo.value(model.x_ch[t])   for t in range(1, 97)]),
            "x_dis":              np.array([pyo.value(model.x_dis[t])  for t in range(1, 97)]),
            "x_sell":             np.array([pyo.value(model.x_sell[t]) for t in range(1, 97)]),
            "x_buy":              np.array([pyo.value(model.x_buy[t])  for t in range(1, 97)]),
            "r_up":               np.array([pyo.value(model.r_up[t])   for t in range(1, 97)]),
            "r_down":             np.array([pyo.value(model.r_down[t]) for t in range(1, 97)]),
            "a_up_prev":          np.array([pyo.value(model.a_up[t])   for t in range(1, 97)]),
            "a_down_prev":        np.array([pyo.value(model.a_down[t]) for t in range(1, 97)]),
            "beneficio_previsto": pyo.value(model.obj),
            "precios_previstos":  precios.copy(),
        }
    except Exception:
        return None


# =============================================================================
# SIMULAR UN DIA COMPLETO
# =============================================================================

def simular_dia(
    fecha: str,
    precios: np.ndarray,
    opt,
    n_normal: int,
    n_extremo: int,
    seed: int,
) -> tuple:
    """
    Para un dia dado:
      1. Obtiene el schedule optimo
      2. Simula n_normal escenarios normales + n_extremo extremos
      3. Devuelve (df_simulaciones, fila_resumen)
    """
    schedule = obtener_schedule_dia(precios, opt)
    if schedule is None:
        return None, None

    rng   = np.random.default_rng(seed)
    filas = []

    for i in range(n_normal):
        esc = generar_escenario_ejecucion(schedule, rng, extremo=False, **PARAMS_SIM)
        res = simular_ejecucion(schedule, esc)
        filas.append({
            "fecha":  fecha,
            "sim_id": i + 1,
            "tipo":   "normal",
            **{k: v for k, v in res.items() if k != "soc_series"},
        })

    for i in range(n_extremo):
        esc = generar_escenario_ejecucion(schedule, rng, extremo=True, **PARAMS_SIM)
        res = simular_ejecucion(schedule, esc)
        filas.append({
            "fecha":  fecha,
            "sim_id": n_normal + i + 1,
            "tipo":   "extremo",
            **{k: v for k, v in res.items() if k != "soc_series"},
        })

    df_dia = pd.DataFrame(filas)
    ben_n  = df_dia[df_dia["tipo"] == "normal"]["beneficio_real [€]"]
    ben_e  = df_dia[df_dia["tipo"] == "extremo"]["beneficio_real [€]"]
    df_nor = df_dia[df_dia["tipo"] == "normal"]
    df_ext = df_dia[df_dia["tipo"] == "extremo"]

    fila_resumen = {
        "fecha":                         fecha,
        "beneficio_previsto [€]":        round(schedule["beneficio_previsto"], 2),
        # Normales
        "ben_medio [€]":                 round(ben_n.mean(), 2),
        "ben_std [€]":                   round(ben_n.std(), 2),
        "ben_P10 [€]":                   round(ben_n.quantile(0.10), 2),
        "ben_P50 [€]":                   round(ben_n.quantile(0.50), 2),
        "ben_P90 [€]":                   round(ben_n.quantile(0.90), 2),
        "VaR_95 [€]":                    round(ben_n.quantile(0.05), 2),
        "pct_negativo [%]":              round((ben_n < 0).mean() * 100, 1),
        "desv_media [€]":                round((ben_n - schedule["beneficio_previsto"]).mean(), 2),
        "desv_media [%]":                round((ben_n - schedule["beneficio_previsto"]).mean()
                                               / max(abs(schedule["beneficio_previsto"]), 1) * 100, 1),
        # Nuevas metricas de penalizacion y recorte
        "penalizacion_soc_media [€]":    round(df_nor["penalizacion_soc [€]"].mean(), 2),
        "energia_recortada_media [MWh]": round(df_nor["energia_recortada [MWh]"].mean(), 4),
        "soc_final_desv_media [MWh]":    round(df_nor["soc_final_desv [MWh]"].mean(), 4),
        # Extremos
        "ben_extremo_medio [€]":         round(ben_e.mean(), 2),
        "peor_escenario [€]":            round(ben_e.min(), 2),
        "CVaR_95 [€]":                   round(ben_e[ben_e <= ben_e.quantile(0.05)].mean(), 2),
        "soc_critico_medio":             round(df_ext["intervalos_soc_critico"].mean(), 1),
        "penalizacion_soc_ext [€]":      round(df_ext["penalizacion_soc [€]"].mean(), 2),
        "energia_recortada_ext [MWh]":   round(df_ext["energia_recortada [MWh]"].mean(), 4),
        # Operacional
        "pujas_perdidas_media":          round(df_nor["n_pujas_perdidas"].mean(), 1),
        "fallos_tecnicos_media":         round(df_nor["n_fallos_tecnicos"].mean(), 1),
        "n_sim_normal":                  n_normal,
        "n_sim_extremo":                 n_extremo,
    }

    return df_dia, fila_resumen


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    if not CSV_PRECIOS.exists():
        print(f"[!] No se encuentra '{CSV_PRECIOS}'.")
        print(f"    Ejecuta primero: python -m parseo.omie {MES} {ANIO}")
        sys.exit()

    df_precios = pd.read_csv(CSV_PRECIOS)
    for col in COLUMNAS_Q:
        if col not in df_precios.columns:
            print(f"[!] Columna '{col}' no encontrada en {CSV_PRECIOS}")
            sys.exit()

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SIMULADOR MENSUAL — {NOMBRES_MES[MES]} {ANIO}")
    print(f"  Dias a simular    : {len(df_precios)}")
    print(f"  Sim. normales/dia : {N_NORMAL}")
    print(f"  Sim. extremas/dia : {N_EXTREMO}")
    print(f"  Precios desde     : {CSV_PRECIOS}")
    print(f"  Carpeta salida    : {CARPETA_SALIDA}")
    print(f"{'='*65}\n")

    opt = pyo.SolverFactory(SOLVER)
    for k, v in SOLVER_OPTIONS.items():
        opt.options[k] = v

    resumen_mensual = []
    dias_ok  = 0
    dias_err = 0

    for idx, fila_precio in df_precios.iterrows():
        fecha   = fila_precio["fecha"]
        precios = fila_precio[COLUMNAS_Q].values.astype(float)

        print(f"  -> {fecha} ... ", end="", flush=True)

        seed_dia = SEED_BASE + idx

        df_dia, fila_res = simular_dia(
            fecha=fecha,
            precios=precios,
            opt=opt,
            n_normal=N_NORMAL,
            n_extremo=N_EXTREMO,
            seed=seed_dia,
        )

        if df_dia is None:
            print("ERROR (schedule fallido)")
            dias_err += 1
            continue

        csv_dia = CARPETA_SALIDA / f"simulacion_{fecha}.csv"
        df_dia.to_csv(csv_dia, index=False)
        resumen_mensual.append(fila_res)
        dias_ok += 1

        ben_prev = fila_res["beneficio_previsto [€]"]
        ben_p50  = fila_res["ben_P50 [€]"]
        var      = fila_res["VaR_95 [€]"]
        pen      = fila_res["penalizacion_soc_media [€]"]
        print(f"OK  prev={ben_prev:+.2f}E  P50={ben_p50:+.2f}E  VaR95={var:+.2f}E  penSOC={pen:.2f}E")

    df_resumen  = pd.DataFrame(resumen_mensual)
    csv_resumen = CARPETA_SALIDA / f"resumen_sim_{ANIO}_{MES:02d}.csv"
    df_resumen.to_csv(csv_resumen, index=False)

    print(f"\n{'='*65}")
    print(f"  {NOMBRES_MES[MES]} {ANIO} — SIMULACION COMPLETADA")
    print(f"{'='*65}")
    print(f"  Dias simulados    : {dias_ok}/{len(df_precios)}")
    if dias_err:
        print(f"  Dias con error    : {dias_err}")

    if not df_resumen.empty:
        prev_total = df_resumen["beneficio_previsto [€]"].sum()
        p50_total  = df_resumen["ben_P50 [€]"].sum()
        coste_inc  = prev_total - p50_total

        print(f"\n  --- ESCENARIOS NORMALES ---")
        print(f"  Beneficio previsto total  : {prev_total:>10.2f} E")
        print(f"  Beneficio P50 total       : {p50_total:>10.2f} E")
        print(f"  Coste incertidumbre total : {coste_inc:>10.2f} E ({coste_inc/max(abs(prev_total),1)*100:.1f}%)")
        print(f"  P10 acumulado (pesimista) : {df_resumen['ben_P10 [€]'].sum():>10.2f} E")
        print(f"  P90 acumulado (optimista) : {df_resumen['ben_P90 [€]'].sum():>10.2f} E")
        print(f"  Penaliz. SOC total media  : {df_resumen['penalizacion_soc_media [€]'].sum():>10.2f} E")
        print(f"  Energia recortada media   : {df_resumen['energia_recortada_media [MWh]'].mean():>10.4f} MWh/dia")
        print(f"\n  --- ESCENARIOS EXTREMOS ---")
        print(f"  Peor dia absoluto         : {df_resumen['peor_escenario [€]'].min():>10.2f} E")
        print(f"  CVaR 95% medio            : {df_resumen['CVaR_95 [€]'].mean():>10.2f} E")
        print(f"\n  Resumen guardado : {csv_resumen}")
        print(f"  CSVs diarios en  : {CARPETA_SALIDA}/")
    print(f"{'='*65}\n")