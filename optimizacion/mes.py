"""
Optimización Mensual de Batería — Loop Diario
Autor: Hugo Raggini Paternain

Lee el CSV de precios mensuales generado por parseo.omie,
corre la optimización diaria para cada día y guarda un CSV
de resultados por día en: resultados/optimizacion/YYYY-MM/

Uso:
    python -m optimizacion.mes              # interactivo
    python -m optimizacion.mes 1 2026       # enero 2026
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimizacion.bateria import construir_modelo, extraer_resultados

# =============================================================================
# CONFIGURACIÓN
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

# --- Mes y año ---
if len(sys.argv) == 3:
    MES  = int(sys.argv[1])
    ANIO = int(sys.argv[2])
else:
    MES  = int(input("Mes (número 1-12): "))
    ANIO = int(input("Año (ej. 2026): "))

CSV_PRECIOS    = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[MES].lower()}_{ANIO}.csv"
CARPETA_SALIDA = ROOT / "resultados" / "optimizacion" / f"{ANIO}-{MES:02d}"

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Cargar precios ---
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

    print(f"\n=== Optimización mensual — {NOMBRES_MES[MES]} {ANIO} ===")
    print(f"    Precios desde    : {CSV_PRECIOS}")
    print(f"    Días a optimizar : {len(df_precios)}")
    print(f"    Carpeta salida   : {CARPETA_SALIDA}\n")

    opt = pyo.SolverFactory(SOLVER)
    for key, val in SOLVER_OPTIONS.items():
        opt.options[key] = val

    resumen = []

    for _, fila in df_precios.iterrows():
        fecha   = fila["fecha"]
        precios = fila[COLUMNAS_Q].values.astype(float)

        print(f"  -> {fecha} ...", end=" ", flush=True)

        try:
            model = construir_modelo(precios)
            res   = opt.solve(model, tee=False)

            status = res.solver.termination_condition
            if status not in (pyo.TerminationCondition.optimal,
                              pyo.TerminationCondition.maxTimeLimit):
                raise RuntimeError(f"Solver terminó con estado: {status}")

            df_dia, beneficio = extraer_resultados(model)
            df_dia.insert(0, "fecha", fecha)

            csv_dia = CARPETA_SALIDA / f"resultado_{fecha}.csv"
            df_dia.to_csv(csv_dia, index=False)

            resumen.append({
                "fecha":                    fecha,
                "beneficio [€]":            round(beneficio, 2),
                "energia_vendida [MWh]":    round(df_dia["x_sell [MWh]"].sum(), 3),
                "energia_comprada [MWh]":   round(df_dia["x_buy [MWh]"].sum(), 3),
                "energia_cargada [MWh]":    round(df_dia["x_ch [MWh]"].sum(), 3),
                "energia_descargada [MWh]": round(df_dia["x_dis [MWh]"].sum(), 3),
                "reserva_up [MWh]":         round(df_dia["r_up [MWh]"].sum(), 3),
                "reserva_down [MWh]":       round(df_dia["r_down [MWh]"].sum(), 3),
                "activacion_up [MWh]":      round(df_dia["a_up [MWh]"].sum(), 3),
                "activacion_down [MWh]":    round(df_dia["a_down [MWh]"].sum(), 3),
                "precio_medio [€/MWh]":     round(float(np.mean(precios)), 2),
                "precio_max [€/MWh]":       round(float(np.max(precios)), 2),
                "precio_min [€/MWh]":       round(float(np.min(precios)), 2),
                "spread [€/MWh]":           round(float(np.max(precios) - np.min(precios)), 2),
            })

            print(f"OK — {beneficio:+.2f} €")

        except Exception as e:
            print(f"ERROR — {e}")
            resumen.append({"fecha": fecha, "beneficio [€]": None, "error": str(e)})

    # --- Guardar resumen mensual ---
    df_resumen  = pd.DataFrame(resumen)
    csv_resumen = CARPETA_SALIDA / f"resumen_{ANIO}_{MES:02d}.csv"
    df_resumen.to_csv(csv_resumen, index=False)

    beneficios_ok = df_resumen["beneficio [€]"].dropna()
    dias_ok  = len(beneficios_ok)
    dias_err = len(df_resumen) - dias_ok

    print(f"\n{'='*50}")
    print(f"  {NOMBRES_MES[MES]} {ANIO} — RESULTADOS FINALES")
    print(f"{'='*50}")
    print(f"  Días optimizados : {dias_ok}/{len(df_resumen)}")
    if dias_err:
        print(f"  Días con error   : {dias_err}")
    print(f"  Beneficio total  : {beneficios_ok.sum():+.2f} €")
    print(f"  Beneficio medio  : {beneficios_ok.mean():+.2f} €/día")
    print(f"  Mejor día        : {df_resumen.loc[df_resumen['beneficio [€]'].idxmax(), 'fecha']}  ({beneficios_ok.max():+.2f} €)")
    print(f"  Peor día         : {df_resumen.loc[df_resumen['beneficio [€]'].idxmin(), 'fecha']}  ({beneficios_ok.min():+.2f} €)")
    print(f"\n  Resumen guardado : {csv_resumen}")
    print(f"  CSVs diarios en  : {CARPETA_SALIDA}/")