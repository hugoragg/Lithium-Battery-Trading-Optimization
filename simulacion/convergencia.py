"""
Análisis de Convergencia Monte Carlo
Autor: Hugo Raggini Paternain
-------------------------------
Comprueba si N=200 simulaciones es suficiente ejecutando el mismo día
con distintos valores de N y comprobando si P50 y VaR95 se estabilizan.

Uso:
    python -m simulacion.convergencia              # interactivo
    python -m simulacion.convergencia 2026-01-01   # fecha directa
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimizacion.bateria import construir_modelo
from simulacion.dia import generar_escenario_ejecucion, simular_ejecucion
import pyomo.environ as pyo

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

NOMBRES_MES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}
COLUMNAS_Q      = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]
CARPETA_PRECIOS = ROOT / "datos" / "precios"

# Valores de N a probar
N_VALORES = [25, 50, 100, 150, 200, 300, 500]

# Repeticiones por cada N (para medir varianza del estimador)
N_REPETICIONES = 10

SEED_BASE = 42

PARAMS_SIM = {
    "sigma_spot":            0.12,
    "sigma_pi_disp_up":      0.1786,
    "sigma_pi_disp_down":    0.3642,
    "sigma_pi_act_up":       0.4195,
    "sigma_pi_act_down":     1.0268,
    "p_no_puja":             0.05,
    "sigma_activacion_up":   0.4229,
    "sigma_activacion_down": 0.4334,
    "p_fallo_tecnico":       0.02,
}

# Estilo
BG, PBG = "#F5F6FA", "#FFFFFF"
CS, CB, CF, CACC = "#27AE60", "#E74C3C", "#F5A623", "#2C3E50"
TKW = dict(fontsize=10, fontweight="bold", color="#2C3E50", pad=12, loc="left")
GKW = dict(fontsize=8, framealpha=0.8, edgecolor="none")

# =============================================================================
# CARGA DE DATOS Y SCHEDULE
# =============================================================================

if len(sys.argv) == 2:
    entrada = sys.argv[1].strip()
else:
    print("\n=== Análisis de Convergencia Monte Carlo ===")
    entrada = input("Fecha (DD/MM/YYYY o YYYY-MM-DD): ").strip()

try:
    fecha_dt  = pd.to_datetime(entrada, dayfirst=("/" in entrada))
    fecha_str = fecha_dt.strftime("%Y-%m-%d")
except Exception:
    print("[!] Formato no reconocido.")
    sys.exit()

mes, anio = fecha_dt.month, fecha_dt.year
csv_path  = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[mes].lower()}_{anio}.csv"

if not csv_path.exists():
    print(f"[!] No se encuentra '{csv_path}'.")
    sys.exit()

df_p    = pd.read_csv(csv_path)
fila    = df_p[df_p["fecha"] == fecha_str]
if fila.empty:
    print(f"[!] Fecha '{fecha_str}' no encontrada.")
    sys.exit()

precios = fila[COLUMNAS_Q].values[0].astype(float)

# Obtener schedule (una sola vez)
print(f"\n  Optimizando schedule para {fecha_str}...")
model = construir_modelo(precios)
opt   = pyo.SolverFactory("highs")
opt.options.update({"time_limit": 120, "mip_rel_gap": 0.001,
                    "output_flag": 0, "log_to_console": 0})
opt.solve(model, tee=False)

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
    "precios_previstos":  precios.copy(),
}

ben_prev = schedule["beneficio_previsto"]
print(f"  Schedule obtenido. Beneficio previsto: {ben_prev:.2f} €\n")

# =============================================================================
# ANÁLISIS DE CONVERGENCIA
# =============================================================================

print(f"  Analizando convergencia con {N_REPETICIONES} repeticiones por N...\n")
print(f"  {'N':>6}  {'P50 medio':>10}  {'P50 std':>8}  {'VaR95 medio':>12}  {'VaR95 std':>10}")
print(f"  {'-'*55}")

resultados = []

for n in N_VALORES:
    p50_reps   = []
    var95_reps = []

    for rep in range(N_REPETICIONES):
        seed = SEED_BASE + rep * 1000
        rng  = np.random.default_rng(seed)
        bens = []

        for _ in range(n):
            esc = generar_escenario_ejecucion(schedule, rng, **PARAMS_SIM)
            res = simular_ejecucion(schedule, esc)
            bens.append(res["beneficio_real [€]"])

        bens = np.array(bens)
        p50_reps.append(np.percentile(bens, 50))
        var95_reps.append(np.percentile(bens, 5))

    p50_m   = np.mean(p50_reps)
    p50_s   = np.std(p50_reps)
    var95_m = np.mean(var95_reps)
    var95_s = np.std(var95_reps)

    resultados.append({
        "N":          n,
        "P50_medio":  p50_m,
        "P50_std":    p50_s,
        "VaR95_medio":var95_m,
        "VaR95_std":  var95_s,
    })

    print(f"  {n:>6}  {p50_m:>10.2f}€  {p50_s:>7.2f}€  {var95_m:>12.2f}€  {var95_s:>9.2f}€")

df_conv = pd.DataFrame(resultados)

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.canvas.manager.set_window_title(f"Convergencia Monte Carlo — {fecha_str}")
fig.suptitle(
    f"ANÁLISIS DE CONVERGENCIA MONTE CARLO — {fecha_str}    "
    f"[Previsto: {ben_prev:.2f}€  |  Repeticiones por N: {N_REPETICIONES}]",
    fontsize=11, fontweight="bold", y=0.97
)

def _ax(ax, ylabel=""):
    ax.set_facecolor(PBG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555", labelsize=8)
    ax.grid(axis="y", alpha=0.2, lw=0.5, color="#AAAAAA")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#555")
    ax.set_xlabel("N simulaciones", fontsize=9, color="#555")

# Gráfica 1 — Convergencia P50
ax = axes[0]
ax.plot(df_conv["N"], df_conv["P50_medio"], color=CS, lw=2.5,
        marker="o", ms=6, label="P50 medio")
ax.fill_between(df_conv["N"],
                df_conv["P50_medio"] - df_conv["P50_std"],
                df_conv["P50_medio"] + df_conv["P50_std"],
                alpha=0.2, color=CS, label="±1 std entre repeticiones")
ax.axhline(ben_prev, color=CACC, lw=1.5, ls="--", label=f"Previsto: {ben_prev:.2f}€")

# Marcar N=200
if 200 in df_conv["N"].values:
    val_200 = df_conv.loc[df_conv["N"] == 200, "P50_medio"].values[0]
    ax.axvline(200, color=CF, lw=1.5, ls=":", label="N=200 (actual)")
    ax.annotate(f"{val_200:.1f}€", xy=(200, val_200),
                xytext=(220, val_200 + df_conv["P50_std"].mean()),
                fontsize=8, color=CF, fontweight="bold")

ax.set_title("① Convergencia del P50", **TKW)
_ax(ax, "P50 (€)")
ax.legend(**GKW)

# Gráfica 2 — Convergencia VaR95
ax = axes[1]
ax.plot(df_conv["N"], df_conv["VaR95_medio"], color=CB, lw=2.5,
        marker="s", ms=6, label="VaR95 medio")
ax.fill_between(df_conv["N"],
                df_conv["VaR95_medio"] - df_conv["VaR95_std"],
                df_conv["VaR95_medio"] + df_conv["VaR95_std"],
                alpha=0.2, color=CB, label="±1 std entre repeticiones")

if 200 in df_conv["N"].values:
    val_200 = df_conv.loc[df_conv["N"] == 200, "VaR95_medio"].values[0]
    ax.axvline(200, color=CF, lw=1.5, ls=":", label="N=200 (actual)")
    ax.annotate(f"{val_200:.1f}€", xy=(200, val_200),
                xytext=(220, val_200 + df_conv["VaR95_std"].mean()),
                fontsize=8, color=CF, fontweight="bold")

ax.set_title("② Convergencia del VaR 95%", **TKW)
_ax(ax, "VaR 95% (€)")
ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])

# =============================================================================
# CONCLUSIÓN AUTOMÁTICA
# =============================================================================

# Criterio: std del P50 < 1% del P50 medio → convergido
p50_ref    = df_conv.loc[df_conv["N"] == max(N_VALORES), "P50_medio"].values[0]
umbral_pct = 0.01

print(f"\n  {'='*55}")
print(f"  CONCLUSIÓN")
print(f"  {'='*55}")
for _, row in df_conv.iterrows():
    estable_p50  = row["P50_std"]   / abs(p50_ref) < umbral_pct
    estable_var  = row["VaR95_std"] / abs(p50_ref) < umbral_pct
    estado = "ESTABLE" if (estable_p50 and estable_var) else "inestable"
    print(f"  N={row['N']:>4}  P50_std={row['P50_std']:>6.2f}€  "
          f"VaR95_std={row['VaR95_std']:>6.2f}€  → {estado}")

# N mínimo recomendado
n_recomendado = None
for _, row in df_conv.iterrows():
    if (row["P50_std"] / abs(p50_ref) < umbral_pct and
        row["VaR95_std"] / abs(p50_ref) < umbral_pct):
        n_recomendado = int(row["N"])
        break

if n_recomendado:
    print(f"\n  N mínimo recomendado: {n_recomendado} simulaciones")
    if n_recomendado <= 200:
        print(f"  Tu N=200 es suficiente.")
    else:
        print(f"  Considera aumentar a N={n_recomendado}.")
else:
    print(f"\n  Ningún N probado alcanza convergencia con umbral {umbral_pct*100:.0f}%.")
    print(f"  Considera aumentar N_VALORES o relajar el umbral.")

print(f"  {'='*55}\n")

# Guardar CSV de convergencia
csv_out = ROOT / "resultados" / "simulacion" / "dias_sueltos" / f"convergencia_{fecha_str}.csv"
csv_out.parent.mkdir(parents=True, exist_ok=True)
df_conv.to_csv(csv_out, index=False)
print(f"  CSV guardado: {csv_out}")

plt.show()