"""
Visualización de Simulación — Día Suelto
Autor: Hugo Raggini Paternain

Lee los CSVs de simulación generados por simulador_ejecucion.py
y los compara con el beneficio previsto del modelo.

Uso:
    python visualizacion_simulacion_dia.py              # interactivo
    python visualizacion_simulacion_dia.py 2026-01-01   # fecha directa
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CARPETA_SIM = Path("Resultados_Sim") / "dias_sueltos"
CARPETA_DET = Path("Resultados Días Sueltos")

SOC_INIT = 1.0
E_MAX    = 2.0
SOC_MIN  = E_MAX * (1 - 0.93)

# --- Fecha ---
if len(sys.argv) == 2:
    entrada = sys.argv[1].strip()
else:
    print("\n=== Visualización Simulación — Día Suelto ===")
    entrada = input("Fecha (DD/MM/YYYY o YYYY-MM-DD): ").strip()

try:
    fecha_dt  = pd.to_datetime(entrada, dayfirst=("/" in entrada))
    fecha_str = fecha_dt.strftime("%Y-%m-%d")
except Exception:
    print("[!] Formato de fecha no reconocido.")
    sys.exit()

# =============================================================================
# CARGA DE DATOS
# =============================================================================

csv_normal  = CARPETA_SIM / f"sim_normal_{fecha_str}.csv"
csv_extremo = CARPETA_SIM / f"sim_extremo_{fecha_str}.csv"

for f in [csv_normal, csv_extremo]:
    if not f.exists():
        print(f"[!] No se encuentra '{f}'. Ejecuta primero simulador_ejecucion.py.")
        sys.exit()

df_n = pd.read_csv(csv_normal)
df_e = pd.read_csv(csv_extremo)

# SOC previsto — desde Resultados Días Sueltos si existe
csv_det = CARPETA_DET / f"resultado_{fecha_str}.csv"
df_det  = pd.read_csv(csv_det) if csv_det.exists() else None

print(f"  Simulaciones normales : {len(df_n)}")
print(f"  Simulaciones extremas : {len(df_e)}\n")

ben_prev = df_n["beneficio_previsto [€]"].iloc[0]

# =============================================================================
# ESTILO
# =============================================================================

CF, CS, CB, CCH, CDI, CSOC, CP = "#F5A623", "#27AE60", "#E74C3C", "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22"
CRU, CRD, CAU, CAD, CACC, BG, PBG = "#E74C3C", "#3498DB", "#C0392B", "#2980B9", "#2C3E50", "#F5F6FA", "#FFFFFF"
CGRIS = "#95A5A6"

TKW = dict(fontsize=10, fontweight="bold", color="#2C3E50", pad=12, loc="left")
LKW = dict(fontsize=9, color="#555")
GKW = dict(fontsize=8, framealpha=0.8, edgecolor="none", ncol=2)

def _ax(ax, ylabel=""):
    ax.set_facecolor(PBG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555", labelsize=8)
    ax.grid(axis="y", alpha=0.2, lw=0.5, color="#AAAAAA")
    if ylabel:
        ax.set_ylabel(ylabel, **LKW)

xt_pos = np.arange(0, 96, 4)
xt_lbl = [f"{h:02d}h" for h in range(24)]

def _xt(ax):
    ax.set_xticks(xt_pos)
    ax.set_xticklabels(xt_lbl, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-1, 96)

# =============================================================================
# DASHBOARD 1 — DISTRIBUCIÓN DE BENEFICIOS
# =============================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig1.canvas.manager.set_window_title(f"Simulación — Distribución | {fecha_str}")

ben_n  = df_n["beneficio_real [€]"]
ben_e  = df_e["beneficio_real [€]"]
p10    = ben_n.quantile(0.10)
p50    = ben_n.quantile(0.50)
p90    = ben_n.quantile(0.90)
var95  = ben_n.quantile(0.05)
coste_incert = ben_prev - p50

fig1.suptitle(
    f"DISTRIBUCIÓN DE BENEFICIOS — {fecha_str}    "
    f"[Previsto: {ben_prev:+.2f}€  |  P50 real: {p50:+.2f}€  |  "
    f"Coste incertidumbre: {coste_incert:+.2f}€ ({coste_incert/max(abs(ben_prev),1)*100:.1f}%)]",
    fontsize=11, fontweight="bold", y=0.97
)

# ① Histograma normales con percentiles y previsto
ax = axes1[0, 0]
rango = (min(ben_n.min(), ben_e.min()) * 1.05,
         max(ben_n.max(), ben_e.max()) * 1.05)
ax.hist(ben_n.values, bins=40, range=rango, color=CCH, alpha=0.7,
        label=f"Normales (n={len(df_n)})")
ax.axvline(p10,      color=CB,   lw=1.5, ls="--", label=f"P10: {p10:.1f}€")
ax.axvline(p50,      color=CS,   lw=2,   ls="-",  label=f"P50: {p50:.1f}€")
ax.axvline(p90,      color=CAU,  lw=1.5, ls="--", label=f"P90: {p90:.1f}€")
ax.axvline(var95,    color=CB,   lw=2,   ls=":",  label=f"VaR95: {var95:.1f}€")
ax.axvline(ben_prev, color=CACC, lw=2,   ls="-",  label=f"Previsto: {ben_prev:.1f}€")
ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("① Distribución Beneficios — Normales", **TKW)
_ax(ax, "frecuencia")
ax.set_xlabel("Beneficio (€)", fontsize=8, color="#555")
ax.legend(fontsize=7, framealpha=0.8, edgecolor="none")

# ② Normales vs extremos superpuestos (densidad)
ax = axes1[0, 1]
ax.hist(ben_n.values, bins=40, range=rango, color=CCH, alpha=0.6,
        density=True, label=f"Normales  med={ben_n.mean():.1f}€")
ax.hist(ben_e.values, bins=40, range=rango, color=CB,  alpha=0.6,
        density=True, label=f"Extremos  med={ben_e.mean():.1f}€")
ax.axvline(ben_prev, color=CACC, lw=2, ls="--", label=f"Previsto: {ben_prev:.1f}€")
ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("② Normales vs Extremos (densidad)", **TKW)
_ax(ax, "densidad")
ax.set_xlabel("Beneficio (€)", fontsize=8, color="#555")
ax.legend(fontsize=7, framealpha=0.8, edgecolor="none")

# ③ Box plot comparativo
ax = axes1[1, 0]
bp = ax.boxplot([ben_n.values, ben_e.values],
                labels=["Normales", "Extremos"],
                patch_artist=True,
                medianprops=dict(color="white", lw=2))
for patch, color in zip(bp["boxes"], [CCH, CB]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(ben_prev, color=CACC, lw=1.5, ls="--", label=f"Previsto: {ben_prev:.1f}€")
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("③ Box Plot — Normales vs Extremos", **TKW)
_ax(ax, "€"); ax.legend(**GKW)

# ④ KPIs resumen
ax = axes1[1, 1]
ax.axis("off")
metricas = [
    ("Beneficio previsto",   f"{ben_prev:+.2f} €",                   CACC),
    ("P10 (pesimista)",      f"{p10:+.2f} €",                        CB),
    ("P50 (mediana)",        f"{p50:+.2f} €",                        CS),
    ("P90 (optimista)",      f"{p90:+.2f} €",                        CAU),
    ("VaR 95%",              f"{var95:+.2f} €",                      CB),
    ("Coste incertidumbre",  f"{coste_incert:+.2f} € ({coste_incert/max(abs(ben_prev),1)*100:.1f}%)", CF),
    ("Media extremos",       f"{ben_e.mean():+.2f} €",               CB),
    ("Peor escenario",       f"{ben_e.min():+.2f} €",                CRU),
    ("% sim. negativas",     f"{(ben_n < 0).mean()*100:.1f} %",      CF),
    ("Penaliz. SOC media",   f"{df_n['penalizacion_soc [€]'].mean():.2f} €", CP),
]
for i, (label, valor, color) in enumerate(metricas):
    y = 1 - (i + 0.5) / len(metricas)
    ax.text(0.05, y, label, transform=ax.transAxes,
            fontsize=9, color="#555", va="center")
    ax.text(0.72, y, valor, transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=color, va="center")
ax.set_title("④ Métricas Clave del Día", **TKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 2 — SOC Y DESGLOSE DE INGRESOS
# =============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig2.canvas.manager.set_window_title(f"Simulación — SOC e Ingresos | {fecha_str}")
fig2.suptitle(f"SOC Y DESGLOSE DE INGRESOS — {fecha_str}",
              fontsize=11, fontweight="bold", y=0.97)

# ⑤ SOC previsto + distribución SOC final
ax = axes2[0, 0]
x96 = np.arange(96)

if df_det is not None and "SOC [MWh]" in df_det.columns:
    soc_det = df_det["SOC [MWh]"].values
    ax.plot(x96, soc_det, color=CACC, lw=2.5, zorder=5,
            label="SOC previsto (modelo)")

soc_finals_n = df_n["soc_final [MWh]"].values
soc_finals_e = df_e["soc_final [MWh]"].values

ax2b = ax.twinx()
ax2b.boxplot([soc_finals_n, soc_finals_e],
             positions=[88, 93], widths=3,
             patch_artist=True,
             boxprops=dict(facecolor=CCH, alpha=0.5),
             medianprops=dict(color="white", lw=2))
ax2b.set_ylabel("SOC final (MWh)", fontsize=7, color=CGRIS)
ax2b.tick_params(labelsize=7)

ax.axhline(SOC_INIT, color=CACC, lw=1.2, ls=":",
           label=f"SOC objetivo ({SOC_INIT} MWh)")
ax.axhline(SOC_MIN,  color=CP,   lw=1.2, ls=":",
           label=f"SOC mín ({SOC_MIN:.2f} MWh)")
ax.axhline(E_MAX,    color=CB,   lw=1.2, ls=":",
           label=f"E_max ({E_MAX} MWh)")
ax.set_ylim(-0.05, E_MAX * 1.2)
ax.set_title("⑤ SOC Previsto + Distribución SOC Final", **TKW)
_ax(ax, "MWh"); _xt(ax)
ax.legend(fontsize=7, framealpha=0.8, edgecolor="none")

# ⑥ Desvío SOC final
ax = axes2[0, 1]
desv_n  = df_n["soc_final_desv [MWh]"].values
desv_e  = df_e["soc_final_desv [MWh]"].values
rango_d = (min(desv_n.min(), desv_e.min()) - 0.05,
           max(desv_n.max(), desv_e.max()) + 0.05)
ax.hist(desv_n, bins=30, range=rango_d, color=CCH, alpha=0.6,
        density=True, label="Normales")
ax.hist(desv_e, bins=30, range=rango_d, color=CB,  alpha=0.6,
        density=True, label="Extremos")
ax.axvline(0, color=CACC, lw=2, ls="--", label="Sin desvío")
ax.axvline(desv_n.mean(), color=CCH, lw=1.5, ls=":",
           label=f"Media N: {desv_n.mean():+.3f} MWh")
ax.axvline(desv_e.mean(), color=CB,  lw=1.5, ls=":",
           label=f"Media E: {desv_e.mean():+.3f} MWh")
ax.set_title("⑥ Distribución Desvío SOC Final", **TKW)
_ax(ax, "densidad")
ax.set_xlabel("Desvío SOC final (MWh)", fontsize=8, color="#555")
ax.legend(fontsize=7, framealpha=0.8, edgecolor="none")

# ⑦ Desglose de ingresos — previsto vs media simulación
ax = axes2[1, 0]
categorias = ["Spot", "Disponib.", "Activación", "Degradación", "Penal. SOC"]

# Previsto — desde el schedule (columnas de simulación tienen beneficio_previsto
# pero no el desglose, así que lo calculamos desde df_det si existe)
if df_det is not None:
    PI_DISP, PI_ACT_UP, PI_ACT_DOWN, C_DEG_V = 10.0, 114.30, 50.73, 2.0
    ing_spot_prev = (df_det["p_sell [€/MWh]"] * df_det["x_sell [MWh]"]
                   - df_det["p_buy_eff [€/MWh]"] * df_det["x_buy [MWh]"]).sum()
    ing_disp_prev = (PI_DISP * (df_det["r_up [MWh]"] + df_det["r_down [MWh]"])).sum()
    ing_act_prev  = ((df_det["p_act_up [€/MWh]"]   - df_det["p_sell [€/MWh]"])   * df_det["a_up [MWh]"]
                   + (df_det["p_act_down [€/MWh]"] - df_det["p_buy_eff [€/MWh]"]) * df_det["a_down [MWh]"]).sum()
    deg_prev      = (C_DEG_V * (df_det["x_ch [MWh]"] + df_det["x_dis [MWh]"]
                              + df_det["a_up [MWh]"] + df_det["a_down [MWh]"])).sum()
    vals_prev = [ing_spot_prev, ing_disp_prev, ing_act_prev, -deg_prev, 0]
    tiene_prev = True
else:
    tiene_prev = False

vals_sim = [
    df_n["ingreso_spot_real [€]"].mean(),
    df_n["ingreso_disp_real [€]"].mean(),
    df_n["ingreso_act_real [€]"].mean(),
    -df_n["coste_deg_real [€]"].mean(),
    -df_n["penalizacion_soc [€]"].mean(),
]

x_cat = np.arange(len(categorias))

if tiene_prev:
    ancho = 0.35
    colores_prev = [CS if v >= 0 else CB for v in vals_prev]
    colores_sim  = [CCH if v >= 0 else CDI for v in vals_sim]
    ax.bar(x_cat - ancho/2, vals_prev, width=ancho, color=colores_prev,
           alpha=0.8, label="Previsto (schedule)")
    ax.bar(x_cat + ancho/2, vals_sim,  width=ancho, color=colores_sim,
           alpha=0.8, label="Sim. media normal")
else:
    colores_sim = [CCH if v >= 0 else CDI for v in vals_sim]
    ax.bar(x_cat, vals_sim, width=0.6, color=colores_sim,
           alpha=0.8, label="Sim. media normal")

ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_xticks(x_cat)
ax.set_xticklabels(categorias, fontsize=8)
ax.set_title("⑦ Desglose de Ingresos — Previsto vs Real", **TKW)
_ax(ax, "€"); ax.legend(**GKW)

# ⑧ Penalización SOC y energía recortada por escenario
ax = axes2[1, 1]
pen_vals = df_n["penalizacion_soc [€]"].values
rec_vals = df_n["energia_recortada [MWh]"].values
sim_ids  = np.arange(len(df_n))

ax2r = ax.twinx()
ax.scatter(sim_ids,  pen_vals, color=CP,  alpha=0.4, s=8, label="Penaliz. SOC (€)")
ax2r.scatter(sim_ids, rec_vals, color=CRU, alpha=0.4, s=8, label="Energía recortada (MWh)")
ax.axhline(pen_vals.mean(),  color=CP,  lw=1.5, ls="--",
           label=f"Media pen.: {pen_vals.mean():.2f}€")
ax2r.axhline(rec_vals.mean(), color=CRU, lw=1.5, ls="--",
             label=f"Media rec.: {rec_vals.mean():.3f} MWh")

ax.set_ylabel("Penalización SOC (€)", fontsize=8, color=CP)
ax2r.set_ylabel("Energía recortada (MWh)", fontsize=8, color=CRU)
ax.tick_params(axis="y", colors=CP, labelsize=7)
ax2r.tick_params(axis="y", colors=CRU, labelsize=7)
ax.set_xlabel("Escenario (#)", fontsize=8, color="#555")

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2,
          fontsize=7, framealpha=0.8, edgecolor="none")
ax.set_title("⑧ Penalización SOC y Energía Recortada (Normales)", **TKW)
_ax(ax)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# MOSTRAR
# =============================================================================

print(f"=== Dashboards generados — {fecha_str} ===")
print(f"  Dashboard 1: Distribución de beneficios")
print(f"  Dashboard 2: SOC, desglose de ingresos y penalizaciones")
plt.show()