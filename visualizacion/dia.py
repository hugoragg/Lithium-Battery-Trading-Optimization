"""
Visualización de Resultados — Día Suelto
Autor: Hugo Raggini Paternain

Lee el CSV de 'resultados/dias_sueltos/' generado por optimizacion.bateria
y genera los dashboards de operación física y resultados económicos.

Uso:
    python -m visualizacion.dia              # pregunta fecha interactivamente
    python -m visualizacion.dia 2026-03-13   # fecha directa
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

CARPETA = ROOT / "resultados" / "dias_sueltos"

# --- Determinar fecha ---
if len(sys.argv) == 2:
    fecha_str = sys.argv[1].strip()
    # Normalizar formato si viene como DD/MM/YYYY
    if "/" in fecha_str:
        fecha_str = pd.to_datetime(fecha_str, dayfirst=True).strftime("%Y-%m-%d")
else:
    print("\n=== Visualización — Día Suelto ===")
    while True:
        entrada = input("Fecha a visualizar (DD/MM/YYYY o YYYY-MM-DD): ").strip()
        try:
            if "/" in entrada:
                fecha_str = pd.to_datetime(entrada, dayfirst=True).strftime("%Y-%m-%d")
            else:
                fecha_str = pd.to_datetime(entrada).strftime("%Y-%m-%d")
            break
        except Exception:
            print("  [!] Formato no reconocido. Prueba: 13/03/2026 o 2026-03-13")

# --- Buscar CSV ---
csv_path = CARPETA / f"resultado_{fecha_str}.csv"

if not csv_path.exists():
    # Listar fechas disponibles como ayuda
    disponibles = sorted([f.stem.replace("resultado_", "") for f in CARPETA.glob("resultado_*.csv")])
    print(f"\n[!] No se encuentra '{csv_path}'.")
    if disponibles:
        print(f"    Fechas disponibles en '{CARPETA}':")
        for d in disponibles:
            print(f"      · {d}")
    else:
        print(f"    No hay resultados en '{CARPETA}'. Ejecuta primero optimizacion.bateria.")
    sys.exit()

df = pd.read_csv(csv_path)
print(f"\n  Cargando resultados de: {csv_path}")

# --- Validación de columnas ---
cols_necesarias = [
    "pi_disp_up [€/MWh]", "pi_disp_down [€/MWh]",
    "pi_act_up [€/MWh]", "pi_act_down [€/MWh]",
    "E_max [MWh]", "SOC_min [MWh]",
]
for col in cols_necesarias:
    if col not in df.columns:
        print(f"\n[!] Error: La columna '{col}' no existe en el CSV.")
        sys.exit()

# =============================================================================
# 2. PROCESAMIENTO DINÁMICO
# =============================================================================

E_MAX_CSV   = df["E_max [MWh]"].iloc[0]
SOC_MIN_CSV = df["SOC_min [MWh]"].iloc[0]

cols_fisicas = ["x_sell [MWh]", "x_buy [MWh]", "x_ch [MWh]", "x_dis [MWh]",
                "r_up [MWh]", "r_down [MWh]", "a_up [MWh]", "a_down [MWh]"]
df[cols_fisicas] = df[cols_fisicas].clip(lower=0)

ing_venta  = df["p [€/MWh]"]     * df["x_sell [MWh]"]
cst_compra = df["p_eff [€/MWh]"] * df["x_buy [MWh]"]
ing_disp   = (df["pi_disp_up [€/MWh]"]   * df["r_up [MWh]"]
            + df["pi_disp_down [€/MWh]"] * df["r_down [MWh]"])
ing_act    = ((df["pi_act_up [€/MWh]"]   - df["p [€/MWh]"])     * df["a_up [MWh]"]) \
           + ((df["pi_act_down [€/MWh]"] - df["p_eff [€/MWh]"]) * df["a_down [MWh]"])

df["ben_neto"] = ing_venta - cst_compra + ing_disp + ing_act
df["ben_acum"] = df["ben_neto"].cumsum()

TV, TC = ing_venta.sum(), cst_compra.sum()
TR, TA = ing_disp.sum(), ing_act.sum()
TB     = df["ben_neto"].sum()

x      = np.arange(len(df))
xt_pos = x[::4]
xt_lbl = df["hora"][::4]

# =============================================================================
# 3. ESTILO Y HELPERS
# =============================================================================

CF, CS, CB, CCH, CDI, CSOC, CP = "#F5A623", "#27AE60", "#E74C3C", "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22"
CRU, CRD, CAU, CAD, CACC, BG, PBG = "#E74C3C", "#3498DB", "#C0392B", "#2980B9", "#2C3E50", "#F5F6FA", "#FFFFFF"

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

def _xt(ax):
    ax.set_xticks(xt_pos)
    ax.set_xticklabels(xt_lbl, rotation=45, ha="right")
    ax.set_xlim(-1, 96)

# =============================================================================
# DASHBOARD 1 — OPERACIÓN FÍSICA
# =============================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig1.canvas.manager.set_window_title(f"Dashboard Físico — {fecha_str}")
fig1.suptitle(f"OPERACIÓN FÍSICA — BATERÍA {E_MAX_CSV} MWh  |  {fecha_str}",
              fontsize=12, fontweight="bold", y=0.96)

ax = axes1[0, 0]
ax.plot(x, df["p [€/MWh]"], color=CP, lw=1.5)
ax.set_title("Precio Mercado Diario (Venta)", **TKW)
_ax(ax, "EUR/MWh"); _xt(ax)

ax = axes1[0, 1]
ax.bar(x, df["x_buy [MWh]"],  color=CCH, alpha=0.7, label="Compra Red")
ax.plot(x, df["x_sell [MWh]"], color=CS,  lw=1.5,   label="Venta Red")
ax.set_title("Flujos de Energía con la Red", **TKW)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

ax = axes1[1, 0]
ax.fill_between(x, df["SOC [MWh]"], alpha=0.15, color=CSOC)
ax.plot(x, df["SOC [MWh]"], color=CSOC, lw=2, label="SOC")
ax.axhline(E_MAX_CSV,   color=CB, ls=":", lw=1.2, label=f"E_max ({E_MAX_CSV} MWh)")
ax.axhline(SOC_MIN_CSV, color=CP, ls=":", lw=1.2, label=f"SOC_min ({SOC_MIN_CSV:.2f} MWh)")
ax.set_title("Estado de Carga (SOC)", **TKW)
ax.set_ylim(-0.05, E_MAX_CSV * 1.2)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

ax = axes1[1, 1]
ax.bar(x,  df["x_ch [MWh]"],  color=CCH, alpha=0.7, label="Carga")
ax.bar(x, -df["x_dis [MWh]"], color=CDI, alpha=0.7, label="Descarga")
ax.set_title("Ciclos de Carga/Descarga", **TKW)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 2 — RESULTADOS ECONÓMICOS
# =============================================================================

fig2 = plt.figure(figsize=(14, 9), facecolor=BG)
fig2.canvas.manager.set_window_title(f"Dashboard Económico — {fecha_str}")

gs = gridspec.GridSpec(3, 2, figure=fig2,
                       height_ratios=[0.25, 1, 1],
                       hspace=0.5, wspace=0.25,
                       top=0.92, bottom=0.12, left=0.08, right=0.92)

# KPIs
ax_kpi = fig2.add_subplot(gs[0, :])
ax_kpi.axis("off")
kpis = [
    ("BENEFICIO NETO",  f"{TB:+.2f} EUR",   "#27AE60"),
    ("Ingreso Venta",   f"{TV:.2f} EUR",     CS),
    ("Ingreso Reserva", f"{TR+TA:.2f} EUR",  "#7F8C8D"),
    ("Coste Compra",    f"-{TC:.2f} EUR",    CB),
]
for i, (lbl, val, col) in enumerate(kpis):
    cx = (i + 0.5) / len(kpis)
    ax_kpi.text(cx, 0.6, lbl, ha="center", fontsize=8,  color="#666", transform=ax_kpi.transAxes)
    ax_kpi.text(cx, 0.1, val, ha="center", fontsize=11, fontweight="bold", color=col, transform=ax_kpi.transAxes)

ax_b = fig2.add_subplot(gs[1, 0])
ax_b.bar(x,  df["r_up [MWh]"],   color=CRU, alpha=0.6, label="Reserva Up")
ax_b.bar(x, -df["r_down [MWh]"], color=CRD, alpha=0.6, label="Reserva Down")
ax_b.set_title("Banda de Reserva Ofertada", **TKW)
_ax(ax_b, "MWh"); _xt(ax_b); ax_b.legend(**GKW)

ax_c = fig2.add_subplot(gs[1, 1])
ax_c.bar(x,  df["a_up [MWh]"],   color=CAU, alpha=0.7, label="Act. Up")
ax_c.bar(x, -df["a_down [MWh]"], color=CAD, alpha=0.7, label="Act. Down")
ax_c.set_title("Energía Real Activada", **TKW)
_ax(ax_c, "MWh"); _xt(ax_c); ax_c.legend(**GKW)

ax_d = fig2.add_subplot(gs[2, :])
ax_d.bar(x, ing_venta,            color=CS,        alpha=0.5, label="Venta")
ax_d.bar(x, ing_disp + ing_act,   bottom=ing_venta, color="#95A5A6", alpha=0.4, label="Serv. Ajuste")
ax_d.bar(x, -cst_compra,          color=CB,        alpha=0.5, label="Compra")

ax_d2 = ax_d.twinx()
ax_d2.plot(x, df["ben_acum"], color=CACC, lw=2, label="Acumulado")
ax_d2.set_ylabel("EUR Acumulados", color=CACC, fontsize=8)

ax_d.set_title("Desglose Financiero e Ingresos Acumulados", **TKW)
_ax(ax_d, "EUR/15min"); _xt(ax_d)
ax_d.legend(loc="upper left", **GKW)

# =============================================================================
# MOSTRAR
# =============================================================================

print(f"  Beneficio total: {TB:+.2f} €")
print(f"  Generando dashboards...\n")
plt.show()