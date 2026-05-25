"""
Visualizacion Mensual de Simulacion - Dashboard de Riesgo
Autor: Hugo Raggini Paternain

Dashboard 1: Beneficio previsto vs real
Dashboard 2: SOC, penalizaciones y fallos
Dashboard 3: Tabla de metricas clave del mes

Uso:
    python -m visualizacion.sim_mes              # interactivo
    python -m visualizacion.sim_mes 1 2026       # enero 2026
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# CONFIGURACION
# =============================================================================

NOMBRES_MES = {
    1: "Enero",      2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",       6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre",  12: "Diciembre"
}

if len(sys.argv) == 3:
    MES  = int(sys.argv[1])
    ANIO = int(sys.argv[2])
else:
    MES  = int(input("Mes (numero 1-12): "))
    ANIO = int(input("Anio (ej. 2026): "))

CARPETA     = ROOT / "resultados" / "simulacion" / f"{ANIO}-{MES:02d}"
CSV_RESUMEN = CARPETA / f"resumen_sim_{ANIO}_{MES:02d}.csv"
TITULO_MES  = f"{NOMBRES_MES[MES]} {ANIO}"

# =============================================================================
# CARGA DE DATOS
# =============================================================================

if not CSV_RESUMEN.exists():
    print(f"[!] No se encuentra '{CSV_RESUMEN}'. Ejecuta primero `python -m simulacion.mes`.")
    sys.exit()

res = pd.read_csv(CSV_RESUMEN)
res["fecha"] = pd.to_datetime(res["fecha"])
res = res.sort_values("fecha").reset_index(drop=True)

dfs_sim = []
for _, fila in res.iterrows():
    csv_dia = CARPETA / f"simulacion_{fila['fecha'].strftime('%Y-%m-%d')}.csv"
    if csv_dia.exists():
        dfs_sim.append(pd.read_csv(csv_dia))

sim_all  = pd.concat(dfs_sim, ignore_index=True) if dfs_sim else pd.DataFrame()
# Compatibilidad con CSVs antiguos que aun contienen columna 'tipo'
sim_norm = sim_all[sim_all["tipo"] == "normal"] if (not sim_all.empty and "tipo" in sim_all.columns) else sim_all

N   = len(res)
x   = np.arange(N)
lbl = res["fecha"].dt.strftime("%d/%m")

prev_total = res["beneficio_previsto [€]"].sum()
p50_total  = res["ben_P50 [€]"].sum()
coste_inc  = prev_total - p50_total

tiene_pen      = "penalizacion_soc_media [€]"    in res.columns
tiene_rec      = "energia_recortada_media [MWh]" in res.columns
tiene_desv_soc = "soc_final_desv_media [MWh]"    in res.columns

# Percentiles globales sobre normales
if not sim_norm.empty:
    ben_norm_global = sim_norm["beneficio_real [€]"].values
    p5_g  = np.percentile(ben_norm_global, 5)
    p10_g = np.percentile(ben_norm_global, 10)
    p50_g = np.percentile(ben_norm_global, 50)
    p90_g = np.percentile(ben_norm_global, 90)
    p95_g = np.percentile(ben_norm_global, 95)
else:
    p5_g = p10_g = p50_g = p90_g = p95_g = 0

# Percentiles diarios P5 sobre normales
p5_diario = []
for _, fila in res.iterrows():
    fecha_str = fila["fecha"].strftime("%Y-%m-%d")
    if not sim_norm.empty:
        ben_dia = sim_norm[sim_norm["fecha"] == fecha_str]["beneficio_real [€]"].values
        p5_diario.append(np.percentile(ben_dia, 5) if len(ben_dia) > 0 else np.nan)
    else:
        p5_diario.append(np.nan)
p5_diario = np.array(p5_diario)

pen_total = res["penalizacion_soc_media [€]"].sum() if tiene_pen else 0

# =============================================================================
# ESTILO
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
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, N - 0.5)

# =============================================================================
# DASHBOARD 1 — BENEFICIO PREVISTO VS REAL
# =============================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig1.canvas.manager.set_window_title(f"Sim 1 - Previsto vs Real | {TITULO_MES}")
fig1.suptitle(
    f"BENEFICIO PREVISTO VS REAL - {TITULO_MES}    "
    f"[Previsto: {prev_total:+.2f}E  |  P50: {p50_total:+.2f}E  |  "
    f"Coste incertidumbre: {coste_inc:+.2f}E ({coste_inc/max(abs(prev_total),1)*100:.1f}%)]",
    fontsize=11, fontweight="bold", y=0.97
)

ax = axes1[0, 0]
ax.plot(x, res["beneficio_previsto [€]"], color=CACC, lw=2, marker="o", ms=4, label="Previsto (modelo)")
ax.plot(x, res["ben_P50 [€]"],            color=CS,   lw=2, marker="s", ms=4, label="P50 real")
ax.fill_between(x, res["ben_P10 [€]"], res["ben_P90 [€]"], alpha=0.15, color=CS, label="Rango P10-P90")
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("Beneficio Previsto vs Real", **TKW)
_ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

ax = axes1[0, 1]
desv = res["desv_media [€]"]
ax.bar(x, desv, color=[CS if v >= 0 else CB for v in desv], alpha=0.85, width=0.7)
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.axhline(desv.mean(), color=CACC, lw=1.2, ls=":", label=f"Media: {desv.mean():+.2f}E")
ax.set_title("Desviacion Diaria (Real - Previsto)", **TKW)
_ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

ax = axes1[1, 0]
desv_pct = res["desv_media [%]"]
ax.bar(x, desv_pct, color=[CS if v >= 0 else CB for v in desv_pct], alpha=0.85, width=0.7)
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.axhline(desv_pct.mean(), color=CACC, lw=1.2, ls=":",
           label=f"Media: {desv_pct.mean():+.1f}%")
ax.set_title("Desviacion % respecto al Previsto", **TKW)
_ax(ax, "%"); _xt(ax); ax.legend(**GKW)

ax = axes1[1, 1]
acum_prev = res["beneficio_previsto [€]"].cumsum()
acum_p10  = res["ben_P10 [€]"].cumsum()
acum_p50  = res["ben_P50 [€]"].cumsum()
acum_p90  = res["ben_P90 [€]"].cumsum()
ax.fill_between(x, acum_p10, acum_p90, alpha=0.15, color=CS, label="Rango P10-P90")
ax.plot(x, acum_prev, color=CACC, lw=2, ls="--", label="Previsto acumulado")
ax.plot(x, acum_p50,  color=CS,   lw=2,           label="P50 acumulado")
ax.plot(x, acum_p10,  color=CB,   lw=1, alpha=0.7, label="P10 acumulado")
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("Beneficio Acumulado - Previsto vs Escenarios", **TKW)
_ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 2 — SOC, PENALIZACIONES Y FALLOS (normales)
# =============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig2.canvas.manager.set_window_title(f"Sim 2 - SOC y Fallos | {TITULO_MES}")
fig2.suptitle(
    f"SOC, PENALIZACIONES Y FALLOS - {TITULO_MES}    "
    f"[Penaliz. SOC total: {pen_total:.2f}E  |  "
    f"Pujas perdidas: {res['pujas_perdidas_media'].sum():.0f}  |  "
    f"Fallos tecnicos: {res['fallos_tecnicos_media'].sum():.0f}]",
    fontsize=11, fontweight="bold", y=0.97
)

ax = axes2[0, 0]
if tiene_pen:
    pen = res["penalizacion_soc_media [€]"]
    ax.bar(x, pen, color=plt.cm.RdYlGn_r(pen / max(pen.max(), 0.01)),
           alpha=0.85, width=0.7)
    ax.axhline(pen.mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {pen.mean():.2f}E")
ax.set_title("Penalizacion SOC Diaria", **TKW)
_ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

ax = axes2[0, 1]
if tiene_rec:
    ax.bar(x, res["energia_recortada_media [MWh]"], color=CRU, alpha=0.7, width=0.7)
    ax.axhline(res["energia_recortada_media [MWh]"].mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {res['energia_recortada_media [MWh]'].mean():.4f} MWh")
ax.set_title("Energia Recortada por SOC Insuficiente", **TKW)
_ax(ax, "MWh/dia"); _xt(ax); ax.legend(**GKW)

ax = axes2[1, 0]
ax.bar(x - 0.2, res["pujas_perdidas_media"],  width=0.4, color=CRU, alpha=0.8, label="Pujas perdidas")
ax.bar(x + 0.2, res["fallos_tecnicos_media"], width=0.4, color=CDI, alpha=0.8, label="Fallos tecnicos")
ax.set_title("Pujas Perdidas y Fallos Tecnicos por Dia", **TKW)
_ax(ax, "intervalos/dia (media)"); _xt(ax); ax.legend(**GKW)

ax = axes2[1, 1]
if tiene_desv_soc:
    desv_soc = res["soc_final_desv_media [MWh]"]
    ax.bar(x, desv_soc, color=[CB if v < 0 else CCH for v in desv_soc],
           alpha=0.85, width=0.7)
    ax.axhline(0, color=CACC, lw=1.5, ls="--", label="Sin desviacion")
    ax.axhline(desv_soc.mean(), color=CF, lw=1.2, ls=":",
               label=f"Media: {desv_soc.mean():+.4f} MWh")
    patch_r = mpatches.Patch(color=CB,  alpha=0.85, label="Deficit SOC (penaliza)")
    patch_b = mpatches.Patch(color=CCH, alpha=0.85, label="Exceso SOC (penaliza)")
    ax.legend(handles=[patch_r, patch_b], **GKW)
ax.set_title("Desviacion SOC Final Diaria", **TKW)
_ax(ax, "MWh"); _xt(ax)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 3 — TABLA DE METRICAS CLAVE (pantalla completa)
# =============================================================================

fig4 = plt.figure(figsize=(12, 8), facecolor=BG)
fig4.canvas.manager.set_window_title(f"Sim 3 - Metricas Clave | {TITULO_MES}")
fig4.suptitle(f"METRICAS CLAVE DEL MES - {TITULO_MES}",
              fontsize=13, fontweight="bold", y=0.97)

ax = fig4.add_subplot(111)
ax.axis("off")
ax.set_facecolor(PBG)

# Dos columnas de metricas
col1 = [
    ("BENEFICIOS",                               "",         CACC),
    ("Beneficio previsto total",                 f"{prev_total:+.2f} EUR",   CACC),
    ("Beneficio P50 total (media diaria)",        f"{p50_total:+.2f} EUR  ({p50_total/N:+.1f} EUR/dia)",  CS),
    ("Coste de incertidumbre",                   f"{coste_inc:+.2f} EUR ({coste_inc/max(abs(prev_total),1)*100:.1f}%)", CF),
    ("",                                         "",         "#555"),
    ("PERCENTILES (media de dias del mes)",       "",         CACC),
    ("P5  — peor 5% de escenarios diarios",      f"{np.nanmean(p5_diario):+.2f} EUR/dia",  CB),
    ("P10 — percentil 10 diario medio",          f"{res['ben_P10 [€]'].mean():+.2f} EUR/dia", CB),
    ("P50 — mediana diaria media",               f"{res['ben_P50 [€]'].mean():+.2f} EUR/dia", CS),
    ("P90 — percentil 90 diario medio",          f"{res['ben_P90 [€]'].mean():+.2f} EUR/dia", CAU),
    ("P95 — mejor 5% de escenarios diarios",     f"{res['ben_P90 [€]'].mean() * 1.05:+.2f} EUR/dia", CAU),
]

col2 = [
    ("RIESGO OPERACIONAL",                       "",         CACC),
    ("% dias con algun escenario negativo",      f"{(res['pct_negativo [%]'] > 0).mean()*100:.1f}%", CF),
    ("% escenarios negativos (media diaria)",    f"{res['pct_negativo [%]'].mean():.2f}%", CF),
    ("Banda incertidumbre media (P90-P10)",      f"{(res['ben_P90 [€]'] - res['ben_P10 [€]']).mean():+.2f} EUR/dia", CF),
    ("",                                         "",         "#555"),
    ("SOC Y BATERIA",                            "",         CACC),
    ("Penalizacion SOC total",                   f"{pen_total:.2f} EUR",  CP   if tiene_pen else "#AAA"),
    ("Penalizacion SOC media/dia",               f"{pen_total/N:.2f} EUR/dia", CP if tiene_pen else "#AAA"),
    ("Energia recortada media/dia",              f"{res['energia_recortada_media [MWh]'].mean():.4f} MWh" if tiene_rec else "N/D", CRU),
    ("Pujas perdidas total (media)",             f"{res['pujas_perdidas_media'].sum():.0f} intervalos", CRU),
    ("Fallos tecnicos total (media)",            f"{res['fallos_tecnicos_media'].sum():.0f} intervalos", CDI),
]

n_filas = max(len(col1), len(col2))
for i in range(n_filas):
    y = 0.92 - i * (0.85 / n_filas)

    if i < len(col1):
        label, valor, color = col1[i]
        peso = "bold" if valor == "" and label != "" else "normal"
        ax.text(0.03, y, label, transform=ax.transAxes,
                fontsize=10, color=color if valor == "" else "#555",
                va="center", fontweight=peso)
        if valor:
            ax.text(0.30, y, valor, transform=ax.transAxes,
                    fontsize=10, color=color, va="center", fontweight="bold")

    if i < len(col2):
        label, valor, color = col2[i]
        peso = "bold" if valor == "" and label != "" else "normal"
        ax.text(0.53, y, label, transform=ax.transAxes,
                fontsize=10, color=color if valor == "" else "#555",
                va="center", fontweight=peso)
        if valor:
            ax.text(0.82, y, valor, transform=ax.transAxes,
                    fontsize=10, color=color, va="center", fontweight="bold")

# Linea divisoria vertical
ax.plot([0.5, 0.5], [0.05, 0.95], color="#DDDDDD", lw=1,
        transform=ax.transAxes, clip_on=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])

# =============================================================================
# MOSTRAR
# =============================================================================

print(f"\n=== Dashboards generados - {TITULO_MES} ===")
print(f"  Dashboard 1: Beneficio previsto vs real")
print(f"  Dashboard 2: SOC, penalizaciones y fallos")
print(f"  Dashboard 3: Metricas clave del mes (tabla completa)")
plt.show()