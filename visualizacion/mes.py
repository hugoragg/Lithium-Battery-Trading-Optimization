"""
Visualización Mensual de Resultados — Dashboard estilo diario
Autor: Hugo Raggini Paternain

Uso:
    python -m visualizacion.mes              # interactivo
    python -m visualizacion.mes 1 2026       # enero 2026
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# CONFIGURACIÓN
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
    MES  = int(input("Mes (número 1-12): "))
    ANIO = int(input("Año (ej. 2026): "))

CARPETA     = ROOT / "resultados" / "optimizacion" / f"{ANIO}-{MES:02d}"
CSV_RESUMEN = CARPETA / f"resumen_{ANIO}_{MES:02d}.csv"
TITULO_MES  = f"{NOMBRES_MES[MES]} {ANIO}"

# =============================================================================
# CARGA DE DATOS
# =============================================================================

if not CSV_RESUMEN.exists():
    print(f"[!] No se encuentra '{CSV_RESUMEN}'. Ejecuta primero `python -m optimizacion.mes`.")
    sys.exit()

resumen = pd.read_csv(CSV_RESUMEN)
resumen["fecha"] = pd.to_datetime(resumen["fecha"])
resumen = resumen.dropna(subset=["beneficio [€]"]).reset_index(drop=True)

# Cargar CSVs diarios y concatenar
dfs_diarios = []
for _, fila in resumen.iterrows():
    fecha_str = fila["fecha"].strftime("%Y-%m-%d")
    csv_dia   = CARPETA / f"resultado_{fecha_str}.csv"
    if csv_dia.exists():
        df = pd.read_csv(csv_dia)
        df["fecha"] = fecha_str
        dfs_diarios.append(df)

detalle = pd.concat(dfs_diarios, ignore_index=True) if dfs_diarios else pd.DataFrame()

# Columnas de intervalo y hora para agregaciones
if not detalle.empty:
    detalle["intervalo"] = detalle.groupby("fecha").cumcount()
    detalle["hora"]      = detalle["intervalo"] // 4

N = len(resumen)
x = np.arange(N)
lbl = resumen["fecha"].dt.strftime("%d/%m")

E_MAX_VAL   = detalle["capacidad_nominal [MWh]"].iloc[0] if not detalle.empty and "capacidad_nominal [MWh]" in detalle.columns else 2.0
SOC_MIN_VAL = detalle["soc_min_limit [MWh]"].iloc[0]     if not detalle.empty and "soc_min_limit [MWh]"    in detalle.columns else 0.14

# =============================================================================
# ESTILO — idéntico al visualizador diario
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

def _xt_horas(ax):
    pos = np.arange(24)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{h:02d}h" for h in pos], rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, 23.5)

# =============================================================================
# DASHBOARD 1 — BENEFICIOS Y OPERACIÓN ECONÓMICA
# =============================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig1.canvas.manager.set_window_title(f"Dashboard Mensual 1 — Beneficios | {TITULO_MES}")

beneficio_total = resumen["beneficio [€]"].sum()
beneficio_medio = resumen["beneficio [€]"].mean()
dias_pos = (resumen["beneficio [€]"] > 0).sum()

fig1.suptitle(
    f"BENEFICIOS Y OPERACIÓN ECONÓMICA — {TITULO_MES}    "
    f"[Total: {beneficio_total:+.2f} €  |  Media: {beneficio_medio:+.2f} €/día  |  Días positivos: {dias_pos}/{N}]",
    fontsize=11, fontweight="bold", y=0.97
)

# ① Beneficio neto diario
ax = axes1[0, 0]
colores_b = [CS if v >= 0 else CB for v in resumen["beneficio [€]"]]
ax.bar(x, resumen["beneficio [€]"], color=colores_b, alpha=0.85, width=0.7)
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.axhline(beneficio_medio, color=CACC, lw=1.2, ls=":", label=f"Media: {beneficio_medio:+.2f} €")
ax.set_title("① Beneficio Neto Diario", **TKW)
_ax(ax, "€"); _xt(ax); ax.legend(**GKW)

# ② Beneficio acumulado
ax = axes1[0, 1]
acum = resumen["beneficio [€]"].cumsum()
color_acum = CS if acum.iloc[-1] >= 0 else CB
ax.fill_between(x, acum, alpha=0.15, color=color_acum)
ax.plot(x, acum, color=color_acum, lw=2)
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("② Beneficio Acumulado del Mes", **TKW)
_ax(ax, "€"); _xt(ax)

# ③ Desglose de ingresos diario (calculado desde CSVs cuartohorarios)
ax = axes1[1, 0]
if not detalle.empty and all(c in detalle.columns for c in [
        "p_sell [€/MWh]", "p_buy_eff [€/MWh]",
        "x_sell [MWh]", "x_buy [MWh]",
        "r_up [MWh]", "r_down [MWh]",
        "a_up [MWh]", "a_down [MWh]"]):

    from parametros import PI_DISP_UP, PI_DISP_DOWN, PI_ACT_UP, PI_ACT_DOWN, C_DEG

    # Ingresos por intervalo
    detalle["ing_arbitraje"]  = (detalle["p_sell [€/MWh]"]    * detalle["x_sell [MWh]"]
                                - detalle["p_buy_eff [€/MWh]"] * detalle["x_buy [MWh]"])
    detalle["ing_disp"]       = PI_DISP_UP * detalle["r_up [MWh]"] + PI_DISP_DOWN * detalle["r_down [MWh]"]
    detalle["ing_activacion"] = ((PI_ACT_UP   - detalle["p_sell [€/MWh]"]) * detalle["a_up [MWh]"]
                                + (PI_ACT_DOWN - detalle["p_buy_eff [€/MWh]"]) * detalle["a_down [MWh]"])
    detalle["cst_deg"]        = C_DEG * (detalle["x_ch [MWh]"] + detalle["x_dis [MWh]"]
                                        + detalle["a_up [MWh]"] + detalle["a_down [MWh]"])

    # Agregar por día — mismo orden que resumen
    fechas_ord = resumen["fecha"].dt.strftime("%Y-%m-%d").values
    ing_arb_d  = detalle.groupby("fecha")["ing_arbitraje"].sum().reindex(fechas_ord).values
    ing_dis_d  = detalle.groupby("fecha")["ing_disp"].sum().reindex(fechas_ord).values
    ing_act_d  = detalle.groupby("fecha")["ing_activacion"].sum().reindex(fechas_ord).values
    cst_deg_d  = detalle.groupby("fecha")["cst_deg"].sum().reindex(fechas_ord).values

    # Barras apiladas — positivos arriba, degradación abajo
    # DESPUÉS
    ax.bar(x, ing_arb_d, color=CS, alpha=0.85, width=0.7, label="Arbitraje spot")
    ax.bar(x, ing_dis_d, color=CCH, alpha=0.85, width=0.7,
        bottom=np.maximum(ing_arb_d, 0), label="Disponibilidad aFRR")
    ax.bar(x, ing_act_d, color=CAU, alpha=0.85, width=0.7,
        bottom=np.maximum(ing_arb_d, 0) + ing_dis_d, label="Activación aFRR")
    ax.bar(x, -cst_deg_d, color=CB, alpha=0.70, width=0.7,
        bottom=np.minimum(ing_arb_d, 0), label="Coste degradación")

ax.set_title("③ Desglose de Ingresos Diario", **TKW)
_ax(ax, "€"); _xt(ax); ax.legend(**GKW)


# ④ Beneficio por día de la semana
ax = axes1[1, 1]
resumen["dia_semana"] = resumen["fecha"].dt.dayofweek
orden = [0, 1, 2, 3, 4, 5, 6]
nombres_dias = ["Lunes", "Martes", "Miérc.", "Jueves", "Viernes", "Sábado", "Domingo"]
media_semana = resumen.groupby("dia_semana")["beneficio [€]"].mean().reindex(orden)
colores_sem  = [CS if v >= 0 else CB for v in media_semana.values]
ax.bar(np.arange(7), media_semana.values, color=colores_sem, alpha=0.85, width=0.6)
ax.set_xticks(np.arange(7))
ax.set_xticklabels(nombres_dias, rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("④ Beneficio Medio por Día de la Semana", **TKW)
_ax(ax, "€/día")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# =============================================================================
# DASHBOARD 2 — PRECIOS OMIE Y SPREADS
# =============================================================================

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG)  # 2 filas, 1 columna
fig2.canvas.manager.set_window_title(f"Dashboard Mensual 2 — Precios | {TITULO_MES}")

precio_medio_mes = resumen["precio_medio [€/MWh]"].mean() if "precio_medio [€/MWh]" in resumen.columns else 0
spread_medio_mes = resumen["spread [€/MWh]"].mean()       if "spread [€/MWh]"       in resumen.columns else 0

fig2.suptitle(
    f"PRECIOS OMIE Y SPREADS — {TITULO_MES}    "
    f"[Precio medio: {precio_medio_mes:.2f} €/MWh  |  Spread medio: {spread_medio_mes:.2f} €/MWh]",
    fontsize=11, fontweight="bold", y=0.97
)

# ⑤ Precio medio / máx / mín diario
ax = axes2[0]
if "precio_medio [€/MWh]" in resumen.columns:
    ax.fill_between(x, resumen["precio_min [€/MWh]"], resumen["precio_max [€/MWh]"],
                    alpha=0.12, color=CP, label="Rango")
    ax.plot(x, resumen["precio_max [€/MWh]"], color=CRU, lw=1,  ls="--", alpha=0.7, label="Máx")
    ax.plot(x, resumen["precio_min [€/MWh]"], color=CRD, lw=1,  ls="--", alpha=0.7, label="Mín")
    ax.plot(x, resumen["precio_medio [€/MWh]"], color=CP, lw=2, label="Medio")
ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
ax.set_title("⑤ Precio Diario (Máx / Medio / Mín)", **TKW)
_ax(ax, "€/MWh"); _xt(ax); ax.legend(**GKW)

# ⑥ Spread diario
ax = axes2[1]
if "spread [€/MWh]" in resumen.columns:
    colores_sp = plt.cm.RdYlGn(resumen["spread [€/MWh]"] / resumen["spread [€/MWh]"].max())
    ax.bar(x, resumen["spread [€/MWh]"], color=colores_sp, alpha=0.85, width=0.7)
    ax.axhline(spread_medio_mes, color=CACC, lw=1.2, ls=":",
               label=f"Media: {spread_medio_mes:.1f} €/MWh")
ax.set_title("⑥ Spread Diario (Precio Máx − Mín)", **TKW)
_ax(ax, "€/MWh"); _xt(ax); ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 3 — COMPORTAMIENTO DE LA BATERÍA
# =============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig3.canvas.manager.set_window_title(f"Dashboard Mensual 3 — Batería | {TITULO_MES}")

ciclos_mes = ((resumen["energia_cargada [MWh]"] + resumen["energia_descargada [MWh]"]) / (2 * E_MAX_VAL)).sum() if "energia_cargada [MWh]" in resumen.columns else 0

fig3.suptitle(
    f"COMPORTAMIENTO DE LA BATERÍA — {TITULO_MES}    "
    f"[Ciclos totales: {ciclos_mes:.1f}  |  Capacidad: {E_MAX_VAL} MWh  |  SOC_min: {SOC_MIN_VAL:.2f} MWh]",
    fontsize=11, fontweight="bold", y=0.97
)

# ⑨ SOC continuo a lo largo del mes
ax = axes3[0, 0]
if not detalle.empty and "SOC [MWh]" in detalle.columns:
    soc_total = detalle["SOC [MWh]"].values
    x_det     = np.arange(len(soc_total))
    ax.fill_between(x_det, soc_total, alpha=0.15, color=CSOC)
    ax.plot(x_det, soc_total, color=CSOC, lw=0.8)
    ax.axhline(E_MAX_VAL,   color=CB,  ls=":", lw=1.2, label=f"E_max ({E_MAX_VAL} MWh)")
    ax.axhline(SOC_MIN_VAL, color=CP,  ls=":", lw=1.2, label=f"SOC_min ({SOC_MIN_VAL:.2f} MWh)")
    for i in range(1, N):
        ax.axvline(i * 96, color="#DDDDDD", lw=0.5, ls="--")
    tick_vals = [i * 96 for i in range(N)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(0, len(soc_total))
    ax.set_ylim(-0.05, E_MAX_VAL * 1.15)
ax.set_title("⑨ SOC — Todo el Mes", **TKW)
_ax(ax, "MWh"); ax.legend(**GKW)

# ⑩ Ciclos equivalentes por día
ax = axes3[0, 1]
if "energia_cargada [MWh]" in resumen.columns:
    ciclos_dia = (resumen["energia_cargada [MWh]"] + resumen["energia_descargada [MWh]"]) / (2 * E_MAX_VAL)
    ax.bar(x, ciclos_dia, color=CF, alpha=0.85, width=0.7)
    ax.axhline(ciclos_dia.mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {ciclos_dia.mean():.2f} ciclos/día")
ax.set_title("⑩ Ciclos Equivalentes por Día", **TKW)
_ax(ax, "ciclos/día"); _xt(ax); ax.legend(**GKW)

# ⑪ Energía cargada vs descargada por día
ax = axes3[1, 0]
if "energia_cargada [MWh]" in resumen.columns:
    ax.bar(x - 0.2, resumen["energia_cargada [MWh]"],    width=0.4, color=CCH, alpha=0.8, label="Cargada")
    ax.bar(x + 0.2, resumen["energia_descargada [MWh]"], width=0.4, color=CDI, alpha=0.8, label="Descargada")
ax.set_title("⑪ Energía Cargada vs Descargada por Día", **TKW)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

# ⑫ SOC medio por hora del día
ax = axes3[1, 1]
if not detalle.empty and "SOC [MWh]" in detalle.columns:
    soc_hora = detalle.groupby("hora")["SOC [MWh]"].mean()
    ax.fill_between(soc_hora.index, soc_hora.values, alpha=0.15, color=CSOC)
    ax.plot(soc_hora.index, soc_hora.values, color=CSOC, lw=2, marker="o", ms=4)
    ax.axhline(E_MAX_VAL,   color=CB, ls=":", lw=1.2, label=f"E_max")
    ax.axhline(SOC_MIN_VAL, color=CP, ls=":", lw=1.2, label=f"SOC_min")
    ax.set_ylim(-0.05, E_MAX_VAL * 1.15)
ax.set_title("⑫ SOC Medio por Hora del Día", **TKW)
_ax(ax, "MWh"); _xt_horas(ax); ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# DASHBOARD 4 — RESERVA SECUNDARIA (aFRR)
# =============================================================================
# =============================================================================
# DASHBOARD 4 — RESERVA SECUNDARIA (aFRR)
# =============================================================================

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
fig4.canvas.manager.set_window_title(f"Dashboard Mensual 4 — aFRR | {TITULO_MES}")

reserva_total = resumen["reserva_up [MWh]"].sum() + resumen["reserva_down [MWh]"].sum() if "reserva_up [MWh]" in resumen.columns else 0
activ_total   = resumen["activacion_up [MWh]"].sum() + resumen["activacion_down [MWh]"].sum() if "activacion_up [MWh]" in resumen.columns else 0

fig4.suptitle(
    f"RESERVA SECUNDARIA (aFRR) — {TITULO_MES}    "
    f"[Reserva total: {reserva_total:.1f} MWh  |  Activación total: {activ_total:.1f} MWh]",
    fontsize=11, fontweight="bold", y=0.97
)

# ⑬ Banda de reserva ofertada diaria
ax = axes4[0, 0]
if "reserva_up [MWh]" in resumen.columns:
    ax.bar(x - 0.2, resumen["reserva_up [MWh]"],   width=0.4, color=CRU, alpha=0.8, label="Reserva Up")
    ax.bar(x + 0.2, resumen["reserva_down [MWh]"], width=0.4, color=CRD, alpha=0.8, label="Reserva Down")
ax.set_title("⑬ Banda de Reserva Ofertada (aFRR)", **TKW)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

# ⑭ Activaciones diarias
ax = axes4[0, 1]
if "activacion_up [MWh]" in resumen.columns:
    ax.bar(x - 0.2, resumen["activacion_up [MWh]"],   width=0.4, color=CAU, alpha=0.85, label="Act. Up")
    ax.bar(x + 0.2, resumen["activacion_down [MWh]"], width=0.4, color=CAD, alpha=0.85, label="Act. Down")
ax.set_title("⑭ Energía Activada por Día (aFRR)", **TKW)
_ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

# ⑮ Scatter: spread diario vs beneficio diario
ax = axes4[1, 0]
if "spread [€/MWh]" in resumen.columns:
    spreads    = resumen["spread [€/MWh]"].values
    beneficios = resumen["beneficio [€]"].values

    colores_sc = [CS if v >= 0 else CB for v in beneficios]
    ax.scatter(spreads, beneficios, color=colores_sc, alpha=0.85, s=60, zorder=3)

    # Línea de tendencia
    z = np.polyfit(spreads, beneficios, 1)
    p = np.poly1d(z)
    x_line = np.linspace(spreads.min(), spreads.max(), 100)
    ax.plot(x_line, p(x_line), color=CACC, lw=1.5, ls="--", label="Tendencia")

    # Correlación de Pearson
    corr = np.corrcoef(spreads, beneficios)[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=9, color=CACC, fontweight="bold")

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")

ax.set_title("⑮ Spread Diario vs Beneficio Diario", **TKW)
ax.set_xlabel("Spread €/MWh", fontsize=8, color="#555")
_ax(ax, "€"); ax.legend(**GKW)

# ⑯ Reserva y activación media por hora del día
ax = axes4[1, 1]
if not detalle.empty and "r_up [MWh]" in detalle.columns:
    res_up_h   = detalle.groupby("hora")["r_up [MWh]"].mean()
    res_down_h = detalle.groupby("hora")["r_down [MWh]"].mean()
    act_up_h   = detalle.groupby("hora")["a_up [MWh]"].mean()
    act_down_h = detalle.groupby("hora")["a_down [MWh]"].mean()
    ax.bar(res_up_h.index - 0.3, res_up_h.values,   width=0.2, color=CRU, alpha=0.5, label="Reserva Up")
    ax.bar(res_up_h.index - 0.1, res_down_h.values, width=0.2, color=CRD, alpha=0.5, label="Reserva Down")
    ax.bar(res_up_h.index + 0.1, act_up_h.values,   width=0.2, color=CAU, alpha=0.8, label="Act. Up")
    ax.bar(res_up_h.index + 0.3, act_down_h.values, width=0.2, color=CAD, alpha=0.8, label="Act. Down")
ax.set_title("⑯ Reserva y Activación Media por Hora", **TKW)
_ax(ax, "MWh"); _xt_horas(ax); ax.legend(**GKW)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# MOSTRAR
# =============================================================================

print(f"\n=== Dashboards mensuales generados — {TITULO_MES} ===")
plt.show()