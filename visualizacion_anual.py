"""
Visualizador Central — Batería Pura (Arbitraje + aFRR)
Autor: Hugo Raggini Paternain
------------------------------------------------------
Punto de entrada único para visualizar resultados del TFG.

Modos:
  det  → resultados del optimizador (determinista)
  sim  → resultados de simulación Monte Carlo
  comp → comparación determinista vs simulación

Uso interactivo:
  python visualizador.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================================================
# CONFIGURACION
# =============================================================================

NOMBRES_MES = {
    1: "Enero",      2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",       6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre",  12: "Diciembre"
}

CARPETA_RES     = Path("Resultados")
CARPETA_SIM     = Path("Resultados_Sim")
CARPETA_PRECIOS = Path("Precios")
COLUMNAS_Q      = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]

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

# =============================================================================
# HELPERS — CARGA DE DATOS
# =============================================================================

def meses_en_rango(mes_ini, anio_ini, mes_fin, anio_fin):
    """Genera lista de (mes, anio) en el rango indicado."""
    meses = []
    m, a = mes_ini, anio_ini
    while (a, m) <= (anio_fin, mes_fin):
        meses.append((m, a))
        m += 1
        if m > 12:
            m = 1
            a += 1
    return meses

def cargar_resumen_det(mes, anio):
    """Carga el resumen mensual determinista."""
    csv = CARPETA_RES / f"{anio}-{mes:02d}" / f"resumen_{anio}_{mes:02d}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes_label"] = f"{NOMBRES_MES[mes][:3]} {anio}"
    return df

def cargar_resumen_sim(mes, anio):
    """Carga el resumen mensual de simulacion."""
    csv = CARPETA_SIM / f"{anio}-{mes:02d}" / f"resumen_sim_{anio}_{mes:02d}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes_label"] = f"{NOMBRES_MES[mes][:3]} {anio}"
    return df

def cargar_precios_mes(mes, anio):
    """Carga CSV de precios del mes."""
    csv = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[mes].lower()}_{anio}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

# =============================================================================
# INPUT INTERACTIVO
# =============================================================================

def pedir_modo():
    print("\n" + "="*55)
    print("  VISUALIZADOR CENTRAL — BATERIA PURA (Arbitraje+aFRR)")
    print("="*55)
    print("  Modos disponibles:")
    print("    det  → Resultados del optimizador (determinista)")
    print("    sim  → Resultados de simulacion Monte Carlo")
    print("    comp → Comparacion determinista vs simulacion")
    print("="*55)
    while True:
        modo = input("\n  Modo (det / sim / comp): ").strip().lower()
        if modo in ("det", "sim", "comp"):
            return modo
        print("  [!] Introduce 'det', 'sim' o 'comp'")

def pedir_rango():
    print("\n  Introduce el rango de fechas a visualizar.")
    while True:
        try:
            mes_ini  = int(input("  Mes inicio   (1-12): "))
            anio_ini = int(input("  Año inicio        : "))
            mes_fin  = int(input("  Mes fin      (1-12): "))
            anio_fin = int(input("  Año fin           : "))
            if (anio_ini, mes_ini) <= (anio_fin, mes_fin):
                return mes_ini, anio_ini, mes_fin, anio_fin
            print("  [!] El rango es inválido — la fecha inicio debe ser anterior al fin.")
        except ValueError:
            print("  [!] Introduce números válidos.")

# =============================================================================
# MODO DET — DASHBOARDS DETERMINISTAS
# =============================================================================

def modo_det(meses, titulo_rango):
    """Carga todos los resúmenes mensuales deterministas y genera dashboards."""

    dfs = []
    for m, a in meses:
        df = cargar_resumen_det(m, a)
        if df is not None:
            dfs.append(df)
        else:
            print(f"  [!] Sin datos deterministas para {NOMBRES_MES[m]} {a}")

    if not dfs:
        print("  [!] No se encontraron datos deterministas en el rango indicado.")
        return

    det = pd.concat(dfs, ignore_index=True).sort_values("fecha").reset_index(drop=True)
    x   = np.arange(len(det))
    lbl_fecha = det["fecha"].dt.strftime("%d/%m/%y")

    # Etiquetas de mes para separadores
    meses_cambio = det.groupby(det["fecha"].dt.to_period("M")).first().index

    def _xt(ax, paso=7):
        ticks = x[::paso]
        ax.set_xticks(ticks)
        ax.set_xticklabels(lbl_fecha.iloc[ticks], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, len(det) - 0.5)

    def _separadores(ax):
        """Líneas verticales en cambio de mes."""
        for _, grupo in det.groupby(det["fecha"].dt.to_period("M")):
            idx = grupo.index[0]
            if idx > 0:
                ax.axvline(idx - 0.5, color="#DDDDDD", lw=0.8, ls="--", zorder=0)

    # Dashboard 1 — Beneficio y acumulado
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig1.canvas.manager.set_window_title(f"Det 1 — Beneficio | {titulo_rango}")
    fig1.suptitle(
        f"RESULTADOS DETERMINISTAS — {titulo_rango}    "
        f"[Total: {det['beneficio [€]'].sum():+.2f}€  |  "
        f"Media: {det['beneficio [€]'].mean():+.2f}€/día]",
        fontsize=11, fontweight="bold", y=0.97
    )

    ax = axes1[0, 0]
    colores_b = [CS if v >= 0 else CB for v in det["beneficio [€]"]]
    ax.bar(x, det["beneficio [€]"], color=colores_b, alpha=0.85, width=0.7)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.axhline(det["beneficio [€]"].mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {det['beneficio [€]'].mean():+.2f}€")
    _separadores(ax)
    ax.set_title("Beneficio Neto Diario", **TKW)
    _ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

    ax = axes1[0, 1]
    acum = det["beneficio [€]"].cumsum()
    color_acum = CS if acum.iloc[-1] >= 0 else CB
    ax.fill_between(x, acum, alpha=0.15, color=color_acum)
    ax.plot(x, acum, color=color_acum, lw=2)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    _separadores(ax)
    ax.set_title("Beneficio Acumulado", **TKW)
    _ax(ax, "EUR"); _xt(ax)

    ax = axes1[1, 0]
    if "spread [€/MWh]" in det.columns:
        colores_sp = plt.cm.RdYlGn(det["spread [€/MWh]"] / det["spread [€/MWh]"].max())
        ax.bar(x, det["spread [€/MWh]"], color=colores_sp, alpha=0.85, width=0.7)
        ax.axhline(det["spread [€/MWh]"].mean(), color=CACC, lw=1.2, ls=":",
                   label=f"Media: {det['spread [€/MWh]'].mean():.1f}€/MWh")
        _separadores(ax)
    ax.set_title("Spread Diario (Máx − Mín)", **TKW)
    _ax(ax, "EUR/MWh"); _xt(ax); ax.legend(**GKW)

    ax = axes1[1, 1]
    if "precio_medio [€/MWh]" in det.columns:
        ax.fill_between(x, det["precio_min [€/MWh]"], det["precio_max [€/MWh]"],
                        alpha=0.12, color=CP, label="Rango")
        ax.plot(x, det["precio_medio [€/MWh]"], color=CP, lw=1.5, label="Precio medio")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
        _separadores(ax)
    ax.set_title("Precio OMIE Diario (Máx/Medio/Mín)", **TKW)
    _ax(ax, "EUR/MWh"); _xt(ax); ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Dashboard 2 — Reserva y batería
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig2.canvas.manager.set_window_title(f"Det 2 — Batería y Reserva | {titulo_rango}")
    fig2.suptitle(f"BATERÍA Y RESERVA SECUNDARIA — {titulo_rango}",
                  fontsize=11, fontweight="bold", y=0.97)

    ax = axes2[0, 0]
    if "reserva_up [MWh]" in det.columns:
        ax.bar(x - 0.2, det["reserva_up [MWh]"],   width=0.4, color=CRU, alpha=0.8, label="Reserva Up")
        ax.bar(x + 0.2, det["reserva_down [MWh]"], width=0.4, color=CB,  alpha=0.8, label="Reserva Down")
        _separadores(ax)
    ax.set_title("Banda de Reserva Ofertada (aFRR)", **TKW)
    _ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

    ax = axes2[0, 1]
    if "activacion_up [MWh]" in det.columns:
        ax.bar(x - 0.2, det["activacion_up [MWh]"],   width=0.4, color=CAU, alpha=0.85, label="Act. Up")
        ax.bar(x + 0.2, det["activacion_down [MWh]"], width=0.4, color=CAD, alpha=0.85, label="Act. Down")
        _separadores(ax)
    ax.set_title("Energía Activada aFRR", **TKW)
    _ax(ax, "MWh"); _xt(ax); ax.legend(**GKW)

    ax = axes2[1, 0]
    if "energia_cargada [MWh]" in det.columns:
        ciclos = (det["energia_cargada [MWh]"] + det["energia_descargada [MWh]"]) / (2 * 2.0)
        ax.bar(x, ciclos, color=CF, alpha=0.85, width=0.7)
        ax.axhline(ciclos.mean(), color=CACC, lw=1.2, ls=":",
                   label=f"Media: {ciclos.mean():.2f} ciclos/día")
        _separadores(ax)
    ax.set_title("Ciclos Equivalentes por Día", **TKW)
    _ax(ax, "ciclos/día"); _xt(ax); ax.legend(**GKW)

    ax = axes2[1, 1]
    if "spread [€/MWh]" in det.columns:
        spreads    = det["spread [€/MWh]"].values
        beneficios = det["beneficio [€]"].values
        colores_sc = [CS if v >= 0 else CB for v in beneficios]
        ax.scatter(spreads, beneficios, color=colores_sc, alpha=0.8, s=30, zorder=3)
        if len(spreads) > 2:
            z = np.polyfit(spreads, beneficios, 1)
            xl = np.linspace(spreads.min(), spreads.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color=CACC, lw=1.5, ls="--", label="Tendencia")
            corr = np.corrcoef(spreads, beneficios)[0, 1]
            ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=9, color=CACC, fontweight="bold")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Spread vs Beneficio Diario", **TKW)
    ax.set_xlabel("Spread (EUR/MWh)", fontsize=8, color="#555")
    _ax(ax, "EUR"); ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MODO SIM — DASHBOARDS SIMULACION
# =============================================================================

def modo_sim(meses, titulo_rango):
    """Carga todos los resúmenes de simulación y genera dashboards."""

    dfs = []
    for m, a in meses:
        df = cargar_resumen_sim(m, a)
        if df is not None:
            dfs.append(df)
        else:
            print(f"  [!] Sin datos de simulación para {NOMBRES_MES[m]} {a}")

    if not dfs:
        print("  [!] No se encontraron datos de simulación en el rango indicado.")
        return

    sim = pd.concat(dfs, ignore_index=True).sort_values("fecha").reset_index(drop=True)
    x   = np.arange(len(sim))
    lbl_fecha = sim["fecha"].dt.strftime("%d/%m/%y")

    def _xt(ax, paso=7):
        ticks = x[::paso]
        ax.set_xticks(ticks)
        ax.set_xticklabels(lbl_fecha.iloc[ticks], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, len(sim) - 0.5)

    def _sep(ax):
        for _, grupo in sim.groupby(sim["fecha"].dt.to_period("M")):
            idx = grupo.index[0]
            if idx > 0:
                ax.axvline(idx - 0.5, color="#DDDDDD", lw=0.8, ls="--", zorder=0)

    prev_total = sim["beneficio_previsto [€]"].sum()
    p50_total  = sim["ben_P50 [€]"].sum()
    coste_inc  = prev_total - p50_total

    # Dashboard 1 — Previsto vs real
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig1.canvas.manager.set_window_title(f"Sim 1 — Previsto vs Real | {titulo_rango}")
    fig1.suptitle(
        f"SIMULACION — PREVISTO VS REAL — {titulo_rango}    "
        f"[Previsto: {prev_total:+.2f}€  |  P50: {p50_total:+.2f}€  |  "
        f"Coste incertidumbre: {coste_inc:+.2f}€ ({coste_inc/max(abs(prev_total),1)*100:.1f}%)]",
        fontsize=11, fontweight="bold", y=0.97
    )

    ax = axes1[0, 0]
    ax.plot(x, sim["beneficio_previsto [€]"], color=CACC, lw=1.5, label="Previsto")
    ax.plot(x, sim["ben_P50 [€]"],            color=CS,   lw=1.5, label="P50 real")
    ax.fill_between(x, sim["ben_P10 [€]"], sim["ben_P90 [€]"],
                    alpha=0.12, color=CS, label="Rango P10-P90")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    _sep(ax)
    ax.set_title("Beneficio Previsto vs Real (P10/P50/P90)", **TKW)
    _ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

    ax = axes1[0, 1]
    desv = sim["desv_media [€]"]
    ax.bar(x, desv, color=[CS if v >= 0 else CB for v in desv], alpha=0.85, width=0.7)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.axhline(desv.mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {desv.mean():+.2f}€")
    _sep(ax)
    ax.set_title("Desviación Diaria (Real − Previsto)", **TKW)
    _ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

    ax = axes1[1, 0]
    p5_diario = sim["VaR_95 [€]"]
    ax.plot(x, sim["ben_P50 [€]"], color=CS,   lw=1.5, marker="", label="P50")
    ax.plot(x, sim["ben_P10 [€]"], color=CF,   lw=1.2, label="P10")
    ax.plot(x, p5_diario,           color=CB,   lw=1.5, label="P5 (VaR95)")
    ax.fill_between(x, p5_diario, sim["ben_P10 [€]"],
                    alpha=0.15, color=CB, label="Zona P5-P10")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    _sep(ax)
    ax.set_title("Percentiles de Riesgo Diarios", **TKW)
    _ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

    ax = axes1[1, 1]
    acum_prev = sim["beneficio_previsto [€]"].cumsum()
    acum_p50  = sim["ben_P50 [€]"].cumsum()
    acum_p10  = sim["ben_P10 [€]"].cumsum()
    acum_p90  = sim["ben_P90 [€]"].cumsum()
    ax.fill_between(x, acum_p10, acum_p90, alpha=0.12, color=CS, label="Rango P10-P90")
    ax.plot(x, acum_prev, color=CACC, lw=2, ls="--", label="Previsto acumulado")
    ax.plot(x, acum_p50,  color=CS,   lw=2,           label="P50 acumulado")
    ax.plot(x, acum_p10,  color=CB,   lw=1, alpha=0.7, label="P10 acumulado")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    _sep(ax)
    ax.set_title("Beneficio Acumulado — Previsto vs Escenarios", **TKW)
    _ax(ax, "EUR"); _xt(ax); ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MODO COMP — COMPARACION DETERMINISTA VS SIMULACION
# =============================================================================

def modo_comp(meses, titulo_rango):
    """Compara resultados deterministas vs simulacion en el rango."""

    dfs_det, dfs_sim = [], []
    for m, a in meses:
        d = cargar_resumen_det(m, a)
        s = cargar_resumen_sim(m, a)
        if d is not None:
            dfs_det.append(d)
        else:
            print(f"  [!] Sin datos det para {NOMBRES_MES[m]} {a}")
        if s is not None:
            dfs_sim.append(s)
        else:
            print(f"  [!] Sin datos sim para {NOMBRES_MES[m]} {a}")

    if not dfs_det or not dfs_sim:
        print("  [!] Faltan datos para la comparación.")
        return

    det = pd.concat(dfs_det, ignore_index=True).sort_values("fecha").reset_index(drop=True)
    sim = pd.concat(dfs_sim, ignore_index=True).sort_values("fecha").reset_index(drop=True)

    # Alinear por fecha
    det = det[det["fecha"].isin(sim["fecha"])].reset_index(drop=True)
    sim = sim[sim["fecha"].isin(det["fecha"])].reset_index(drop=True)

    # Agregacion mensual
    det["mes"] = det["fecha"].dt.to_period("M")
    sim["mes"] = sim["fecha"].dt.to_period("M")

    det_mes = det.groupby("mes").agg(
        ben_total=("beneficio [€]", "sum"),
        dias=("beneficio [€]", "count")
    ).reset_index()

    sim_mes = sim.groupby("mes").agg(
        prev_total=("beneficio_previsto [€]", "sum"),
        p50_total=("ben_P50 [€]", "sum"),
        p10_total=("ben_P10 [€]", "sum"),
        p90_total=("ben_P90 [€]", "sum"),
    ).reset_index()

    merged = det_mes.merge(sim_mes, on="mes")
    merged["coste_inc"] = merged["prev_total"] - merged["p50_total"]
    merged["coste_inc_pct"] = merged["coste_inc"] / merged["prev_total"].abs().clip(lower=1) * 100
    merged["mes_label"] = merged["mes"].dt.strftime("%b %Y")

    xm  = np.arange(len(merged))
    lbl = merged["mes_label"].values

    # Nivel diario para gráfica P10/P50/P90
    x   = np.arange(len(sim))
    lbl_d = sim["fecha"].dt.strftime("%d/%m/%y")

    def _xt_d(ax, paso=7):
        ticks = x[::paso]
        ax.set_xticks(ticks)
        ax.set_xticklabels(lbl_d.iloc[ticks], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, len(sim) - 0.5)

    def _sep_d(ax):
        for _, grupo in sim.groupby(sim["fecha"].dt.to_period("M")):
            idx = grupo.index[0]
            if idx > 0:
                ax.axvline(idx - 0.5, color="#DDDDDD", lw=0.8, ls="--", zorder=0)

    tot_det  = merged["ben_total"].sum()
    tot_p50  = merged["p50_total"].sum()
    tot_inc  = merged["coste_inc"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(f"Comparación Det vs Sim | {titulo_rango}")
    fig.suptitle(
        f"COMPARACION DETERMINISTA VS SIMULACION — {titulo_rango}    "
        f"[Det total: {tot_det:+.2f}€  |  Sim P50: {tot_p50:+.2f}€  |  "
        f"Coste incertidumbre: {tot_inc:+.2f}€ ({tot_inc/max(abs(tot_det),1)*100:.1f}%)]",
        fontsize=11, fontweight="bold", y=0.97
    )

    # 1 Beneficio total acumulado por mes
    ax = axes[0, 0]
    ancho = 0.35
    ax.bar(xm - ancho/2, merged["ben_total"], width=ancho,
           color=CACC, alpha=0.85, label="Determinista (real histórico)")
    ax.bar(xm + ancho/2, merged["p50_total"], width=ancho,
           color=CS,   alpha=0.85, label="Simulación P50")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_xticks(xm)
    ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=8)
    ax.set_title("Beneficio Total por Mes — Det vs Sim P50", **TKW)
    _ax(ax, "EUR"); ax.legend(**GKW)

    # 2 Coste de incertidumbre mes a mes
    ax = axes[0, 1]
    colores_ci = [CB if v > 0 else CS for v in merged["coste_inc"]]
    bars = ax.bar(xm, merged["coste_inc"], color=colores_ci, alpha=0.85, width=0.6)
    ax2r = ax.twinx()
    ax2r.plot(xm, merged["coste_inc_pct"], color=CF, lw=2, marker="o", ms=5,
              label="Coste inc. %")
    ax2r.set_ylabel("Coste incertidumbre (%)", fontsize=8, color=CF)
    ax2r.tick_params(colors=CF, labelsize=8)
    ax2r.spines[["top"]].set_visible(False)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.axhline(merged["coste_inc"].mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {merged['coste_inc'].mean():+.2f}€")
    ax.set_xticks(xm)
    ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=8)
    ax.set_title("Coste de Incertidumbre Mensual", **TKW)
    _ax(ax, "EUR")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, **GKW)

    # 3 P10/P50/P90 vs Previsto — nivel diario
    ax = axes[1, 0]
    ax.plot(x, sim["beneficio_previsto [€]"], color=CACC, lw=1.5, ls="--",
            label="Previsto (modelo)")
    ax.plot(x, sim["ben_P50 [€]"],            color=CS,   lw=1.5,
            label="P50 real")
    ax.fill_between(x, sim["ben_P10 [€]"], sim["ben_P90 [€]"],
                    alpha=0.12, color=CS, label="Rango P10-P90")
    ax.fill_between(x, sim["ben_P10 [€]"], sim["ben_P90 [€]"].where(
        sim["ben_P90 [€]"] < sim["beneficio_previsto [€]"], other=sim["beneficio_previsto [€]"]),
        alpha=0.10, color=CB)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    _sep_d(ax)
    ax.set_title("P10/P50/P90 vs Previsto — Diario", **TKW)
    _ax(ax, "EUR"); _xt_d(ax); ax.legend(**GKW)

    # 4 Acumulados comparativos
    ax = axes[1, 1]
    acum_det  = det["beneficio [€]"].values
    # Alinear det con sim por fecha
    det_aligned = det.set_index("fecha")
    sim_aligned = sim.set_index("fecha")
    fechas_comunes = sim_aligned.index.intersection(det_aligned.index)
    acum_d   = det_aligned.loc[fechas_comunes, "beneficio [€]"].cumsum().values
    acum_p50 = sim_aligned.loc[fechas_comunes, "ben_P50 [€]"].cumsum().values
    acum_p10 = sim_aligned.loc[fechas_comunes, "ben_P10 [€]"].cumsum().values
    acum_p90 = sim_aligned.loc[fechas_comunes, "ben_P90 [€]"].cumsum().values
    xc = np.arange(len(fechas_comunes))

    ax.fill_between(xc, acum_p10, acum_p90, alpha=0.12, color=CS, label="Rango P10-P90 sim")
    ax.plot(xc, acum_d,   color=CACC, lw=2, ls="--", label="Det acumulado")
    ax.plot(xc, acum_p50, color=CS,   lw=2,           label="Sim P50 acumulado")
    ax.plot(xc, acum_p10, color=CB,   lw=1, alpha=0.7, label="Sim P10 acumulado")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")

    # Separadores de mes
    fechas_ser = pd.Series(fechas_comunes)
    for _, grupo in fechas_ser.groupby(fechas_ser.dt.to_period("M")):
        idx = grupo.index[0]
        if idx > 0:
            ax.axvline(idx - 0.5, color="#DDDDDD", lw=0.8, ls="--", zorder=0)

    ax.set_title("Beneficio Acumulado — Det vs Simulación", **TKW)
    _ax(ax, "EUR")
    paso = max(1, len(xc) // 8)
    ax.set_xticks(xc[::paso])
    ax.set_xticklabels(
        pd.Series(fechas_comunes[::paso]).dt.strftime("%d/%m/%y"),
        rotation=45, ha="right", fontsize=7
    )
    ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    modo = pedir_modo()
    mes_ini, anio_ini, mes_fin, anio_fin = pedir_rango()
    meses = meses_en_rango(mes_ini, anio_ini, mes_fin, anio_fin)

    titulo_rango = (
        f"{NOMBRES_MES[mes_ini][:3]} {anio_ini}"
        if len(meses) == 1
        else f"{NOMBRES_MES[mes_ini][:3]} {anio_ini} — {NOMBRES_MES[mes_fin][:3]} {anio_fin}"
    )

    print(f"\n  Modo     : {modo.upper()}")
    print(f"  Periodo  : {titulo_rango}")
    print(f"  Meses    : {len(meses)}\n")

    if   modo == "det":
        modo_det(meses, titulo_rango)
    elif modo == "sim":
        modo_sim(meses, titulo_rango)
    elif modo == "comp":
        modo_comp(meses, titulo_rango)