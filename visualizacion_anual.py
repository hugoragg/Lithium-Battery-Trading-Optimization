"""
Visualizador Central — Batería Pura (Arbitraje + aFRR)
Autor: Hugo Raggini Paternain
------------------------------------------------------
Punto de entrada único para visualizar resultados del TFG.
Toda la visualización es a nivel MENSUAL.

Modos:
  det  → resultados del optimizador (determinista)
  sim  → resultados de simulación Monte Carlo
  comp → comparación determinista vs simulación

Uso:
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

CARPETA_RES = Path("Resultados")
CARPETA_SIM = Path("Resultados_Sim")

CF, CS, CB, CCH, CDI, CSOC, CP = "#F5A623", "#27AE60", "#E74C3C", "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22"
CRU, CRD, CAU, CAD, CACC, BG, PBG = "#E74C3C", "#3498DB", "#C0392B", "#2980B9", "#2C3E50", "#F5F6FA", "#FFFFFF"

TKW = dict(fontsize=10, fontweight="bold", color="#2C3E50", pad=12, loc="left")
LKW = dict(fontsize=9, color="#555")
GKW = dict(fontsize=8, framealpha=0.8, edgecolor="none", ncol=2)

def _ax(ax, ylabel=""):
    ax.set_facecolor(PBG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555", labelsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.5, color="#AAAAAA")
    if ylabel:
        ax.set_ylabel(ylabel, **LKW)

def _xt(ax, lbl):
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=9)
    ax.set_xlim(-0.5, len(lbl) - 0.5)

# =============================================================================
# HELPERS
# =============================================================================

def meses_en_rango(mes_ini, anio_ini, mes_fin, anio_fin):
    meses = []
    m, a = mes_ini, anio_ini
    while (a, m) <= (anio_fin, mes_fin):
        meses.append((m, a))
        m += 1
        if m > 12:
            m, a = 1, a + 1
    return meses

def cargar_resumen_det(mes, anio):
    csv = CARPETA_RES / f"{anio}-{mes:02d}" / f"resumen_{anio}_{mes:02d}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def cargar_resumen_sim(mes, anio):
    csv = CARPETA_SIM / f"{anio}-{mes:02d}" / f"resumen_sim_{anio}_{mes:02d}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def agregar_mensual_det(dfs_det, meses):
    """Agrega CSVs diarios deterministas a nivel mensual."""
    filas = []
    for (m, a), df in zip(meses, dfs_det):
        label = f"{NOMBRES_MES[m][:3]} {a}"
        ok = df.dropna(subset=["beneficio [€]"])
        fila = {
            "mes_label":              label,
            "mes":                    f"{a}-{m:02d}",
            "beneficio_total [€]":    ok["beneficio [€]"].sum(),
            "beneficio_medio [€]":    ok["beneficio [€]"].mean(),
            "beneficio_max [€]":      ok["beneficio [€]"].max(),
            "beneficio_min [€]":      ok["beneficio [€]"].min(),
            "dias":                   len(ok),
            "dias_positivos":         (ok["beneficio [€]"] > 0).sum(),
        }
        for col in ["energia_vendida [MWh]", "energia_comprada [MWh]",
                    "reserva_up [MWh]", "reserva_down [MWh]",
                    "activacion_up [MWh]", "activacion_down [MWh]",
                    "precio_medio [€/MWh]", "precio_max [€/MWh]",
                    "precio_min [€/MWh]", "spread [€/MWh]"]:
            if col in ok.columns:
                fila[col] = ok[col].mean()
        filas.append(fila)
    return pd.DataFrame(filas)

def agregar_mensual_sim(dfs_sim, meses):
    """Agrega CSVs de simulacion a nivel mensual."""
    filas = []
    for (m, a), df in zip(meses, dfs_sim):
        label = f"{NOMBRES_MES[m][:3]} {a}"
        fila = {
            "mes_label":               label,
            "mes":                     f"{a}-{m:02d}",
            "prev_total [€]":          df["beneficio_previsto [€]"].sum(),
            "p10_total [€]":           df["ben_P10 [€]"].sum(),
            "p50_total [€]":           df["ben_P50 [€]"].sum(),
            "p90_total [€]":           df["ben_P90 [€]"].sum(),
            "p50_medio [€/dia]":       df["ben_P50 [€]"].mean(),
            "p10_medio [€/dia]":       df["ben_P10 [€]"].mean(),
            "p90_medio [€/dia]":       df["ben_P90 [€]"].mean(),
            "var95_medio [€/dia]":     df["VaR_95 [€]"].mean(),
            "std_medio [€/dia]":       df["ben_std [€]"].mean(),
            "desv_total [€]":          df["desv_media [€]"].sum(),
            "pct_neg_medio [%]":       df["pct_negativo [%]"].mean(),
            "peor_dia [€]":            df["peor_escenario [€]"].min(),
            "cvar_medio [€/dia]":      df["CVaR_95 [€]"].mean(),
            "coste_inc [€]":           df["beneficio_previsto [€]"].sum() - df["ben_P50 [€]"].sum(),
        }
        if "penalizacion_soc_media [€]" in df.columns:
            fila["pen_soc_total [€]"] = df["penalizacion_soc_media [€]"].sum()
        filas.append(fila)
    return pd.DataFrame(filas)

# =============================================================================
# INPUT INTERACTIVO
# =============================================================================

def pedir_modo():
    print("\n" + "="*58)
    print("  VISUALIZADOR CENTRAL — BATERIA PURA (Arbitraje + aFRR)")
    print("="*58)
    print("  Modos:")
    print("    det  → Resultados del optimizador (determinista)")
    print("    sim  → Resultados de simulación Monte Carlo")
    print("    comp → Comparación determinista vs simulación")
    print("="*58)
    while True:
        modo = input("\n  Modo (det / sim / comp): ").strip().lower()
        if modo in ("det", "sim", "comp"):
            return modo
        print("  [!] Introduce 'det', 'sim' o 'comp'")

def pedir_rango():
    print("\n  Rango de meses a visualizar:")
    while True:
        try:
            mes_ini  = int(input("  Mes inicio  (1-12): "))
            anio_ini = int(input("  Año inicio       : "))
            mes_fin  = int(input("  Mes fin     (1-12): "))
            anio_fin = int(input("  Año fin          : "))
            if (anio_ini, mes_ini) <= (anio_fin, mes_fin):
                return mes_ini, anio_ini, mes_fin, anio_fin
            print("  [!] La fecha inicio debe ser anterior o igual al fin.")
        except ValueError:
            print("  [!] Introduce números válidos.")

# =============================================================================
# MODO DET
# =============================================================================

def modo_det(det, titulo_rango):
    x   = np.arange(len(det))
    lbl = det["mes_label"].tolist()

    total = det["beneficio_total [€]"].sum()
    media = det["beneficio_medio [€]"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.canvas.manager.set_window_title(f"Det — {titulo_rango}")
    fig.suptitle(
        f"RESULTADOS DETERMINISTAS — {titulo_rango}    "
        f"[Total: {total:+.2f}€  |  Media mensual: {det['beneficio_total [€]'].mean():+.2f}€/mes]",
        fontsize=11, fontweight="bold", y=0.97
    )

    # Beneficio total por mes
    ax = axes[0, 0]
    colores = [CS if v >= 0 else CB for v in det["beneficio_total [€]"]]
    ax.bar(x, det["beneficio_total [€]"], color=colores, alpha=0.85, width=0.6)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.axhline(det["beneficio_total [€]"].mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {det['beneficio_total [€]'].mean():+.2f}€")
    ax.set_title("Beneficio Total por Mes", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl); ax.legend(**GKW)

    # Beneficio acumulado
    ax = axes[0, 1]
    acum = det["beneficio_total [€]"].cumsum()
    color_a = CS if acum.iloc[-1] >= 0 else CB
    ax.fill_between(x, acum, alpha=0.15, color=color_a)
    ax.plot(x, acum, color=color_a, lw=2.5, marker="o", ms=6)
    for i, v in enumerate(acum):
        ax.annotate(f"{v:+.0f}€", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=color_a)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Beneficio Acumulado", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl)

    # Spread medio mensual
    ax = axes[1, 0]
    if "spread [€/MWh]" in det.columns:
        colores_sp = plt.cm.RdYlGn(det["spread [€/MWh]"] / det["spread [€/MWh]"].max())
        ax.bar(x, det["spread [€/MWh]"], color=colores_sp, alpha=0.85, width=0.6)
        ax.axhline(det["spread [€/MWh]"].mean(), color=CACC, lw=1.2, ls=":",
                   label=f"Media: {det['spread [€/MWh]'].mean():.1f}€/MWh")
    ax.set_title("Spread Medio Mensual (Máx − Mín)", **TKW)
    _ax(ax, "EUR/MWh"); _xt(ax, lbl); ax.legend(**GKW)

    # Scatter spread vs beneficio mensual
    ax = axes[1, 1]
    if "spread [€/MWh]" in det.columns:
        sp  = det["spread [€/MWh]"].values
        ben = det["beneficio_total [€]"].values
        ax.scatter(sp, ben, color=[CS if v >= 0 else CB for v in ben],
                   s=80, alpha=0.85, zorder=3)
        for i, (xi, yi, lab) in enumerate(zip(sp, ben, lbl)):
            ax.annotate(lab, (xi, yi), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color="#555")
        if len(sp) > 2:
            z = np.polyfit(sp, ben, 1)
            xl = np.linspace(sp.min(), sp.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color=CACC, lw=1.5, ls="--", label="Tendencia")
            corr = np.corrcoef(sp, ben)[0, 1]
            ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=9, color=CACC, fontweight="bold")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Spread vs Beneficio Mensual", **TKW)
    ax.set_xlabel("Spread medio (EUR/MWh)", fontsize=8, color="#555")
    _ax(ax, "EUR"); ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Dashboard 2 — Reserva y operación
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig2.canvas.manager.set_window_title(f"Det 2 — Reserva | {titulo_rango}")
    fig2.suptitle(f"RESERVA SECUNDARIA Y OPERACION — {titulo_rango}",
                  fontsize=11, fontweight="bold", y=0.97)

    ax = axes2[0, 0]
    if "reserva_up [MWh]" in det.columns:
        ax.bar(x - 0.2, det["reserva_up [MWh]"],   width=0.4, color=CRU, alpha=0.8, label="Reserva Up")
        ax.bar(x + 0.2, det["reserva_down [MWh]"], width=0.4, color=CB,  alpha=0.8, label="Reserva Down")
    ax.set_title("Reserva aFRR Ofertada Media Mensual", **TKW)
    _ax(ax, "MWh/dia"); _xt(ax, lbl); ax.legend(**GKW)

    ax = axes2[0, 1]
    if "activacion_up [MWh]" in det.columns:
        ax.bar(x - 0.2, det["activacion_up [MWh]"],   width=0.4, color=CAU, alpha=0.85, label="Act. Up")
        ax.bar(x + 0.2, det["activacion_down [MWh]"], width=0.4, color=CAD, alpha=0.85, label="Act. Down")
    ax.set_title("Energía Activada aFRR Media Mensual", **TKW)
    _ax(ax, "MWh/dia"); _xt(ax, lbl); ax.legend(**GKW)

    ax = axes2[1, 0]
    if "precio_medio [€/MWh]" in det.columns:
        ax.fill_between(x, det["precio_min [€/MWh]"], det["precio_max [€/MWh]"],
                        alpha=0.12, color=CP, label="Rango Máx-Mín")
        ax.plot(x, det["precio_medio [€/MWh]"], color=CP, lw=2,
                marker="o", ms=6, label="Precio medio")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Precio OMIE Mensual (Máx/Medio/Mín)", **TKW)
    _ax(ax, "EUR/MWh"); _xt(ax, lbl); ax.legend(**GKW)

    ax = axes2[1, 1]
    if "energia_vendida [MWh]" in det.columns:
        ax.bar(x - 0.2, det["energia_vendida [MWh]"],  width=0.4, color=CS,  alpha=0.8, label="Vendida")
        ax.bar(x + 0.2, det["energia_comprada [MWh]"], width=0.4, color=CCH, alpha=0.8, label="Comprada")
    ax.set_title("Energía Spot Vendida vs Comprada Media Mensual", **TKW)
    _ax(ax, "MWh/dia"); _xt(ax, lbl); ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MODO SIM
# =============================================================================

def modo_sim(sim, titulo_rango):
    x   = np.arange(len(sim))
    lbl = sim["mes_label"].tolist()

    prev_total = sim["prev_total [€]"].sum()
    p50_total  = sim["p50_total [€]"].sum()
    coste_inc  = sim["coste_inc [€]"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.canvas.manager.set_window_title(f"Sim — {titulo_rango}")
    fig.suptitle(
        f"SIMULACION — {titulo_rango}    "
        f"[Previsto: {prev_total:+.2f}€  |  P50: {p50_total:+.2f}€  |  "
        f"Coste incertidumbre: {coste_inc:+.2f}€ ({coste_inc/max(abs(prev_total),1)*100:.1f}%)]",
        fontsize=11, fontweight="bold", y=0.97
    )

    # Previsto vs P50 por mes
    ax = axes[0, 0]
    ax.plot(x, sim["prev_total [€]"], color=CACC, lw=2, marker="o", ms=6,
            label="Previsto total")
    ax.plot(x, sim["p50_total [€]"],  color=CS,   lw=2, marker="s", ms=6,
            label="P50 real")
    ax.fill_between(x, sim["p10_total [€]"], sim["p90_total [€]"],
                    alpha=0.12, color=CS, label="Rango P10-P90")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Beneficio Total Mensual — Previsto vs P50", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl); ax.legend(**GKW)

    # Percentiles P10/P50/P90 medio diario por mes
    ax = axes[0, 1]
    ax.plot(x, sim["p50_medio [€/dia]"],  color=CS,  lw=2, marker="o", ms=5, label="P50/día")
    ax.plot(x, sim["p10_medio [€/dia]"],  color=CF,  lw=1.5, marker="s", ms=5, label="P10/día")
    ax.plot(x, sim["var95_medio [€/dia]"], color=CB, lw=2, marker="^", ms=5, label="P5/día")
    ax.fill_between(x, sim["var95_medio [€/dia]"], sim["p10_medio [€/dia]"],
                    alpha=0.15, color=CB, label="Zona P5-P10")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Percentiles de Riesgo Medio Diario por Mes", **TKW)
    _ax(ax, "EUR/dia"); _xt(ax, lbl); ax.legend(**GKW)

    # Coste de incertidumbre mensual
    ax = axes[1, 0]
    colores_ci = [CB if v > 0 else CS for v in sim["coste_inc [€]"]]
    ax.bar(x, sim["coste_inc [€]"], color=colores_ci, alpha=0.85, width=0.6)
    ax2r = ax.twinx()
    pct = sim["coste_inc [€]"] / sim["prev_total [€]"].abs().clip(lower=1) * 100
    ax2r.plot(x, pct, color=CF, lw=2, marker="o", ms=5, label="Coste inc. %")
    ax2r.set_ylabel("Coste incertidumbre (%)", fontsize=8, color=CF)
    ax2r.tick_params(colors=CF, labelsize=8)
    ax2r.spines[["top"]].set_visible(False)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, **GKW)
    ax.set_title("Coste de Incertidumbre Mensual (EUR y %)", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl)

    # Volatilidad y % negativos por mes
    ax = axes[1, 1]
    ax2b = ax.twinx()
    ax.bar(x, sim["pct_neg_medio [%]"], color=CB, alpha=0.6, width=0.6,
           label="% sim. negativas")
    ax2b.plot(x, sim["std_medio [€/dia]"], color=CF, lw=2, marker="o", ms=5,
              label="Desv. estándar media")
    _ax(ax, "% escenarios negativos")
    ax2b.set_ylabel("Desv. estándar (EUR/día)", fontsize=8, color=CF)
    ax2b.tick_params(colors=CF, labelsize=8)
    ax2b.spines[["top"]].set_visible(False)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, **GKW)
    ax.set_title("% Simulaciones Negativas y Volatilidad por Mes", **TKW)
    _xt(ax, lbl)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MODO COMP
# =============================================================================

def modo_comp(det, sim, titulo_rango):

    # Alinear por mes
    det = det.set_index("mes")
    sim = sim.set_index("mes")
    meses_comunes = det.index.intersection(sim.index)
    det = det.loc[meses_comunes].reset_index()
    sim = sim.loc[meses_comunes].reset_index()

    x   = np.arange(len(det))
    lbl = sim["mes_label"].tolist()

    tot_det = det["beneficio_total [€]"].sum()
    tot_p50 = sim["p50_total [€]"].sum()
    tot_inc = sim["coste_inc [€]"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(f"Comparación Det vs Sim | {titulo_rango}")
    fig.suptitle(
        f"COMPARACION DETERMINISTA VS SIMULACION — {titulo_rango}    "
        f"[Det total: {tot_det:+.2f}€  |  Sim P50: {tot_p50:+.2f}€  |  "
        f"Coste incertidumbre: {tot_inc:+.2f}€ ({tot_inc/max(abs(tot_det),1)*100:.1f}%)]",
        fontsize=11, fontweight="bold", y=0.97
    )

    # Beneficio total por mes — det vs sim P50
    ax = axes[0, 0]
    ancho = 0.35
    ax.bar(x - ancho/2, det["beneficio_total [€]"], width=ancho,
           color=CACC, alpha=0.85, label="Determinista")
    ax.bar(x + ancho/2, sim["p50_total [€]"],       width=ancho,
           color=CS, alpha=0.85, label="Simulación P50")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Beneficio Total por Mes — Det vs Sim P50", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl); ax.legend(**GKW)

    # Beneficio acumulado comparativo
    ax = axes[0, 1]
    acum_det = det["beneficio_total [€]"].cumsum().values
    acum_p50 = sim["p50_total [€]"].cumsum().values
    acum_p10 = sim["p10_total [€]"].cumsum().values
    acum_p90 = sim["p90_total [€]"].cumsum().values
    ax.fill_between(x, acum_p10, acum_p90, alpha=0.12, color=CS, label="Rango P10-P90 sim")
    ax.plot(x, acum_det, color=CACC, lw=2.5, marker="o", ms=6, ls="--",
            label="Det acumulado")
    ax.plot(x, acum_p50, color=CS,   lw=2.5, marker="s", ms=6,
            label="Sim P50 acumulado")
    ax.plot(x, acum_p10, color=CB,   lw=1.5, marker="^", ms=4, alpha=0.7,
            label="Sim P10 acumulado")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("Beneficio Acumulado — Det vs Simulación", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl); ax.legend(**GKW)

    # Coste de incertidumbre mes a mes
    ax = axes[1, 0]
    colores_ci = [CB if v > 0 else CS for v in sim["coste_inc [€]"]]
    ax.bar(x, sim["coste_inc [€]"], color=colores_ci, alpha=0.85, width=0.6)
    ax2r = ax.twinx()
    pct = sim["coste_inc [€]"] / sim["prev_total [€]"].abs().clip(lower=1) * 100
    ax2r.plot(x, pct, color=CF, lw=2, marker="o", ms=6, label="Coste inc. %")
    ax2r.set_ylabel("Coste incertidumbre (%)", fontsize=8, color=CF)
    ax2r.tick_params(colors=CF, labelsize=8)
    ax2r.spines[["top"]].set_visible(False)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.axhline(sim["coste_inc [€]"].mean(), color=CACC, lw=1.2, ls=":",
               label=f"Media: {sim['coste_inc [€]'].mean():+.2f}€")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, **GKW)
    ax.set_title("Coste de Incertidumbre Mensual", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl)

    # P10/P50/P90 vs Previsto — totales mensuales
    ax = axes[1, 1]
    ax.plot(x, sim["prev_total [€]"],  color=CACC, lw=2, marker="o", ms=6,
            ls="--", label="Previsto total")
    ax.plot(x, sim["p50_total [€]"],   color=CS,   lw=2, marker="s", ms=6,
            label="P50 total")
    ax.fill_between(x, sim["p10_total [€]"], sim["p90_total [€]"],
                    alpha=0.12, color=CS, label="Rango P10-P90")
    ax.plot(x, sim["p10_total [€]"],   color=CB,   lw=1.5, marker="^", ms=4,
            alpha=0.8, label="P10 total")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_title("P10 / P50 / P90 vs Previsto por Mes", **TKW)
    _ax(ax, "EUR"); _xt(ax, lbl); ax.legend(**GKW)

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

    print(f"\n  Modo    : {modo.upper()}")
    print(f"  Periodo : {titulo_rango}  ({len(meses)} mes{'es' if len(meses)>1 else ''})\n")

    if modo in ("det", "comp"):
        dfs_det = []
        for m, a in meses:
            df = cargar_resumen_det(m, a)
            if df is not None:
                dfs_det.append(df)
            else:
                print(f"  [!] Sin datos det: {NOMBRES_MES[m]} {a}")
        det_men = agregar_mensual_det(
            [cargar_resumen_det(m, a) for m, a in meses if cargar_resumen_det(m, a) is not None],
            [(m, a) for m, a in meses if cargar_resumen_det(m, a) is not None]
        ) if dfs_det else None

    if modo in ("sim", "comp"):
        dfs_sim = []
        for m, a in meses:
            df = cargar_resumen_sim(m, a)
            if df is not None:
                dfs_sim.append(df)
            else:
                print(f"  [!] Sin datos sim: {NOMBRES_MES[m]} {a}")
        sim_men = agregar_mensual_sim(
            [cargar_resumen_sim(m, a) for m, a in meses if cargar_resumen_sim(m, a) is not None],
            [(m, a) for m, a in meses if cargar_resumen_sim(m, a) is not None]
        ) if dfs_sim else None

    if modo == "det":
        if det_men is not None and not det_men.empty:
            modo_det(det_men, titulo_rango)
        else:
            print("  [!] No hay datos deterministas en el rango.")

    elif modo == "sim":
        if sim_men is not None and not sim_men.empty:
            modo_sim(sim_men, titulo_rango)
        else:
            print("  [!] No hay datos de simulación en el rango.")

    elif modo == "comp":
        if det_men is not None and sim_men is not None:
            modo_comp(det_men, sim_men, titulo_rango)
        else:
            print("  [!] Faltan datos para la comparación.")