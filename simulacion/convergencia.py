"""
Análisis de Convergencia Monte Carlo
Autor: Hugo Raggini Paternain
-------------------------------
Determina cuántas simulaciones (N) hacen falta para que los estimadores
P50 (mediana) y P5 (peor 5%) se estabilicen.

Dos modos:
    python -m simulacion.convergencia              # GLOBAL: todos los dias
    python -m simulacion.convergencia 2026-01-01   # UN dia concreto

Criterio de convergencia: el error estandar relativo del estimador (std entre
N_REPETICIONES repeticiones / P50 de referencia) cae por debajo de un umbral de
forma SOSTENIDA (a partir de ese N y para todos los N mayores). El P5 (cola) usa
un umbral mas relajado por ser intrinsecamente mas ruidoso.

Se calcula un N recomendado SEPARADO para cada metrica (P50 y P5). En el modo
global, el N universal de cada metrica es el MAXIMO del N recomendado entre
todos los dias (peor caso): si basta para el dia mas exigente, basta para todos.
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

# Umbrales del error estandar relativo (std / P50 de referencia)
UMBRAL_P50 = 0.05   # 5%  estimador central (estricto)
UMBRAL_P5  = 0.15   # 15% cola, peor 5% (relajado: mas ruidosa por construccion)

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


def _ax(ax, ylabel="", xlabel="N simulaciones"):
    ax.set_facecolor(PBG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555", labelsize=8)
    ax.grid(axis="y", alpha=0.2, lw=0.5, color="#AAAAAA")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#555")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#555")


# =============================================================================
# NÚCLEO DE CÁLCULO
# =============================================================================

def crear_solver():
    opt = pyo.SolverFactory("highs")
    opt.options.update({"time_limit": 120, "mip_rel_gap": 0.001,
                        "output_flag": 0, "log_to_console": 0})
    return opt


def cargar_precios_dia(fecha_str):
    """Devuelve el array de 96 precios de un dia, o None si no hay datos."""
    fecha_dt = pd.to_datetime(fecha_str)
    csv = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[fecha_dt.month].lower()}_{fecha_dt.year}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fila = df[df["fecha"] == fecha_str]
    if fila.empty:
        return None
    return fila[COLUMNAS_Q].values[0].astype(float)


def schedule_de_precios(precios, opt):
    """Optimiza el dia una vez y devuelve el schedule comprometido."""
    model = construir_modelo(precios)
    opt.solve(model, tee=False)
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


def tabla_convergencia(schedule):
    """Para cada N, mide P50/P5 medios y su std entre N_REPETICIONES repeticiones."""
    resultados = []
    for n in N_VALORES:
        p50_reps, p5_reps = [], []
        for rep in range(N_REPETICIONES):
            rng  = np.random.default_rng(SEED_BASE + rep * 1000)
            bens = np.array([
                simular_ejecucion(
                    schedule,
                    generar_escenario_ejecucion(schedule, rng, **PARAMS_SIM)
                )["beneficio_real [€]"]
                for _ in range(n)
            ])
            p50_reps.append(np.percentile(bens, 50))
            p5_reps.append(np.percentile(bens, 5))
        resultados.append({
            "N":         n,
            "P50_medio": np.mean(p50_reps), "P50_std": np.std(p50_reps),
            "P5_medio":  np.mean(p5_reps),  "P5_std":  np.std(p5_reps),
        })
    return pd.DataFrame(resultados)


def _primer_sostenido(rel, umbral):
    """Primer N a partir del cual TODOS los N >= cumplen rel < umbral. None si ninguno."""
    estable = [v < umbral for v in rel]
    for i in range(len(N_VALORES)):
        if all(estable[i:]):
            return int(N_VALORES[i])
    return None


def recomendados_metricas(df_conv):
    """
    Devuelve (n_p50, n_p5, rel_p50, rel_p5):
      · n_p50 / n_p5: N recomendado (sostenido) para cada metrica (None si no converge).
      · rel_p50 / rel_p5: error estandar relativo por N (para imprimir/representar).
    """
    p50_ref = df_conv.loc[df_conv["N"] == max(N_VALORES), "P50_medio"].values[0]
    rel_p50 = (df_conv["P50_std"] / abs(p50_ref)).tolist()
    rel_p5  = (df_conv["P5_std"]  / abs(p50_ref)).tolist()
    return (_primer_sostenido(rel_p50, UMBRAL_P50),
            _primer_sostenido(rel_p5,  UMBRAL_P5),
            rel_p50, rel_p5)


# =============================================================================
# MODO UN DÍA
# =============================================================================

def _marcar(ax, df_conv, col_medio, n_rec):
    """Linea vertical en el N recomendado de esa metrica."""
    if n_rec is None:
        return
    val = df_conv.loc[df_conv["N"] == n_rec, col_medio].values[0]
    col_std = col_medio.replace("medio", "std")
    ax.axvline(n_rec, color=CF, lw=1.5, ls=":", label=f"N recomendado = {n_rec}")
    ax.annotate(f"{val:.1f}€", xy=(n_rec, val),
                xytext=(n_rec + 20, val + df_conv[col_std].mean()),
                fontsize=8, color=CF, fontweight="bold")


def _plot_dia(df_conv, fecha_str, n_p50, n_p5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.canvas.manager.set_window_title(f"Convergencia Monte Carlo — {fecha_str}")
    fig.suptitle(
        f"ANÁLISIS DE CONVERGENCIA MONTE CARLO — {fecha_str}    "
        f"[Repeticiones por N: {N_REPETICIONES}]",
        fontsize=11, fontweight="bold", y=0.97
    )

    ax = axes[0]
    ax.plot(df_conv["N"], df_conv["P50_medio"], color=CS, lw=2.5, marker="o", ms=6, label="P50 medio")
    ax.fill_between(df_conv["N"], df_conv["P50_medio"] - df_conv["P50_std"],
                    df_conv["P50_medio"] + df_conv["P50_std"],
                    alpha=0.2, color=CS, label="±1 std entre repeticiones")
    _marcar(ax, df_conv, "P50_medio", n_p50)
    ax.set_title("① Convergencia del P50", **TKW)
    _ax(ax, "P50 (€)")
    ax.legend(**GKW)

    ax = axes[1]
    ax.plot(df_conv["N"], df_conv["P5_medio"], color=CB, lw=2.5, marker="s", ms=6, label="P5 medio")
    ax.fill_between(df_conv["N"], df_conv["P5_medio"] - df_conv["P5_std"],
                    df_conv["P5_medio"] + df_conv["P5_std"],
                    alpha=0.2, color=CB, label="±1 std entre repeticiones")
    _marcar(ax, df_conv, "P5_medio", n_p5)
    ax.set_title("② Convergencia del P5 (peor 5%)", **TKW)
    _ax(ax, "P5 (€)")
    ax.legend(**GKW)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])


def correr_dia(fecha_str, opt):
    precios = cargar_precios_dia(fecha_str)
    if precios is None:
        print(f"[!] Fecha '{fecha_str}' sin datos en datos/precios/.")
        return

    print(f"\n  Optimizando schedule para {fecha_str}...")
    schedule = schedule_de_precios(precios, opt)
    print(f"  Beneficio previsto: {schedule['beneficio_previsto']:.2f} €\n")

    print(f"  Analizando convergencia con {N_REPETICIONES} repeticiones por N...\n")
    df_conv = tabla_convergencia(schedule)

    print(f"  {'N':>6}  {'P50 medio':>10}  {'P50 std':>8}  {'P5 medio':>12}  {'P5 std':>10}")
    print(f"  {'-'*55}")
    for _, r in df_conv.iterrows():
        print(f"  {int(r['N']):>6}  {r['P50_medio']:>10.2f}€  {r['P50_std']:>7.2f}€  "
              f"{r['P5_medio']:>12.2f}€  {r['P5_std']:>9.2f}€")

    n_p50, n_p5, rel_p50, rel_p5 = recomendados_metricas(df_conv)

    print(f"\n  {'='*60}")
    print(f"  CONCLUSIÓN  (umbral P50 {UMBRAL_P50*100:.0f}%  |  umbral P5 {UMBRAL_P5*100:.0f}%)")
    print(f"  {'='*60}")
    for i, n in enumerate(N_VALORES):
        e50 = "OK" if rel_p50[i] < UMBRAL_P50 else "--"
        e5  = "OK" if rel_p5[i]  < UMBRAL_P5  else "--"
        print(f"  N={n:>4}   P50 e.e.={rel_p50[i]*100:>5.1f}% [{e50}]   "
              f"P5 e.e.={rel_p5[i]*100:>5.1f}% [{e5}]")
    nmax = max(N_VALORES)
    print(f"\n  N recomendado P50: {n_p50 if n_p50 else '>' + str(nmax)}")
    print(f"  N recomendado P5 : {n_p5  if n_p5  else '>' + str(nmax)}")
    print(f"  {'='*60}\n")

    _plot_dia(df_conv, fecha_str, n_p50, n_p5)

    csv_out = ROOT / "resultados" / "simulacion" / "dias_sueltos" / f"convergencia_{fecha_str}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_conv.to_csv(csv_out, index=False)
    print(f"  CSV guardado: {csv_out}")


# =============================================================================
# MODO GLOBAL (N universal sobre todos los días)
# =============================================================================

def _universal(df, col):
    """N universal = max del N recomendado entre dias (None = no converge a N=500)."""
    no_conv  = int(df[col].isna().sum())
    finitos  = df[col].dropna()
    if finitos.empty:
        return None, no_conv, None
    uni        = int(finitos.max())
    peor_fecha = df.loc[finitos.idxmax(), "fecha"]
    return uni, no_conv, peor_fecha


def _plot_global(df, uni_p50, uni_p5):
    nmax = max(N_VALORES)
    etiquetas = [str(n) for n in N_VALORES] + [f">{nmax}"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.canvas.manager.set_window_title("Convergencia Global - N universal")
    fig.suptitle(
        f"CONVERGENCIA GLOBAL — {len(df)} días    "
        f"[N universal P50 = {uni_p50 if uni_p50 else '>' + str(nmax)}  |  "
        f"N universal P5 = {uni_p5 if uni_p5 else '>' + str(nmax)}]",
        fontsize=11, fontweight="bold", y=0.97
    )

    def _hist(ax, col, uni, color, titulo):
        counts = [int((df[col] == n).sum()) for n in N_VALORES] + [int(df[col].isna().sum())]
        x      = np.arange(len(etiquetas))
        cols   = [color] * len(etiquetas)
        if uni in N_VALORES:                       # destacar la barra del N universal
            cols[N_VALORES.index(uni)] = CF
        ax.bar(x, counts, color=cols, alpha=0.85, width=0.7)
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c, str(c), ha="center", va="bottom", fontsize=8, color="#555")
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas)
        ax.set_title(titulo, **TKW)
        _ax(ax, "nº de días", xlabel="N recomendado")

    _hist(axes[0], "N_P50", uni_p50, CS, f"① N recomendado por día — P50 (e.e.<{UMBRAL_P50*100:.0f}%)")
    _hist(axes[1], "N_P5",  uni_p5,  CB, f"② N recomendado por día — P5 (e.e.<{UMBRAL_P5*100:.0f}%)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    png = ROOT / "resultados" / "simulacion" / "convergencia_global.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120)
    print(f"  Figura guardada: {png}")


def correr_global(fechas, opt):
    print(f"\n{'='*60}")
    print(f"  CONVERGENCIA GLOBAL — {len(fechas)} días")
    print(f"  ({N_REPETICIONES} reps x {sum(N_VALORES)} sims por día)")
    print(f"{'='*60}\n")

    filas = []
    for k, fecha in enumerate(fechas, 1):
        precios = cargar_precios_dia(fecha)
        if precios is None:
            continue
        try:
            schedule = schedule_de_precios(precios, opt)
            df_conv  = tabla_convergencia(schedule)
            n_p50, n_p5, _, _ = recomendados_metricas(df_conv)
        except Exception as e:
            print(f"  [{k:>3}/{len(fechas)}] {fecha}  ERROR: {e}")
            continue
        filas.append({"fecha": fecha, "N_P50": n_p50, "N_P5": n_p5})
        s50 = n_p50 if n_p50 else ">" + str(max(N_VALORES))
        s5  = n_p5  if n_p5  else ">" + str(max(N_VALORES))
        print(f"  [{k:>3}/{len(fechas)}] {fecha}   N_P50={s50:<5}  N_P5={s5}")

    df = pd.DataFrame(filas)
    if df.empty:
        print("\n[!] No se pudo procesar ningún día.")
        return

    uni_p50, noconv_p50, peor_p50 = _universal(df, "N_P50")
    uni_p5,  noconv_p5,  peor_p5  = _universal(df, "N_P5")

    nmax = max(N_VALORES)
    print(f"\n  {'='*60}")
    print(f"  CONVERGENCIA GLOBAL  ({len(df)} días  |  P50<{UMBRAL_P50*100:.0f}%  P5<{UMBRAL_P5*100:.0f}%)")
    print(f"  {'='*60}")
    print(f"  Distribución del N recomendado por día:")
    for n in N_VALORES:
        c50 = int((df["N_P50"] == n).sum())
        c5  = int((df["N_P5"]  == n).sum())
        print(f"    N={n:>4}    P50: {c50:>3} días    P5: {c5:>3} días")
    print(f"    no conv   P50: {noconv_p50:>3} días    P5: {noconv_p5:>3} días")
    print()
    print(f"  >> N UNIVERSAL P50 = max sobre días = {uni_p50 if uni_p50 else '>' + str(nmax)}"
          f"   (peor día: {peor_p50})")
    print(f"  >> N UNIVERSAL P5  = max sobre días = {uni_p5 if uni_p5 else '>' + str(nmax)}"
          f"   (peor día: {peor_p5})")
    if noconv_p50 or noconv_p5:
        print(f"  (aviso: hay días que no convergen ni con N={nmax})")
    print(f"  {'='*60}\n")

    _plot_global(df, uni_p50, uni_p5)

    csv_out = ROOT / "resultados" / "simulacion" / "convergencia_global.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"  CSV guardado: {csv_out}")


def todas_las_fechas():
    """Lista ordenada de todas las fechas con datos en datos/precios/."""
    fechas = []
    for csv in sorted(CARPETA_PRECIOS.glob("precios_*.csv")):
        fechas += pd.read_csv(csv)["fecha"].astype(str).tolist()
    return sorted(set(fechas))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    opt  = crear_solver()
    args = sys.argv[1:]

    if len(args) == 0:
        # GLOBAL: todos los días con datos -> N universal por métrica
        fechas = todas_las_fechas()
        if not fechas:
            print("[!] No hay precios en datos/precios/. Ejecuta antes parseo.omie.")
            sys.exit(1)
        correr_global(fechas, opt)
    else:
        entrada = args[0].strip()
        try:
            fecha_dt  = pd.to_datetime(entrada, dayfirst=("/" in entrada))
            fecha_str = fecha_dt.strftime("%Y-%m-%d")
        except Exception:
            print("[!] Formato no reconocido. Usa YYYY-MM-DD o DD/MM/YYYY.")
            sys.exit(1)
        correr_dia(fecha_str, opt)

    plt.show()
