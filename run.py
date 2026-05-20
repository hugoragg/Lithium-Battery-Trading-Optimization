"""
Orquestador del pipeline completo
Autor: Hugo Raggini Paternain

Para cada mes detecta el estado actual y ejecuta lo que falte:
    parseo (xls -> csv)  ->  optimizacion (csv -> resultados/optimizacion)
    -> simulacion Monte Carlo (-> resultados/simulacion)

Pasos ya realizados se omiten (existencia de los CSV de resumen).

Uso:
    python run.py                 # todos los meses con datos en datos/omie_xls/
    python run.py 1 2026          # un solo mes
    python run.py 1 2026 --force  # rehace todos los pasos aunque existan
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Fuerza UTF-8 en los subprocesos para que prints con tildes / flechas no
# casquen bajo el codec cp1252 por defecto de Windows cuando stdout esta
# capturado/redirigido.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

NOMBRES_MES = {
    1: "Enero",      2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",       6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre",  12: "Diciembre",
}
NUM_MES = {v: k for k, v in NOMBRES_MES.items()}


def meses_disponibles() -> list[tuple[int, int]]:
    """Devuelve [(mes, anio), ...] inferidos de las subcarpetas datos/omie_xls/."""
    carpeta = ROOT / "datos" / "omie_xls"
    meses = []
    if not carpeta.exists():
        return meses
    for sub in carpeta.iterdir():
        if not sub.is_dir():
            continue
        partes = sub.name.split()
        if len(partes) == 2 and partes[0] in NUM_MES and partes[1].isdigit():
            meses.append((NUM_MES[partes[0]], int(partes[1])))
    return sorted(meses, key=lambda x: (x[1], x[0]))


def precios_csv(mes: int, anio: int) -> Path:
    return ROOT / "datos" / "precios" / f"precios_{NOMBRES_MES[mes].lower()}_{anio}.csv"


def resumen_det(mes: int, anio: int) -> Path:
    return ROOT / "resultados" / "optimizacion" / f"{anio}-{mes:02d}" / f"resumen_{anio}_{mes:02d}.csv"


def resumen_sim(mes: int, anio: int) -> Path:
    return ROOT / "resultados" / "simulacion" / f"{anio}-{mes:02d}" / f"resumen_sim_{anio}_{mes:02d}.csv"


def ejecutar(modulo: str, *args: str) -> None:
    cmd = [sys.executable, "-m", modulo, *args]
    print(f"  >>> {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, cwd=ROOT, env=ENV)
    if res.returncode != 0:
        print(f"  [!] {modulo} fallo (codigo {res.returncode})")
        sys.exit(res.returncode)


def pipeline_mes(mes: int, anio: int, force: bool) -> None:
    print(f"\n{'='*65}\n  {NOMBRES_MES[mes]} {anio}\n{'='*65}")

    if force or not precios_csv(mes, anio).exists():
        ejecutar("parseo.omie", str(mes), str(anio))
    else:
        print(f"  [skip] {precios_csv(mes, anio).name} ya existe")

    if force or not resumen_det(mes, anio).exists():
        ejecutar("optimizacion.mes", str(mes), str(anio))
    else:
        print(f"  [skip] resumen optimizacion ya existe")

    if force or not resumen_sim(mes, anio).exists():
        ejecutar("simulacion.mes", str(mes), str(anio))
    else:
        print(f"  [skip] resumen simulacion ya existe")


def main() -> None:
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if len(args) == 2 and args[0].isdigit() and args[1].isdigit():
        meses = [(int(args[0]), int(args[1]))]
    elif len(args) == 0:
        meses = meses_disponibles()
        if not meses:
            print("[!] No hay datos en datos/omie_xls/. Anade carpetas tipo 'Enero 2026'.")
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(1)

    print(f"Pipeline para {len(meses)} mes(es): "
          + ", ".join(f"{NOMBRES_MES[m][:3]} {a}" for m, a in meses)
          + (" [FORCE]" if force else ""))

    for m, a in meses:
        pipeline_mes(m, a, force)

    print(f"\n{'='*65}\n  Pipeline completado\n{'='*65}")
    print("\nPara ver dashboards:")
    print("  python -m visualizacion.mes <MES> <ANIO>          # determinista")
    print("  python -m visualizacion.sim_mes <MES> <ANIO>      # simulacion")
    print("  python -m visualizacion.anual                     # rango de meses")


if __name__ == "__main__":
    main()
