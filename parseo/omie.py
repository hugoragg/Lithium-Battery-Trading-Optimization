"""
Parseo de archivos OMIE — Extracción de precios cuartohorarios
Autor: Hugo Raggini Paternain

Lee todos los Excel OMIE de una carpeta con formato:
    INT_PBC_EV_H_1_DD_MM_YYYY_DD_MM_YYYY.xls
Extrae la fila "Precio marginal en el sistema español (EUR/MWh)"
y genera un CSV maestro en: datos/precios/precios_<mes>_<año>.csv

Uso:
    python -m parseo.omie              # pide mes y año interactivamente
    python -m parseo.omie 1 2026       # enero 2026
    python -m parseo.omie 6 2026       # junio 2026
"""

import sys
import pandas as pd
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

FILA_OBJETIVO  = "Precio marginal en el sistema español (EUR/MWh)"
COLUMNAS_Q     = [f"H{h}Q{q}" for h in range(1, 25) for q in range(1, 5)]
CARPETA_PRECIOS = ROOT / "datos" / "precios"

# --- Mes y año: por argumento o interactivo ---
if len(sys.argv) == 3:
    MES  = int(sys.argv[1])
    ANIO = int(sys.argv[2])
else:
    MES  = int(input("Mes (número 1-12): "))
    ANIO = int(input("Año (ej. 2026): "))

CARPETA_DATOS = ROOT / "datos" / "omie_xls" / f"{NOMBRES_MES[MES]} {ANIO}"
CSV_SALIDA    = CARPETA_PRECIOS / f"precios_{NOMBRES_MES[MES].lower()}_{ANIO}.csv"

# =============================================================================
# FUNCIONES
# =============================================================================

def extraer_fecha_nombre(nombre_archivo):
    """Extrae fecha ISO del nombre: INT_PBC_EV_H_1_DD_MM_YYYY_..."""
    partes = Path(nombre_archivo).stem.split("_")
    dia, mes, anio = partes[5], partes[6], partes[7]
    return f"{anio}-{mes}-{dia}"


# Semilla fija para reproducibilidad del muestreo intra-hora
import numpy as np
_RNG_PARSEO = np.random.default_rng(seed=2026)
SIGMA_INTRA = 0.02   # variabilidad intra-hora (~2%)


def expandir_hora_a_cuartos(precio_hora: float) -> list:
    """
    Expande un precio horario a 4 precios cuartohorarios con ruido leve.

    Método: genera 3 factores lognormales aleatorios y calcula el 4º
    de forma que la media de los 4 cuartos sea exactamente el precio horario.
    Así se conserva la energía total y se añade variabilidad realista.

    sigma_intra = 2% → variación típica de ±2€ en un precio de 100€/MWh.
    Los precios negativos conservan el signo correctamente.
    """
    # Ruido lognormal para Q1, Q2, Q3
    epsilons = _RNG_PARSEO.standard_normal(3)
    factores = np.exp(SIGMA_INTRA * epsilons)

    # Aplicar ruido preservando el signo
    signo = np.sign(precio_hora) if precio_hora != 0 else 1.0
    abs_p = abs(precio_hora)

    q1 = signo * abs_p * factores[0]
    q2 = signo * abs_p * factores[1]
    q3 = signo * abs_p * factores[2]

    # Q4 se calcula para que la media sea exactamente precio_hora
    q4 = 4 * precio_hora - q1 - q2 - q3

    return [round(q1, 4), round(q2, 4), round(q3, 4), round(q4, 4)]


def parsear_excel(ruta):
    """Devuelve lista de 96 precios del sistema español."""
    df = pd.read_excel(ruta, header=None, sheet_name=0)

    mascara = df.iloc[:, 0].astype(str).str.strip() == FILA_OBJETIVO
    filas   = df[mascara]

    if filas.empty:
        raise ValueError(f"Fila objetivo no encontrada en {ruta.name}")

    valores = pd.to_numeric(filas.iloc[0, 1:].values, errors="coerce")
    valores = valores[~pd.isna(valores)]

    if len(valores) == 96:
        # Formato cuartohorario nativo — sin modificar
        return valores.tolist()
    elif len(valores) == 24:
        # Formato horario — expandir con interpolacion estocastica conservando media
        cuartos = []
        for precio_h in valores:
            cuartos.extend(expandir_hora_a_cuartos(float(precio_h)))
        return cuartos
    else:
        raise ValueError(f"Formato no reconocido: {len(valores)} valores (se esperaban 24 o 96)")


def parsear_carpeta(carpeta):
    registros = []
    archivos  = sorted(Path(carpeta).glob("INT_PBC_EV_H_1_*.xls"))

    if not archivos:
        print(f"[!] No se encontraron archivos .xls en '{carpeta}'")
        return pd.DataFrame()

    print(f"Archivos encontrados: {len(archivos)}\n")

    for ruta in archivos:
        try:
            fecha   = extraer_fecha_nombre(ruta.name)
            precios = parsear_excel(ruta)
            registros.append({"fecha": fecha, **dict(zip(COLUMNAS_Q, precios))})
            n_horas_unicas = len(set(round(precios[i*4], 1) for i in range(24)))
            tag = "(horario+ruido2%)" if n_horas_unicas <= 24 else ""
            print(f"  v  {ruta.name}  ->  {fecha}  {tag}")
        except Exception as e:
            print(f"  x  {ruta.name}  ->  ERROR: {e}")

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros).sort_values("fecha").reset_index(drop=True)
    return df

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"=== Parseo OMIE — {NOMBRES_MES[MES]} {ANIO} ===")
    print(f"    Carpeta datos : {CARPETA_DATOS}")
    print(f"    Salida        : {CSV_SALIDA}\n")

    df_precios = parsear_carpeta(CARPETA_DATOS)

    if df_precios.empty:
        print("\n[!] No se generó ningún dato.")
    else:
        CARPETA_PRECIOS.mkdir(parents=True, exist_ok=True)
        df_precios.to_csv(CSV_SALIDA, index=False)
        print(f"\n  CSV guardado : '{CSV_SALIDA}'")
        print(f"  Días procesados : {len(df_precios)}")
        print(f"  Rango de fechas : {df_precios['fecha'].min()} -> {df_precios['fecha'].max()}")