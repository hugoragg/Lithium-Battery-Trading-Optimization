"""
Parametros del sistema - Fuente unica de verdad
Autor: Hugo Raggini Paternain

Importado por:
    optimizacion.bateria
    simulacion.dia
    visualizacion.mes
    visualizacion.sim_dia

Cualquier cambio aqui se propaga al modelo, al simulador y a los dashboards.
"""

# =============================================================================
# BATERIA
# =============================================================================

E_MAX     = 2.0           # Capacidad nominal (MWh)
P_CH_MAX  = 0.5           # Potencia maxima de carga    (MW)  -> 4h
P_DIS_MAX = 0.5           # Potencia maxima de descarga (MW)
ETA       = 0.9381        # Eficiencia (round-trip sqrt)
DOD       = 0.80          # Depth of Discharge
SOC_MIN   = E_MAX * (1 - DOD)
SOC_MAX   = E_MAX
SOC_INIT  = E_MAX * 0.5
C_DEG     = 6.63          # Coste de degradacion (EUR/MWh operado)

# =============================================================================
# MERCADO
# =============================================================================

DELTA = 0.25              # Paso temporal (h) - cuartohorario
M_BIG = 10000.0           # Big-M para restricciones binarias del MILP

# =============================================================================
# RESERVA SECUNDARIA (aFRR / REE)
# =============================================================================

ALPHA_UP    = 0.236       # Tasa esperada de activacion al alza
ALPHA_DOWN  = 0.222       # Tasa esperada de activacion a la baja

PI_DISP_UP   = 9.27       # Pago por disponibilidad al alza   (EUR/MWh)
PI_DISP_DOWN = 8.48       # Pago por disponibilidad a la baja (EUR/MWh)
PI_ACT_UP    = 118.15     # Pago por activacion al alza       (EUR/MWh)
PI_ACT_DOWN  = 53.05      # Pago por activacion a la baja     (EUR/MWh)
