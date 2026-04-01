import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread
from numba import cuda
import math
import time
import os

# =====================================================================
# 1. GPU DEVICE FUNCTION (Lógica de Distancia y PBC)
# =====================================================================
@cuda.jit(device=True)
def calcular_distancia_pbc(p1, p2, cell, inv_cell):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]

    # Mínima imagen en coordenadas fraccionarias
    f_dx = dx * inv_cell[0, 0] + dy * inv_cell[1, 0] + dz * inv_cell[2, 0]
    f_dy = dx * inv_cell[0, 1] + dy * inv_cell[1, 1] + dz * inv_cell[2, 1]
    f_dz = dx * inv_cell[0, 2] + dy * inv_cell[1, 2] + dz * inv_cell[2, 2]

    f_dx -= math.floor(f_dx + 0.5)
    f_dy -= math.floor(f_dy + 0.5)
    f_dz -= math.floor(f_dz + 0.5)

    # Regreso a coordenadas reales
    rx = f_dx * cell[0, 0] + f_dy * cell[1, 0] + f_dz * cell[2, 0]
    ry = f_dx * cell[0, 1] + f_dy * cell[1, 1] + f_dz * cell[2, 1]
    rz = f_dx * cell[0, 2] + f_dy * cell[1, 2] + f_dz * cell[2, 2]

    return math.sqrt(rx*rx + ry*ry + rz*rz)

# =====================================================================
# 2. KERNEL GLOBAL (Optimizado para Zero-Allocation)
# =====================================================================
@cuda.jit
def rdf_kernel_ultra(posiciones, cell, inv_cell, r_max, bin_width, n_bins, counts_acumulados):
    i, j = cuda.grid(2)
    n = posiciones.shape[0]

    if i < n and j < n and i < j:
        dist = calcular_distancia_pbc(posiciones[i], posiciones[j], cell, inv_cell)
        if dist < r_max:
            bin_idx = int(dist / bin_width)
            if bin_idx < n_bins:
                cuda.atomic.add(counts_acumulados, bin_idx, 2)

# =====================================================================
# 3. EJECUCIÓN PRINCIPAL
# =====================================================================
ruta = "grande.extxyz"
if not os.path.exists(ruta):
    ruta = r"C:\Users\ProBook\Downloads\ElladBTadmorDataBasis.xyz"

r_max = 6.0
n_bins = 200
bin_width = r_max / n_bins

print(f"--- Iniciando RDF Adaptativa GPU (RTX 3050) ---")
t_start = time.perf_counter()

# Inicialización de memoria en GPU
d_counts_total = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
d_pos = None
d_cell = cuda.device_array((3, 3), dtype=np.float64)
d_inv_cell = cuda.device_array((3, 3), dtype=np.float64)

current_n_at = 0
frame_count = 0
norm_factor_acumulado = 0

for atoms in iread(ruta):
    n_at = len(atoms)
    
    # GESTIÓN DINÁMICA DE MEMORIA (Corrige el error de incompatible shape)
    if n_at != current_n_at:
        d_pos = cuda.device_array((n_at, 3), dtype=np.float64)
        current_n_at = n_at
        # Recalcular grid de hilos
        threads = (16, 16)
        blocks = (math.ceil(n_at/16), math.ceil(n_at/16))

    # Transferencia de datos a la GPU (Eficiencia copy_to_device)
    pos_data = atoms.get_positions().astype(np.float64)
    cell_data = np.array(atoms.get_cell()).astype(np.float64)
    inv_cell_data = np.linalg.inv(cell_data).astype(np.float64)

    d_pos.copy_to_device(pos_data)
    d_cell.copy_to_device(cell_data)
    d_inv_cell.copy_to_device(inv_cell_data)
    
    # Factor de normalización para este frame: N * densidad
    vol = atoms.get_volume()
    norm_factor_acumulado += (n_at * (n_at / vol))

    # Lanzamiento del Kernel
    rdf_kernel_ultra[blocks, threads](
        d_pos, d_cell, d_inv_cell, r_max, bin_width, n_bins, d_counts_total
    )
    
    frame_count += 1
    if frame_count % 500 == 0:
        print(f"Progreso: {frame_count} frames procesados...")

# Descarga de resultados finales
counts_final = d_counts_total.copy_to_host()
t_end = time.perf_counter() - t_start

# =====================================================================
# 4. NORMALIZACIÓN Y SALIDA
# =====================================================================
bin_edges = np.linspace(0, r_max, n_bins + 1)
r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
vol_capas = 4 * np.pi * r_centers**2 * bin_width

# g(r) = Sum(counts) / (Sum(N*rho) * Vol_capas)
rdf_final = counts_final / ( (norm_factor_acumulado / frame_count) * vol_capas * frame_count )

print(f"\n✅ Análisis completado en {t_end:.2f} segundos.")
print(f"Frames totales: {frame_count}")

plt.figure(figsize=(9, 5))
plt.plot(r_centers, rdf_final, color='navy', lw=1.8, label='RDF Promedio GPU')
plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
plt.title("Función de Distribución Radial (Optimización CUDA)")
plt.xlabel("Distancia r (Å)")
plt.ylabel("g(r)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
