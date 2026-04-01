import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread
from numba import cuda, float32
import math
import time
import os

# =====================================================================
# 1. GPU DEVICE FUNCTION (PBC Estática)
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
# 2. CONFIGURACIÓN Y PRE-PROCESAMIENTO
# =====================================================================
ruta = "evolucion_bicapa.xyz" # Asegúrate de que la ruta sea correcta

r_max = 6.0
n_bins = 200
bin_width = r_max / n_bins

print(f"--- Iniciando RDF GPU (Celda Estática - RTX 3050) ---")

# --- LEER PRIMER FRAME PARA EXTRAER CELDA ---
atoms_stream = iread(ruta)
first_frame = next(atoms_stream)
n_at = len(first_frame)
vol = first_frame.get_volume()

# Preparar Celda e Inversa una sola vez (en float32 para velocidad)
cell_data = np.array(first_frame.get_cell()).astype(np.float32)
inv_cell_data = np.linalg.inv(cell_data).astype(np.float32)

# Subir celda a la GPU permanentemente
d_cell = cuda.to_device(cell_data)
d_inv_cell = cuda.to_device(inv_cell_data)

# Reservar memoria para posiciones y conteos
d_pos = cuda.device_array((n_at, 3), dtype=np.float32)
d_counts_total = cuda.to_device(np.zeros(n_bins, dtype=np.int64))

# Configuración del Grid (2D como el original)
threads = (16, 16)
blocks = (math.ceil(n_at/16), math.ceil(n_at/16))

# =====================================================================
# 3. BUCLE DE PROCESAMIENTO
# =====================================================================
t_start = time.perf_counter()
frame_count = 0

# Procesar el primer frame ya leído
pos_first = first_frame.get_positions().astype(np.float32)
d_pos.copy_to_device(pos_first)
rdf_kernel_ultra[blocks, threads](d_pos, d_cell, d_inv_cell, r_max, bin_width, n_bins, d_counts_total)
frame_count += 1

# Procesar el resto de los frames
for atoms in atoms_stream:
    # Solo copiamos posiciones, la celda ya está en la GPU
    pos_data = atoms.get_positions().astype(np.float32)
    d_pos.copy_to_device(pos_data)

    rdf_kernel_ultra[blocks, threads](
        d_pos, d_cell, d_inv_cell, r_max, bin_width, n_bins, d_counts_total
    )
    
    frame_count += 1
    if frame_count % 500 == 0:
        print(f"Progreso: {frame_count} frames...")

counts_final = d_counts_total.copy_to_host()
t_end = time.perf_counter() - t_start

# =====================================================================
# 4. NORMALIZACIÓN Y SALIDA
# =====================================================================
norm_factor = n_at * (n_at / vol)
bin_edges = np.linspace(0, r_max, n_bins + 1)
r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
vol_capas = 4 * np.pi * r_centers**2 * bin_width

rdf_final = counts_final / (norm_factor * vol_capas * frame_count)

print(f"\n✅ Análisis completado en {t_end:.2f} segundos.")
print(f"Frames totales: {frame_count}")

plt.figure(figsize=(9, 5))
plt.plot(r_centers, rdf_final, color='navy', lw=1.8, label='RDF Promedio (Celda Fija)')
plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Distancia r (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.show()
