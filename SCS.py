import os
import sys
import time
import numpy as np
import cvxpy as cp
import qutip as qt
import scipy.linalg as la
import scipy.sparse as sp

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

k = 1

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
BASE_PATH = "molecular_model/"  
OUTPUT_PATH = f"molecular_model/results_sdp/{k}"

OS_KS_PATH = os.path.join(OUTPUT_PATH, "Ks")

os.makedirs(OUTPUT_PATH, exist_ok=True)
for path in [OS_KS_PATH]:
    os.makedirs(path, exist_ok=True)

# Parámetros físicos del sistema
N_c = 5
S = 0.5 
N_Q = int(2*S+1)

I_c = qt.qeye(N_c)
I_Q = qt.qeye(N_Q)
a = qt.destroy(N_c)
a_q = qt.tensor(a, I_c)
a_d_q = a_q.dag()
a_p = qt.tensor(I_c, a)
a_d_p = a_p.dag()

Q = 1/np.sqrt(2)*(a_q+a_d_q)
P = 1/np.sqrt(2)*(a_p+a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)

LAMBDA_REG = 1e-4

# ==============================================================================
# FUNCIONES AUXILIARES Y GENERACIÓN DE BASES
# ==============================================================================
H = np.load(BASE_PATH + f"hamiltonians/hamiltonian_k_{k}_2_{N_c}.npy")
rho0 = np.array((qt.qload(BASE_PATH + f"HCEs_no_diag/k_{k}.0/HCE_1.0000")).full())
rho0 = rho0 / np.trace(rho0)

with open(OUTPUT_PATH+f"/datos_WS_{N_c}_1.csv", "w") as f:
    f.write("orden,tiempo,dist\n")


def basis_creator(max_g):
    print(f"Generando base para orden máximo {max_g}...")
    basis_pi = []
    monomios_fase_total = [] 
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            
        monomios_fase_total.extend(monomios_orden)
        
        # Base Pi
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio * Pi_Q))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(qt.identity(2), qt.identity(2), monomio * Pi_P))

    spin_ops = [
        (qt.sigmax(), qt.identity(2)), (qt.sigmay(), qt.identity(2)), (qt.sigmaz(), qt.identity(2)),
        (qt.identity(2), qt.sigmax()), (qt.identity(2), qt.sigmay()), (qt.identity(2), qt.sigmaz()),
        (qt.sigmax(), qt.sigmax()), (qt.sigmax(), qt.sigmay()), (qt.sigmax(), qt.sigmaz()),
        (qt.sigmay(), qt.sigmax()), (qt.sigmay(), qt.sigmay()), (qt.sigmay(), qt.sigmaz()),
        (qt.sigmaz(), qt.sigmax()), (qt.sigmaz(), qt.sigmay()), (qt.sigmaz(), qt.sigmaz())
    ]

    basis_spin_global = []
    for op1, op2 in spin_ops:
        for monomio in monomios_fase_total:
            basis_spin_global.append(qt.tensor(op1, op2, monomio))
            
    return basis_pi, basis_spin_global

for max_orden_polinomio in range(4):

    print(f"\n\n\n\nComenzando orden {max_orden_polinomio}\n\n\n\n")

    basis_pi, basis_spin_global = basis_creator(max_orden_polinomio)

    # ==============================================================================
    # PRECÁLCULO DE LOS TÉRMINOS FIJOS
    # ==============================================================================
    def precalcular_tensor(basis_list, rho0_np):
        N = len(basis_list)
        dim = rho0_np.shape[0]
        A_ij = np.zeros((N, N, dim, dim), dtype=np.complex128)
        
        B_mats = [B.full() for B in basis_list]
        B_dags = [B.conj().T for B in B_mats]
        
        for i in range(N):
            for j in range(N):
                jump = B_mats[i] @ rho0_np @ B_dags[j]
                decay = -0.5 * (B_dags[j] @ B_mats[i] @ rho0_np + rho0_np @ B_dags[j] @ B_mats[i])
                A_ij[i, j] = jump + decay
                
        return A_ij

    print(f"\nPrecalculando términos fijos")
    print(f"Tamaño base Pi: {len(basis_pi)}")
    print(f"Tamaño base Global de Espín: {len(basis_spin_global)}")

    A_ij_pi = precalcular_tensor(basis_pi, rho0)
    A_ij_spin = precalcular_tensor(basis_spin_global, rho0)

    # ==============================================================================
    # COMPRESIÓN
    # ==============================================================================
    def precalcular_vectores_dispersos(A_ij_array):
        N = A_ij_array.shape[0]
        dim = A_ij_array.shape[2]
        
        A_flat = np.zeros((N * N, dim * dim), dtype=np.complex128)
        
        k_idx = 0
        for j in range(N):
            for i in range(N):
                A_flat[k_idx, :] = A_ij_array[i, j].flatten(order='F')
                k_idx += 1
                
        # Convertimos a 0 el ruido numérico
        A_flat[np.abs(A_flat) < 1e-5] = 0.0
        A_sparse = sp.csc_matrix(A_flat)
        
        porcentaje = (A_sparse.nnz / (N * N * dim * dim)) * 100
        print(f"Memoria salvada: {A_sparse.nnz} elementos de {N*N*dim*dim} ({porcentaje:.2f}% de densidad)")
        
        return A_sparse

    print("\nComprimiendo tensores a matrices dispersas")
    A_pi_sparse = precalcular_vectores_dispersos(A_ij_pi)
    A_spin_sparse = precalcular_vectores_dispersos(A_ij_spin)

    # ==============================================================================
    # OPTIMIZACIÓN CONVEXA
    # ==============================================================================
    print("\nConfigurando el problema en CVXPY")
    dim = rho0.shape[0]
    dim_sq = dim * dim

    # Variables de optimización (Chi)
    chi_pi = cp.Variable((len(basis_pi), len(basis_pi)), hermitian=True)
    chi_spin = cp.Variable((len(basis_spin_global), len(basis_spin_global)), hermitian=True)

    # Warm Start
    orden_anterior = max_orden_polinomio - 1
    archivo_anterior = OUTPUT_PATH + f"/Ks/chi_raw_g{orden_anterior}_{N_c}.npz"
    usar_warm_start = False

    if os.path.exists(archivo_anterior):
        print(f"Solución previa detectada. Aplicando Warm Start desde grado {orden_anterior}")
        datos_ant = np.load(archivo_anterior)
        chi_pi_ant = datos_ant['chi_pi']
        chi_spin_ant = datos_ant['chi_spin']
        
        val_pi_nuevo = np.zeros((len(basis_pi), len(basis_pi)), dtype=np.complex128)
        val_spin_nuevo = np.zeros((len(basis_spin_global), len(basis_spin_global)), dtype=np.complex128)
        
        dim_pi_ant = chi_pi_ant.shape[0]
        val_pi_nuevo[:dim_pi_ant, :dim_pi_ant] = chi_pi_ant
        
        dim_spin_ant = chi_spin_ant.shape[0]
        val_spin_nuevo[:dim_spin_ant, :dim_spin_ant] = chi_spin_ant
        
        chi_pi.value = val_pi_nuevo
        chi_spin.value = val_spin_nuevo
        usar_warm_start = True
    else:
        print(f"-> No se encontró solución para el grado {orden_anterior}. Iniciando desde cero.")

    # Variables de Holgura
    E_real = cp.Variable(dim_sq)
    E_imag = cp.Variable(dim_sq)

    # Construcción del lindbladiano total
    L_ham = -1j * (H @ rho0 - rho0 @ H)
    L_ham_vec = L_ham.flatten(order='F')

    L_pi_vec = A_pi_sparse.T @ cp.vec(chi_pi, order='F')
    L_spin_vec = A_spin_sparse.T @ cp.vec(chi_spin, order='F')

    L_total_vec = L_ham_vec + L_pi_vec + L_spin_vec

    # Restricciones y función de coste
    restricciones = [
        chi_pi >> 0,
        chi_spin >> 0,
        cp.real(L_total_vec) == E_real,
        cp.imag(L_total_vec) == E_imag
    ]

    error_frob_sq = cp.sum_squares(E_real) + cp.sum_squares(E_imag)
    traza_total = cp.trace(chi_pi) + cp.trace(chi_spin)
    coste = error_frob_sq + LAMBDA_REG * traza_total

    problema = cp.Problem(cp.Minimize(coste), restricciones)

    print("\nResolviendo con SCS")
    start_time = time.time()

    problema.solve(
        solver=cp.SCS, 
        verbose=True, 
        warm_start=usar_warm_start, 
        gpu=False
    )

    print(f"\nCoste óptimo final: {problema.value:.6e}. Tiempo empleado: {time.time()-start_time} s.")
    with open(OUTPUT_PATH+f"/datos_WS_{N_c}.csv", "a") as f:
        f.write(f"{max_orden_polinomio},{time.time()-start_time},{problema.value:.6f}\n")

    # ==============================================================================
    # EXTRACCIÓN Y GUARDADO DE DATOS
    # ==============================================================================
    def extraer_coefs(chi_matrix_val, basis_list, threshold=1e-5): # Umbral activado
        evals, evecs = la.eigh(chi_matrix_val)
        coefs_list = []
        
        # Ordenamos de mayor a menor (eigh los da de menor a mayor por defecto)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        for i in range(len(evals)):
            if evals[i] > threshold: 
                peso = np.sqrt(evals[i]) 
                coeficientes = peso * evecs[:, i]
                coefs_list.append(coeficientes)
                
        return np.array(coefs_list)

    def guardar_coeficientes(archivo_salida, coefs_pi, coefs_spin):
        np.savez(archivo_salida, coefs_pi=coefs_pi, coefs_spin=coefs_spin)
        print(f"\nCoeficientes guardados exitosamente en: {archivo_salida}.npz")

    print("\nExtrayendo operadores de salto resultantes")

    K_ops_pi = extraer_coefs(chi_pi.value, basis_pi)
    K_ops_spin = extraer_coefs(chi_spin.value, basis_spin_global)
    guardar_coeficientes(OUTPUT_PATH+f"/Ks/{max_orden_polinomio}_{N_c}_0.1", K_ops_pi, K_ops_spin)

    print(f"-> Canales Pi extraídos: {len(K_ops_pi)}")
    print(f"-> Canales Globales de Espín extraídos: {len(K_ops_spin)}")

    # Guardar las matrices puras para el Warm Start del siguiente grado
    np.savez(
        OUTPUT_PATH + f"/Ks/chi_raw_g{max_orden_polinomio}_{N_c}_0.1.npz", 
        chi_pi=chi_pi.value, 
        chi_spin=chi_spin.value
    )
    print(f"Matrices Chi originales guardadas para futuros Warm Starts.")