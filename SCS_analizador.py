import os
import time
import jax
import jax.numpy as jnp
import qutip as qt
import numpy as np

# ==============================================================================
# FASE 0: CONFIGURACIÓN Y HARDWARE
# ==============================================================================
print("\n--- REPORTE DE HARDWARE ---")
dispositivos = jax.devices()
print(f"Dispositivos encontrados: {dispositivos}")

device_str = str(dispositivos[0]).lower()
if 'cuda' in device_str or 'gpu' in device_str:
    print(" GPU detectada correctamente.")
elif 'cpu' in device_str:
    print(" ¡OJO! Estás en CPU.")

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='

# Configuración de rutas 
input_path = '/kaggle/input/datasets/nico880745/molecular-model-tfg-fis/'
output_path = '/kaggle/working/'

# Parámetros
N_c = 10 
S = 0.5 
N_Q = int(2*S+1)

k = 1.0
lambda_energia = 1.0

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

# Carga de H y rho_0
H_np = np.load(input_path+f"HCEs_kaggle/k_{k}/hamiltonian_k_1_2_{N_c}.npy")
rho_0np = np.array((qt.qload(input_path+f"HCEs_kaggle/k_{k}/HCE_1.0000_{N_c}")).full())
rho_0np = rho_0np / np.trace(rho_0np)

H_jax = jnp.array(H_np, dtype=jnp.complex128)
rho0_jax = jnp.array(rho_0np, dtype=jnp.complex128)

start_time_total=time.time()
# ==============================================================================
# FASE 1: CONSTRUCCIÓN DE LA BASE (FÍSICA RIGUROSA)
# ==============================================================================
def basis_creator(max_g):
    print(f"\nPrecalculando tensores base para orden {max_g}")
    basis_pi = []
    ordenes_pi = []       
    
    monomios_fase_total = [] 
    ordenes_monomios = []        
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            ordenes_monomios.append(order) # Guardamos el orden físico del monomio
            
        monomios_fase_total.extend(monomios_orden)
        
        # 1. Base Pi
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio))
            ordenes_pi.append(order)
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_Q))
            ordenes_pi.append(order)
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_P))
            ordenes_pi.append(order)

    # 2. Operadores de Espín
    spin_ops = [
        (qt.sigmax(), I_Q), (qt.sigmay(), I_Q), (qt.sigmaz(), I_Q),
        (I_Q, qt.sigmax()), (I_Q, qt.sigmay()), (I_Q, qt.sigmaz()),
        (qt.sigmax(), qt.sigmax()), (qt.sigmax(), qt.sigmay()), (qt.sigmax(), qt.sigmaz()),
        (qt.sigmay(), qt.sigmax()), (qt.sigmay(), qt.sigmay()), (qt.sigmay(), qt.sigmaz()),
        (qt.sigmaz(), qt.sigmax()), (qt.sigmaz(), qt.sigmay()), (qt.sigmaz(), qt.sigmaz())
    ]

    basis_spin = []
    for op1, op2 in spin_ops:
        op_basis = []
        for monomio in monomios_fase_total:
            op_basis.append(qt.tensor(op1, op2, monomio))
        basis_spin.append(op_basis)
            
    return (
        basis_pi, 
        basis_spin, 
        len(monomios_fase_total), 
        np.array(ordenes_pi, dtype=np.float32), 
        np.array(ordenes_monomios, dtype=np.float32)
    )

# ==============================================================================
# DEFINICIÓN DEL COSTE
# ==============================================================================

dim = rho_0np.shape[0]

def inyectar_warm_start_cvxpy(coefs_pi_cpx, coefs_sigma_d_cpx, coefs_sigma_c_cpx, 
                              NUM_K_PI, NUM_K_SIGMA, num_basis_Pi, num_monomios,
                              num_spin_ops=15):
    p_new = []
    
    # --- CANAL PI ---
    for m in range(NUM_K_PI):
        if m < len(coefs_pi_cpx):
            c_old = np.array(coefs_pi_cpx[m])
            c = np.zeros(num_basis_Pi, dtype=complex)
            
            if len(c_old) <= num_basis_Pi:
                c_scaled = c_old
                c[-len(c_old):] = c_scaled
                c[:-len(c_old)] = np.random.randn(num_basis_Pi - len(c_old)) * 1e-6 + 1j * np.random.randn(num_basis_Pi - len(c_old)) * 1e-6
            else:
                c = c_old[:num_basis_Pi]
        else:
            c = np.random.randn(num_basis_Pi) * 1e-6 + 1j * np.random.randn(num_basis_Pi) * 1e-6
            
        p_new.extend(c.real)
        p_new.extend(c.imag)
        
    # --- CANAL SIGMA ---
    for m in range(NUM_K_SIGMA):
        if m < len(coefs_sigma_d_cpx):
            d = np.array(coefs_sigma_d_cpx[m])
            c_spin = np.array(coefs_sigma_c_cpx[m])
            
            d_scaled = d
        else:
            d_scaled = np.random.randn(num_monomios) * 1e-6 + 1j * np.random.randn(num_monomios) * 1e-6
            c_spin = np.random.randn(num_spin_ops) * 1e-6 + 1j * np.random.randn(num_spin_ops) * 1e-6
            
        p_new.extend(d_scaled.real)
        p_new.extend(d_scaled.imag)
        p_new.extend(c_spin.real)
        p_new.extend(c_spin.imag)
        
    return np.array(p_new, dtype=np.float64)

def guardar_parametros(filename, p, num_pi, num_monomios, NUM_K_PI, NUM_K_SIGMA=1):
    with open(filename, "w", encoding='utf-8') as f:
        f.write("=== OPERADORES DE DERIVADA (Pi) ===\n")
        p_Pi = p[:NUM_K_PI * 2 * num_pi]
        for m in range(NUM_K_PI):
            f.write(f"--- K_Pi_{m+1} ---\n")
            start = m * 2 * num_pi
            c = p_Pi[start : start + num_pi] + 1j * p_Pi[start + num_pi : start + 2*num_pi]
            for val in c: f.write(f"{val.real} + {val.imag}j\n")
            
        f.write("\n=== OPERADOR SIGMA ===\n")
        p_sigma = p[NUM_K_PI * 2 * num_pi:]
        params_per_sigma = 2 * num_monomios + 30
        for m in range(NUM_K_SIGMA):
            start = m * params_per_sigma
            d_real = p_sigma[start : start + num_monomios]
            d_imag = p_sigma[start + num_monomios : start + 2*num_monomios]
            d = d_real + 1j * d_imag
            
            c_start = start + 2*num_monomios
            c_real = p_sigma[c_start : c_start + 15]
            c_imag = p_sigma[c_start + 15 : c_start + 30]
            c = c_real + 1j * c_imag
            
            f.write(f"--- K_Sigma_{m+1}_Polinomio ---\n")
            for val in d: f.write(f"{val.real} + {val.imag}j\n")
            
            f.write(f"--- K_Sigma_{m+1}_Spins ---\n")
            for val in c: f.write(f"{val.real} + {val.imag}j\n")


def cargar_operadores_K(filepath, basis_Pi, basis_spin):
    jump_ops = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("===")]
        
    blocks = {}
    current_name = ""
    for linea in lines:
        if linea.startswith("---"):
            current_name = linea.replace("-", "").strip()
            blocks[current_name] = []
        else:
            str_c = linea.replace(' ', '').replace('+-', '-')
            blocks[current_name].append(complex(str_c))
            
    for name, coefs in blocks.items():
        if "Pi" in name:
            K = 0 * basis_Pi[0]
            for c, op in zip(coefs, basis_Pi):
                K += c * op
            jump_ops.append(K)
            
    sigma_indices = set([name.split("_")[2] for name in blocks.keys() if "Sigma" in name])
    for idx in sigma_indices:
        d_coefs = blocks[f"K_Sigma_{idx}_Polinomio"]
        c_coefs = blocks[f"K_Sigma_{idx}_Spins"]
        
        V_sigma = 0 * basis_spin[0][0]
        for k, c_val in enumerate(c_coefs):
            for m, d_val in enumerate(d_coefs):
                V_sigma += c_val * d_val * basis_spin[k][m]
        jump_ops.append(V_sigma)
        
    return jump_ops

from scipy.sparse.linalg import LinearOperator, eigs

def calcular_autovalores_liouvilliano_gpu(H, K_ops, k=100):
        print(f"Calculando los {k} autovalores principales del Liouvilliano")
        
        @jax.jit
        def apply_L_flat(rho_flat):
            rho = rho_flat.reshape((N, N))
            comm = -1j * (H @ rho - rho @ H)
            dissipator = jnp.zeros_like(rho)
            for K in K_ops:
                K_dag = K.conj().T
                K_tail = K_dag @ K
                dissipator += (K @ rho @ K_dag) - 0.5 * (K_tail @ rho + rho @ K_tail)
            return (comm + dissipator).flatten()

        # Función puente entre SciPy y JAX
        def matvec(v):
            return np.array(apply_L_flat(jnp.array(v, dtype=jnp.complex128)))

        dim = N * N
        L_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
        
        # 4. Buscamos los 'k' autovalores con la parte real más grande (LR)
        evals, _ = eigs(L_op, k=k, which='LR', tol=1e-2, maxiter=1000)
        return evals

rho0qt = qt.Qobj(rho_0np, dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])

# ==============================================================================
# ANÁLISIS DE SCS
# ==============================================================================
from jax.scipy.sparse.linalg import gmres, bicgstab

for order in range(0, 4):
    print(f"\n==================================================")
    print(f" PROCESANDO Y EVALUANDO ORDEN {order}")
    print(f"==================================================")
    
    basis_Pi, basis_spin, num_monomios_actual, pesos_pi, pesos_sigma = basis_creator(order)
    num_basis_Pi_actual = len(basis_Pi)
    num_spin_ops_actual = 15

    ruta_cvxpy = input_path + f"/Ks_subir/{order}_{N_c}.npz"
    if not os.path.exists(ruta_cvxpy):
        ruta_cvxpy = input_path + f"/{order}_{N_c}.npz"
        
    print(f"Cargando solución CVXPY/SCS desde: {ruta_cvxpy}")
    
    try:
        datos_cvxpy = np.load(ruta_cvxpy)
    except FileNotFoundError:
        print(f"No se encontró el archivo en {ruta_cvxpy}. Saltando orden {order}")
        continue
        
    coefs_pi_cvxpy = datos_cvxpy['coefs_pi']
    K_ops_spin_raw = datos_cvxpy['coefs_spin']
    
    NUM_K_PI = len(coefs_pi_cvxpy)
    NUM_K_SIGMA = len(K_ops_spin_raw)
    
    coefs_sig_d_cvxpy, coefs_sig_c_cvxpy = [], []
    
    for m in range(NUM_K_SIGMA):
        num_monomios_cvxpy = len(K_ops_spin_raw[m]) // num_spin_ops_actual
        matriz_canal = K_ops_spin_raw[m].reshape((num_spin_ops_actual, num_monomios_cvxpy))
        U, S, Vh = np.linalg.svd(matriz_canal, full_matrices=False)
        
        c_spin = np.sqrt(S[0]) * U[:, 0]
        d_espacial_cvxpy = np.sqrt(S[0]) * Vh[0, :]
        
        d_espacial = np.zeros(num_monomios_actual, dtype=complex)
        
        d_espacial[-num_monomios_cvxpy:] = d_espacial_cvxpy
        
        coefs_sig_c_cvxpy.append(c_spin)
        coefs_sig_d_cvxpy.append(d_espacial)
    
    p_warm_start = inyectar_warm_start_cvxpy(
        coefs_pi_cvxpy, coefs_sig_d_cvxpy, coefs_sig_c_cvxpy, 
        NUM_K_PI, NUM_K_SIGMA, num_basis_Pi_actual, num_monomios_actual,
        num_spin_ops_actual
    )
    
    archivo_params = output_path + f"results/Ks/K_params_g{order}.txt"
    guardar_parametros(archivo_params, p_warm_start, num_basis_Pi_actual, num_monomios_actual, NUM_K_PI, NUM_K_SIGMA)
    
    # 5. Reconstruir operadores K
    Ks = cargar_operadores_K(
        archivo_params, 
        basis_Pi, 
        basis_spin
    )
    Ks_jax = [jnp.array(K.full(), dtype=jnp.complex128) for K in Ks]
    N = H_jax.shape[0]

    # 6. Cálculo del estado estacionario con preacondicionador Jacobi
    @jax.jit
    def calcular_steady_state_gpu(H, K_ops):
        def apply_L(rho):
            comm = -1j * (H @ rho - rho @ H)
            dissipator = jnp.zeros_like(rho)
            for K in K_ops:
                K_dag = K.conj().T
                K_tail = K_dag @ K
                dissipator += (K @ rho @ K_dag) - 0.5 * (K_tail @ rho + rho @ K_tail)
            return comm + dissipator

        def matvec(rho_flat):
            rho = rho_flat.reshape((N, N))
            drho = apply_L(rho)
            return drho.flatten().at[0].set(jnp.trace(rho))

        # Preacondicionador Jacobi
        H_diag = jnp.diag(H)
        inv_M_vec = 1.0 / (-1j * (H_diag[:, None] - H_diag[None, :]).flatten().at[0].set(1.0) + 1e-12)

        rhs = jnp.zeros(N * N, dtype=jnp.complex128).at[0].set(1.0)
        x0 = (jnp.eye(N, dtype=jnp.complex128) / N).flatten()

        rho_steady_flat, _ = gmres(matvec, rhs, x0=x0, tol=1e-8, restart=50, maxiter=100, M=lambda x: inv_M_vec * x)
        return rho_steady_flat.reshape((N, N))

    print("Calculando estado estacionario en GPU con GMRES...")
    start_time = time.time()
    rho_steady = calcular_steady_state_gpu(H_jax, Ks_jax)
    print(f" Estado estacionario calculated in {time.time() - start_time:.2f} s")

    if jnp.isnan(rho_steady).any():
        print("GMRES no convergió. Intentando solución alternativa con BiCGSTAB")
        # BiCGSTAB suele ser más robusto ante matrices mal condicionadas
        @jax.jit
        def resolver_bicgstab(H, K_ops):
            def matvec(rho_flat):
                rho = rho_flat.reshape((N, N))
                comm = -1j * (H @ rho - rho @ H)
                dissipator = jnp.zeros_like(rho)
                for K in K_ops:
                    K_dag = K.conj().T
                    dissipator += (K @ rho @ K_dag) - 0.5 * (K_dag @ K @ rho + rho @ K_dag @ K)
                return (comm + dissipator).flatten().at[0].set(jnp.trace(rho))
            
            rhs = jnp.zeros(N * N, dtype=jnp.complex128).at[0].set(1.0)
            x0 = (jnp.eye(N, dtype=jnp.complex128) / N).flatten()
            res, _ = bicgstab(matvec, rhs, x0=x0, tol=1e-7, maxiter=200)
            return res.reshape((N, N))
            
        rho_steady = resolver_bicgstab(H_jax, Ks_jax)

        print(f"Estado estacionario calculado en {time.time() - start_time:.2f} s")
    
    diff = rho_steady - rho0_jax
    dist = jnp.sum(jnp.abs(diff)**2) / (N_c**2 * N_Q**2)
    rho_qt = qt.Qobj(np.array(rho_steady), dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
        
    print(f"-> Distancia final al estado objetivo: {dist:.8f}")
    print(f"-> Fidelidad: {qt.fidelity(rho_qt, rho0qt):.6f}")
    
    qt.qsave(rho_qt, output_path + f"/steady_g{order}")