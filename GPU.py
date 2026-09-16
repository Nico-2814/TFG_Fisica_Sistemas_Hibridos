import os
import time
import jax
import jax.numpy as jnp
import qutip as qt
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize as opt
from functools import partial
from jax.scipy.sparse.linalg import gmres

# ==============================================================================
# CONFIGURACIÓN Y HARDWARE
# ==============================================================================
print("\n--- REPORTE DE HARDWARE ---")
dispositivos = jax.devices()
print(f"Dispositivos encontrados: {dispositivos}")

device_str = str(dispositivos[0]).lower()
if 'cuda' in device_str or 'gpu' in device_str:
    print(" GPU detectada correctamente.")
elif 'cpu' in device_str:
    print(" ¡OJO! Estás en CPU.")

# Configuración de rutas 
input_path = '/kaggle/input/datasets/nico880745/molecular-model-tfg-fis/'
output_path = '/kaggle/working/'

directorios = ["results/Ks", "results/steadies", "results/evals"]
for d in directorios:
    os.makedirs(os.path.join(output_path, d), exist_ok=True)

# Parámetros
N_c = 10 
S = 0.5 
N_Q = int(2*S+1)

I_c = qt.qeye(N_c)
I_Q = qt.qeye(N_Q)
a = qt.destroy(N_c)
a_q = qt.tensor(a, I_c)
a_d_q = a_q.dag()
a_p = qt.tensor(I_c, a)
a_d_p = a_p.dag()
k = 0.5
lambda_energia = 1.0

Q = 1/np.sqrt(2)*(a_q+a_d_q)
P = 1/np.sqrt(2)*(a_p+a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)

# Carga de H y rho_0
H_np = np.load(input_path+f"HCEs_kaggle/k_{k}/hamiltonian_k_{k}_2_10.npy")
rho_0np = np.array((qt.qload(input_path+f"HCEs_no_diag/k_{k}/HCE_1.0000_10")).full())
rho_0np = rho_0np / np.trace(rho_0np)

H_jax = jnp.array(H_np, dtype=jnp.complex128)
rho0_jax = jnp.array(rho_0np, dtype=jnp.complex128)

start_time_total=time.time()
# ==============================================================================
# CONSTRUCCIÓN DE LA BASE 
# ==============================================================================
def basis_creator(max_g):
    print(f"\nPrecalculando tensores base para orden {max_g}")
    basis_pi = []
    monomios_fase_total = [] 
    
    for order in range(max_g, -1, -1): 
        monomios_orden = []
        for exp_Q in range(order, -1, -1):
            exp_P = order - exp_Q
            monomios_orden.append(Q**exp_Q * P**exp_P)
            
        monomios_fase_total.extend(monomios_orden)
        
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_Q))
        for monomio in monomios_orden:
            basis_pi.append(qt.tensor(I_Q, I_Q, monomio * Pi_P))

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
            
    return basis_pi, basis_spin, len(monomios_fase_total)

def normalizar_base_matrices(basis):
    xp = jnp if isinstance(basis, jnp.ndarray) else np #decide si son de np o jnp
    normas = xp.linalg.norm(basis, axis=(-2, -1), keepdims=True)
    normas = xp.where(normas == 0, 1.0, normas)
    return basis / normas

def normalizar_base_qutip(basis_list):
    if isinstance(basis_list[0], list): 
        return [[op / max(np.linalg.norm(op.full()), 1e-12) for op in ops] for ops in basis_list]
    else: 
        return [op / max(np.linalg.norm(op.full()), 1e-12) for op in basis_list]


# ==============================================================================
# DEFINICIÓN DEL COSTE
# ==============================================================================
def crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, NUM_K_PI, NUM_K_SIGMA=1):
    num_basis_Pi = basis_Pi_jax.shape[0]
    num_spin_ops = basis_spin_jax.shape[0] 
    num_monomios = basis_spin_jax.shape[1]

    energia_ideal = jnp.trace(H_jax @ rho0_jax).real
    
    @jax.jit
    def cost_jax(p):
        p_Pi_end = NUM_K_PI * 2 * num_basis_Pi
        p_Pi = p[:p_Pi_end]
        p_sigma = p[p_Pi_end:]
        
        c_ops = []
        
        for m in range(NUM_K_PI):
            start = m * 2 * num_basis_Pi
            c = p_Pi[start : start + num_basis_Pi] + 1j * p_Pi[start + num_basis_Pi : start + 2*num_basis_Pi]
            c_ops.append(jnp.sum(c[:, None, None] * basis_Pi_jax, axis=0))
            
        params_per_sigma = 2 * num_monomios + 2 * num_spin_ops
        for m in range(NUM_K_SIGMA):
            start = m * params_per_sigma
            d_cpx = p_sigma[start : start + num_monomios] + 1j * p_sigma[start + num_monomios : start + 2*num_monomios]
            c_start = start + 2*num_monomios
            c_cpx = p_sigma[c_start : c_start + num_spin_ops] + 1j * p_sigma[c_start + num_spin_ops : c_start + 2*num_spin_ops]
            
            tensor_coefs = c_cpx[:, None] * d_cpx[None, :]
            V_sigma = jnp.sum(tensor_coefs[:, :, None, None] * basis_spin_jax, axis=(0, 1))
            c_ops.append(V_sigma)
            
        L_rho0 = -1j * (H_jax @ rho0_jax - rho0_jax @ H_jax)
        
        for K in c_ops:
            K_dag = K.conj().T
            K_tail = K_dag @ K
            L_rho0 += (K @ rho0_jax @ K_dag) - 0.5 * (K_tail @ rho0_jax + rho0_jax @ K_tail)
            
        # Penalizamos cuánto se desvía la derivada de ser 0
        coste_norma = jnp.sum(jnp.abs(L_rho0)**2) 
        
        return coste_norma * 100.0 # Escalado para estabilizar el gradiente
    cost_and_grad = jax.value_and_grad(cost_jax)

    def scipy_objective(p):
        val, grad = cost_and_grad(jnp.array(p))
        return float(val), np.array(grad, dtype=np.float64)
        
    return scipy_objective, cost_and_grad

# ==============================================================================
# MOTOR DE OPTIMIZACIÓN HÍBRIDO (ADAM + L-BFGS-B)
# ==============================================================================
def construir_adam_gpu(cost_and_grad):
    """
    Crea un optimizador Adam acelerado por hardware usando jax.lax.scan.
    Incluye Cosine Annealing y Langevin Dynamics (inyección de ruido).
    """
    @jax.jit(static_argnames=['num_steps'])
    def ejecutar_adam_gpu(p_init, num_steps, lr_max=1e-2, lr_min=1e-5, noise_scale_init=1e-3, seed=42):
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        def step_fn(state, i):
            p, m, v, key, noise_scale = state
            val, grad = cost_and_grad(p)
            
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * jnp.square(grad)
            
            m_hat = m / (1 - b1**(i + 1))
            v_hat = v / (1 - b2**(i + 1))
            
            # 1. Cosine Annealing Learning Rate
            lr_actual = lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * i / num_steps))
            
            # Paso base de Adam
            p_new = p - lr_actual * m_hat / (jnp.sqrt(v_hat) + eps)
            
            # Langevin Dynamics (Ruido Termodinámico)
            key, subkey = jax.random.split(key)
            ruido = jax.random.normal(subkey, p.shape)
            p_new = p_new + ruido * noise_scale
            
            # Enfriamiento: Reducir la magnitud del ruido un 1% cada iteración
            noise_scale_new = noise_scale * 0.99
            
            return (p_new, m, v, key, noise_scale_new), val

        # Inicializamos el estado ampliado con la clave JAX y la escala de ruido
        key = jax.random.PRNGKey(seed)
        init_state = (p_init, jnp.zeros_like(p_init), jnp.zeros_like(p_init), key, noise_scale_init)
        steps = jnp.arange(1, num_steps + 1)
        
        # Ejecución 'del bucle
        final_state, history = jax.lax.scan(step_fn, init_state, steps)
        
        # Retornamos los parámetros (índice 0) y el coste final
        return final_state[0], history[-1]
        
    return ejecutar_adam_gpu

def optimizar_hibrido(p_init, scipy_objective, adam_fn, iter_adam=200, iter_bfgs=1000, lr_max=1e-2):
    jax.clear_caches()
    """Fase 1: Escaneo rápido con Adam | Fase 2: Ajuste fino con L-BFGS-B"""
    p_jax = jnp.array(p_init)
    
    # Adam
    if iter_adam > 0:
        # Pasamos lr_max para alimentar el límite superior del Cosine Annealing
        p_opt_adam, coste_adam = adam_fn(p_jax, iter_adam, lr_max=lr_max)
        print("     Adam completado")
        p_init_bfgs = np.array(p_opt_adam)
    else:
        p_init_bfgs = p_init
        
    # L-BFGS-B 
    res = opt.minimize(scipy_objective, p_init_bfgs, method='L-BFGS-B', jac=True, options={'maxiter': iter_bfgs})
    print(f"     L-BFGS-B: {res.nit} iteraciones")
    return res.x, res.fun


# ==============================================================================
# WARM STARTS
# ==============================================================================

def inyectar_warm_start(p_old, num_pi_old, num_pi_new, num_monomios_old, num_monomios_new, NUM_K_PI, NUM_K_SIGMA=1):
    delta_pi = num_pi_new - num_pi_old
    delta_monomios = num_monomios_new - num_monomios_old
    p_new = []
    
    p_Pi_old = p_old[:NUM_K_PI * 2 * num_pi_old]
    for m in range(NUM_K_PI):
        start = m * 2 * num_pi_old
        real_old = p_Pi_old[start : start + num_pi_old]
        imag_old = p_Pi_old[start + num_pi_old : start + 2 * num_pi_old]
        
        real_new = np.concatenate([np.random.randn(delta_pi) * 1e-6, real_old])
        imag_new = np.concatenate([np.random.randn(delta_pi) * 1e-6, imag_old])
        p_new.extend(real_new)
        p_new.extend(imag_new)
        
    p_sigma_old = p_old[NUM_K_PI * 2 * num_pi_old:]
    params_per_sigma_old = 2 * num_monomios_old + 30
    
    for m in range(NUM_K_SIGMA):
        start = m * params_per_sigma_old
        d_real_old = p_sigma_old[start : start + num_monomios_old]
        d_imag_old = p_sigma_old[start + num_monomios_old : start + 2*num_monomios_old]
        
        c_start = start + 2*num_monomios_old
        c_real = p_sigma_old[c_start : c_start + 15]
        c_imag = p_sigma_old[c_start + 15 : c_start + 30]
        
        d_real_new = np.concatenate([np.random.randn(delta_monomios) * 1e-6, d_real_old])
        d_imag_new = np.concatenate([np.random.randn(delta_monomios) * 1e-6, d_imag_old])
        
        p_new.extend(d_real_new)
        p_new.extend(d_imag_new)
        p_new.extend(c_real)
        p_new.extend(c_imag)
        
    return np.array(p_new)

def inyectar_warm_start_cvxpy(coefs_pi_cpx, coefs_sigma_d_cpx, coefs_sigma_c_cpx, 
                              NUM_K_PI, NUM_K_SIGMA, num_basis_Pi, num_monomios, num_spin_ops=15):
    """
    Convierte las matrices de coeficientes de CVXPY al vector plano p de JAX.
    Aplica relleno de 0s respetando que los órdenes bajos van al final
    """
    p_new = []
    
    for m in range(NUM_K_PI):
        if m < len(coefs_pi_cpx):
            c_old = np.array(coefs_pi_cpx[m])
            c = np.zeros(num_basis_Pi, dtype=complex)
            
            if len(c_old) <= num_basis_Pi:
                c[-len(c_old):] = c_old
                c[:-len(c_old)] = np.random.randn(num_basis_Pi - len(c_old)) * 1e-6 + 1j * np.random.randn(num_basis_Pi - len(c_old)) * 1e-6
            else:
                c = c_old[:num_basis_Pi]
        else:
            c = np.random.randn(num_basis_Pi) * 1e-6 + 1j * np.random.randn(num_basis_Pi) * 1e-6
            
        p_new.extend(c.real)
        p_new.extend(c.imag)
        
    for m in range(NUM_K_SIGMA):
        if m < len(coefs_sigma_d_cpx):
            d = np.array(coefs_sigma_d_cpx[m])
            c_spin = np.array(coefs_sigma_c_cpx[m])
        else:
            d = np.random.randn(num_monomios) * 1e-6 + 1j * np.random.randn(num_monomios) * 1e-6
            c_spin = np.random.randn(num_spin_ops) * 1e-6 + 1j * np.random.randn(num_spin_ops) * 1e-6
            
        p_new.extend(d.real)
        p_new.extend(d.imag)
        p_new.extend(c_spin.real)
        p_new.extend(c_spin.imag)
        
    return np.array(p_new, dtype=np.float64)
def inyectar_warm_start_grid(p_old, num_pi_old, num_pi_new, num_monomios_old, num_monomios_new, num_ops_old, num_ops_new):
    p_new = []
    
    # 1. CANALES PI
    p_Pi_old_total = p_old[:num_ops_old * 2 * num_pi_old]
    for m in range(num_ops_new):
        if m < num_ops_old:
            start = m * 2 * num_pi_old
            real_old = p_Pi_old_total[start : start + num_pi_old]
            imag_old = p_Pi_old_total[start + num_pi_old : start + 2 * num_pi_old]
        else:
            real_old, imag_old = np.array([]), np.array([])
            
        delta_pi = num_pi_new - len(real_old)
        real_new = np.concatenate([np.random.randn(delta_pi) * 1e-4, real_old]) if delta_pi > 0 else real_old[-num_pi_new:]
        imag_new = np.concatenate([np.random.randn(delta_pi) * 1e-4, imag_old]) if delta_pi > 0 else imag_old[-num_pi_new:]
        
        p_new.extend(real_new)
        p_new.extend(imag_new)
        
    # 2. CANALES SIGMA
    p_sigma_old_total = p_old[num_ops_old * 2 * num_pi_old:]
    params_per_sigma_old = 2 * num_monomios_old + 30
    
    for m in range(num_ops_new):
        if m < num_ops_old:
            start = m * params_per_sigma_old
            d_real_old = p_sigma_old_total[start : start + num_monomios_old]
            d_imag_old = p_sigma_old_total[start + num_monomios_old : start + 2*num_monomios_old]
            
            c_start = start + 2*num_monomios_old
            c_real_old = p_sigma_old_total[c_start : c_start + 15]
            c_imag_old = p_sigma_old_total[c_start + 15 : c_start + 30]
        else:
            d_real_old, d_imag_old = np.array([]), np.array([])
            c_real_old = np.random.randn(15) * 1e-4
            c_imag_old = np.random.randn(15) * 1e-4
            
        delta_monomios = num_monomios_new - len(d_real_old)
        d_real_new = np.concatenate([np.random.randn(delta_monomios) * 1e-4, d_real_old]) if delta_monomios > 0 else d_real_old[-num_monomios_new:]
        d_imag_new = np.concatenate([np.random.randn(delta_monomios) * 1e-4, d_imag_old]) if delta_monomios > 0 else d_imag_old[-num_monomios_new:]
        
        p_new.extend(d_real_new)
        p_new.extend(d_imag_new)
        p_new.extend(c_real_old)
        p_new.extend(c_imag_old)
        
    return np.array(p_new)
    
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


import matplotlib.pyplot as plt

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

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse.linalg as spla
import time

import cupy as cp
import cupyx.scipy.sparse.linalg as cspla
import numpy as np
import time

import cupy as cp
import numpy as np
import scipy.sparse.linalg as spla
import time

def calcular_liouvilliano_hibrido_cupy(H, K_ops, ke=10):
    print(f"Calculando los {ke} autovalores (Híbrido SciPy CPU + CuPy GPU)...")
    
    # 1. Subir matrices a GPU una sola vez
    H_gpu = cp.asarray(H, dtype=cp.complex128)
    N = H_gpu.shape[0]
    dim_L = N * N
    
    # Apilar K_ops y precalcular todo lo posible en la GPU
    Ks_gpu = cp.stack([cp.asarray(K, dtype=cp.complex128) for K in K_ops])
    Ks_dag = cp.conj(Ks_gpu).transpose((0, 2, 1))
    K_dag_K_sum = cp.sum(Ks_dag @ Ks_gpu, axis=0)

    def matvec_hibrido(rho_flat_cpu):
        # Enviar el vector de scipy a la GPU
        rho_gpu = cp.asarray(rho_flat_cpu).reshape((N, N))
        
        # 100% en GPU
        comm = -1j * (H_gpu @ rho_gpu - rho_gpu @ H_gpu)
        term1 = cp.sum(Ks_gpu @ rho_gpu @ Ks_dag, axis=0)
        term2 = -0.5 * (K_dag_K_sum @ rho_gpu + rho_gpu @ K_dag_K_sum)
        
        res_gpu = comm + term1 + term2
        
        # Devolver resultado plano a la CPU para que scipy continúe
        return cp.asnumpy(res_gpu).ravel()

    # Operador lineal de SciPy en CPU
    L_op = spla.LinearOperator((dim_L, dim_L), matvec=matvec_hibrido, dtype=np.complex128)

    v0 = np.eye(N, dtype=np.complex128).ravel()
    v0 = v0 / np.linalg.norm(v0)
    
    # Parámetros estabilizadores de Arnoldi
    ncv_val = min(dim_L, max(5 * ke, 50))

    start_time = time.time()
    
    # Ejecución
    print(f"Ejecutando Arnoldi: k={ke}, ncv={ncv_val} (dim_L={dim_L})...")
    try:
        evals, _ = spla.eigs(
            L_op, 
            k=ke, 
            which='LR', 
            v0=v0, 
            ncv=ncv_val,
            tol=1e-6,
            maxiter=3000
        )
    except spla.ArpackNoConvergence as e:
        print(f"\n[!] Advertencia: Solo convergieron {len(e.eigenvalues)} autovalores.")
        evals = e.eigenvalues
        
    print(f"Espectro obtenido en {time.time() - start_time:.2f} s")

    plt.figure(figsize=(10, 8))
    plt.scatter(evals.real, evals.imag, c=evals.real, cmap='viridis_r', s=30, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f"Espectro del liouvilliano, "+r"$\kappa=$"+f"{k}"+r", $N_C=$"+f"{N_c}", fontsize=20)
    plt.xlabel("Re(λ)", fontsize=16)
    plt.ylabel("Im(λ)", fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path+f"results/evals/liouvillian_evals.png")
    plt.close()

    # Ordenar autovalores por módulo de su parte real
    idx = np.argsort(np.abs(evals.real))
    autovalores_ordenados = evals[idx]
    
    print(f"\nAutovalores obtenidos:")
    for i, ev in enumerate(autovalores_ordenados[:10]):
        print(f"λ_{i+1} = {ev.real:.8e} + {ev.imag:.8e}j")
        
    return autovalores_ordenados


import scipy.sparse as sp

def calcular_liouvilliano_matrixfree_cpu(H, K_ops, ke=10):
    N = H.shape[0]
    dim_L = N * N
    print(f"Preparando operadores Matrix-Free para N={N} (dim_L={dim_L})...")
    
    # Asegurar que H y Ks son dispersas
    H_sp = sp.csr_matrix(H, dtype=np.complex128)
    H_sp_T = H_sp.T.tocsr()
    
    Ks_sp = [sp.csr_matrix(K, dtype=np.complex128) for K in K_ops]
    Ks_dag_sp = [K.conjugate().T.tocsr() for K in Ks_sp]
    Ks_dag_T_sp = [K_dag.T.tocsr() for K_dag in Ks_dag_sp]
    
    # Precalcular M = sum(K^\dagger K)
    M_sp = sp.csr_matrix((N, N), dtype=np.complex128)
    for K_dag, K in zip(Ks_dag_sp, Ks_sp):
        M_sp = M_sp + (K_dag @ K)
    M_sp_T = M_sp.T.tocsr()

    def matvec(v):
        rho = v.reshape((N, N))
         
        H_rho = H_sp @ rho
        rho_H = (H_sp_T @ rho.T).T  #
        comm = -1j * (H_rho - rho_H)
        
        diss = np.zeros_like(rho)
        for K, K_dag_T in zip(Ks_sp, Ks_dag_T_sp):
            K_rho = K @ rho
            K_rho_Kdag = (K_dag_T @ K_rho.T).T 
            diss += K_rho_Kdag
            
        M_rho = M_sp @ rho
        rho_M = (M_sp_T @ rho.T).T
        diss -= 0.5 * (M_rho + rho_M)
        
        return (comm + diss).ravel()

    L_op = spla.LinearOperator((dim_L, dim_L), matvec=matvec, dtype=np.complex128)
    
    v0 = np.eye(N, dtype=np.complex128).ravel()
    v0 = v0 / np.linalg.norm(v0)
    
    ncv_val = min(dim_L, max(5 * ke, 50))
    
    print(f"Ejecutando Arnoldi en CPU con ncv={ncv_val}...")
    start_time = time.time()
    
    try:
        evals, _ = spla.eigs(
            L_op, 
            k=ke, 
            which='LR', 
            v0=v0, 
            ncv=ncv_val,
            tol=1e-6,
            maxiter=5000
        )
    except spla.ArpackNoConvergence as e:
        print(f"\n[!] Advertencia: Solo convergieron {len(e.eigenvalues)} autovalores.")
        evals = e.eigenvalues
        
    print(f"Espectro obtenido en {time.time() - start_time:.2f} s")

    tolerancia = 1e-8
    
    # Calcular la multiplicidad de cada autovalor
    multiplicidades = np.zeros(len(evals), dtype=int)
    for i, ev in enumerate(evals):
        distancias = np.abs(evals - ev)
        multiplicidades[i] = np.sum(distancias < tolerancia)

    plt.figure(figsize=(10, 8))
    plt.scatter(evals.real, evals.imag, c=multiplicidades, cmap='viridis_r', s=30, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f"Espectro del liouvilliano, "+r"$\kappa=$"+f"{k}"+r", $N_C=$"+f"{N_c}", fontsize=20)
    plt.xlabel("Re(λ)", fontsize=16)
    plt.ylabel("Im(λ)", fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path+f"results/evals/liouvillian_evals.png")
    plt.close()
    
    # Ordenar y mostrar
    idx = np.argsort(np.abs(evals.real))
    autovalores = evals[idx]
    
    print("\nAutovalores obtenidos:")
    for i, ev in enumerate(autovalores[:10]):
        print(f"λ_{i+1} = {ev.real:.8e} + {ev.imag:.8e}j")
        
    return autovalores

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

        # Operador lineal para el Solver iterativo
    def matvec(rho_flat):
        rho = rho_flat.reshape((N, N))
        drho = apply_L(rho)
        drho_flat = drho.flatten()
            
        trace_rho = jnp.trace(rho)
        return drho_flat.at[0].set(trace_rho)

    rhs = jnp.zeros(N * N, dtype=jnp.complex128).at[0].set(1.0)
        
    x0 = (jnp.eye(N, dtype=jnp.complex128) / N).flatten()

    rho_steady_flat, info = gmres(matvec, rhs, x0=x0, tol=1e-8, restart=50, maxiter=100)
    
    return rho_steady_flat.reshape((N, N))
    
rho0qt = qt.Qobj(rho_0np, dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
# ==============================================================================
# BÚSQUEDA EN CUADRÍCULA Y POST-PROCESADO
# ==============================================================================
grados = [2, 3, 4, 5]
lista_operadores = [5, 10, 15, 20]

# Variables para arrastrar el warm start a través del grid
mejor_p_global = None
num_pi_viejo = None
num_monomios_viejo = None
num_ops_viejo = None

for g in grados:
    print(f"\n==================================================")
    print(f" GENERANDO BASE PARA ORDEN {g}")
    print(f"==================================================")
    
    basis_Pi, basis_spin, num_monomios_actual = basis_creator(g)
    num_basis_Pi_actual = len(basis_Pi)
    num_spin_ops_actual = 15 
    
    basis_Pi_jax = normalizar_base_matrices(jnp.array([op.full() for op in basis_Pi], dtype=jnp.complex128))
    basis_spin_jax = normalizar_base_matrices(jnp.array([[op.full() for op in ops] for ops in basis_spin], dtype=jnp.complex128))

    for num_ops in lista_operadores:
        if time.time() - start_time_total > 41400:
            print("Timeout global alcanzado.")
            break
            
        print(f"\n--- OPTIMIZANDO: ORDEN {g} | OPERADORES Pi/Sigma = {num_ops} ---")
        NUM_K_PI = num_ops
        NUM_K_SIGMA = num_ops
        
        scipy_objective, cost_and_grad = crear_funciones_optimizacion(basis_Pi_jax, basis_spin_jax, NUM_K_PI, NUM_K_SIGMA)
        adam_gpu = construir_adam_gpu(cost_and_grad)
        
        if mejor_p_global is None:
            dim_total = NUM_K_PI * 2 * num_basis_Pi_actual + NUM_K_SIGMA * (2 * num_monomios_actual + 30)
            p_actual = np.random.randn(dim_total) * 1e-4
        else:
            p_actual = inyectar_warm_start_grid(
                mejor_p_global, 
                num_pi_viejo, num_basis_Pi_actual, 
                num_monomios_viejo, num_monomios_actual, 
                num_ops_viejo, num_ops
            )
            
        p_final, coste_final = optimizar_hibrido(p_actual, scipy_objective, adam_gpu, iter_adam=500, iter_bfgs=1000)
        
        # Actualizar variables de estado para la siguiente iteración
        mejor_p_global = p_final
        num_pi_viejo = num_basis_Pi_actual
        num_monomios_viejo = num_monomios_actual
        num_ops_viejo = num_ops
        
        suffix = f"g{g}_ops{num_ops}"
        ruta_ks = output_path + f"results/Ks/K_params_{suffix}.txt"
        guardar_parametros(ruta_ks, p_final, num_basis_Pi_actual, num_monomios_actual, NUM_K_PI, NUM_K_SIGMA)
        
        print(f" Calculando y guardando analíticas para {suffix}...")
        Ks = cargar_operadores_K(ruta_ks, normalizar_base_qutip(basis_Pi), normalizar_base_qutip(basis_spin))
        Ks_jax = [jnp.array(K.full(), dtype=jnp.complex128) for K in Ks]
        N = H_jax.shape[0]
        
        # Steady state
        rho_steady = calcular_steady_state_gpu(H_jax, Ks_jax)
        rho_qt = qt.Qobj(np.array(rho_steady), dims=[[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]])
        qt.qsave(rho_qt, output_path + f"results/steadies/steady_{suffix}")
        
        
        evals = calcular_liouvilliano_matrixfree_cpu(H_jax, Ks_jax, ke=11)
        print(evals)
        np.save(output_path + f"results/evals/liouvillian_evals.npy", evals)
        
        # Métricas
        diff = rho_steady - rho0_jax
        dist = jnp.sum(jnp.abs(diff)**2) / (N_c**2 * N_Q**2)
        print(f" -> Fidelidad vs Rho0: {qt.fidelity(rho_qt, rho0qt):.6f}")
        print(f" -> Distancia: {dist:.8f}")
print("\nOptimización híbrida completada con éxito")