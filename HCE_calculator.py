import numpy as np
from sympy import *
import math
from scipy import special
from qutip import Qobj, qsave
from os import makedirs
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import rcParams
rcParams.update({'font.size': 16})

# ==========================================
# PARÁMETROS FÍSICOS Y DE LA CUADRÍCULA
# ==========================================
q_max = 10 
N_points = 3000 

betas = np.logspace(-0.2, 0.2, 50)    #Temperaturas
#betas=[1]
N_c = 5
N_Q = 2

k_val = 1.0   # Parámetro k

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def phi_n(n, x):
    return 1/math.sqrt(2**n*math.factorial(n))*math.pi**(-0.25)*np.exp(-np.array(x)**2/2)*special.eval_hermite(n, x)

def calcular_bloque_einsum(m, mp, rho_val, phi_p, D2, N2):
    B = np.einsum('ni,ij,mj->nm', phi_p, np.sqrt(rho_val.astype('complex128')), phi_p, optimize=True) * D2
    B2=np.einsum('ab,cd->abcd', B, B, optimize=True)
    return m, mp, B2.reshape(N2, N2)

# Crear directorios
makedirs("molecular_model", exist_ok=True)
makedirs("molecular_model/HCEs", exist_ok=True)
makedirs(f"molecular_model/HCEs/k_{k_val}", exist_ok=True)


Nc2 = N_c**2
NQ2 = N_Q**2
dim = Nc2 * NQ2

Delta = 2 * q_max / (N_points - 1)
qp_list = np.linspace(-q_max, q_max, N_points)
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')

# ==========================================
# DEFINICIÓN DEL HAMILTONIANO
# ==========================================
q, p, A, B, C, D, k_sym, E, T = symbols("q p A B C D k E T")

Afun = Piecewise(
    (A * (1 - exp(-B*q)), q > 0),
    (A * (exp(B*q) - 1), True)
) 
Cfun = C * exp(-D*q**2)

H_sym = Matrix([
    [0.5*k_sym*(p**2+q**2)+2*E, T, T, 0], 
    [T, 0.5*k_sym*(p**2+q**2), 0, T], 
    [T, 0, 0.5*k_sym*(p**2+q**2), T], 
    [0, T, T, 0.5*k_sym*(p**2+q**2)-2*E]
])

parametros = {
    A: 0.01,
    B: 1.6,
    C: 0.005,
    D: 1.0,
    k_sym: k_val
}

H_sub = H_sym.subs({E: Afun, T: Cfun}).subs(parametros)
H_tuple = tuple(tuple(H_sub[i, j] for j in range(NQ2)) for i in range(NQ2))
H_func = lambdify((q, p), H_tuple, modules='numpy')

# ==========================================
# EVALUACIÓN NUMÉRICA
# ==========================================
print("Evaluando matriz H en la cuadrícula")
res = H_func(q_grid, p_grid)
H_grid = np.empty((NQ2, NQ2, N_points, N_points))
for i in range(NQ2):
    for j in range(NQ2):
        H_grid[i, j] = res[i][j]
del res

#Valores de los niveles phi en la malla
phi_evals = np.array([[phi_n(n, x) for x in qp_list] for n in range(N_c)])

print("Diagonalizando H_grid")
# Diagonalizamos H(q,p) en todos los puntos a la vez
evals, evecs = np.linalg.eigh(np.moveaxis(H_grid, [0, 1], [-2, -1])) 
del H_grid

# ==========================================
# BUCLE PARA CADA TEMPERATURA
# ==========================================
for beta in betas:
    print(f"Calculando matriz exponencial para beta={beta:.4f}...")
    exp_evals = np.exp(-beta * evals) 
    expH_grid = np.moveaxis((evecs * exp_evals[..., None, :]) @ np.swapaxes(evecs.conj(), -1, -2), [-2, -1], [0, 1])
    del exp_evals

    tr_grid = np.trace(expH_grid, axis1=0, axis2=1).real
    Z = np.sum(tr_grid) * Delta**2

    # GRÁFICO DEL MAPA DE CALOR DE LA TRAZA (Solo si beta == 1)
    if np.isclose(beta, 1.0, atol=1e-4):
        print("Generando gráfico de la Traza")
        from matplotlib.colors import LogNorm
        plt.figure(figsize=(8, 6))
        mapa = plt.pcolormesh(q_grid, p_grid, tr_grid, cmap='viridis', shading='auto', norm=LogNorm())
        plt.colorbar(mapa)
        plt.title(rf'Tr$\left(e^{{-\beta H(q,p)}}\right)$, $k={k_val}$')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$p$')
        plt.yticks([-10, -5, 0, 5, 10])
        plt.tight_layout()
        plt.savefig(f"molecular_model/trace_k={k_val}.png", dpi=300)
        plt.close()
    
    #\rho(\xi)
    rho_xi = expH_grid / Z
    del expH_grid, tr_grid

    rho_HCE = np.zeros((dim, dim), dtype='complex128')

    # Paralelización de bloques del HCE
    resultados = Parallel(n_jobs=4, prefer="threads")(
        delayed(calcular_bloque_einsum)(m, mp, rho_xi[m, mp], phi_evals, Delta**2, Nc2)
        for m in range(NQ2) for mp in range(NQ2)
    )

    for m, mp, B_2d in resultados:
        rho_HCE[Nc2*m:Nc2*(m+1), Nc2*mp:Nc2*(mp+1)] = B_2d 
        
    rho_qobj = Qobj(rho_HCE)
    rho_qobj.dims = [[N_Q, N_Q, N_c, N_c], [N_Q, N_Q, N_c, N_c]]

    qsave(rho_qobj, f"molecular_model/HCEs/k_{k_val}/HCE_{beta:.4f}_{N_c}")
    
print("Proceso finalizado correctamente.")
