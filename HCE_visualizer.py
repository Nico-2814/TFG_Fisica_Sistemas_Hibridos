import numpy as np
from scipy import special
import math
import qutip as qt
import matplotlib.pyplot as plt



# Parámetros iniciales
q_max = 5
N_points = 2000
N_c = 50
k=0.5

qp_list = np.linspace(-q_max, q_max, N_points)

phi_evals = np.zeros((N_c, N_points))
factor_comun = (np.pi ** -0.25) * np.exp(-qp_list**2 / 2)

HCE=qt.qload(f"molecular_model/HCEs_no_diag/k_{k}/HCE_1.0000_{N_c}")

for n in range(N_c):
    norm = 1.0 / math.sqrt((2**n) * math.factorial(n))
    phi_evals[n, :] = norm * factor_comun * special.eval_hermite(n, qp_list)

print("Phis calculados")

#\Tr_Q(\rho)
rho = HCE.ptrace([2,3]).full().reshape((N_c, N_c, N_c, N_c))

qp_list=np.linspace(-q_max,q_max,N_points)
q_grid, p_grid = np.meshgrid(qp_list, qp_list, indexing='ij')

Phi_q = phi_evals[:, np.newaxis, :] * phi_evals[np.newaxis, :, :] 
Phi_mat = Phi_q.reshape((N_c**2, N_points))

rho_reordered = rho.transpose(0, 2, 1, 3)
del rho
rho_mat = rho_reordered.reshape((N_c**2, N_c**2))
del rho_reordered

prob = Phi_mat.T @ rho_mat @ Phi_mat

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(q_grid, p_grid, prob, cmap='plasma')
ax.set_xlabel("q")
ax.set_ylabel("p")

plt.savefig(f"molecular_model/HCE_repr/classical_{k}.png")

def rho2bloch(ρ):
    sx=(ρ*qt.sigmax()).tr()
    sy=(ρ*qt.sigmay()).tr()
    sz=(ρ*qt.sigmaz()).tr()
    return [sx, sy, sz]


def vectors2points(list):
    listx=[j[0] for j in list]
    listy=[j[1] for j in list]
    listz=[j[2] for j in list]
    return [listx, listy, listz]

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax2 = fig.add_subplot(1,2,2,projection='3d')

b1=qt.Bloch(fig=fig, axes=ax1)
b2=qt.Bloch(fig=fig, axes=ax2)
b1.add_points(rho2bloch(HCE.ptrace([0])))
b2.add_points(rho2bloch(HCE.ptrace([1])))
b1.render()
b2.render()
ax1.set_title("Qubit 1", fontsize = 14, pad = 20)
ax2.set_title("Qubit 2", fontsize = 14, pad = 20)

plt.savefig(f"molecular_model/HCE_repr/quantum_{k}.png")