import numpy as np
import qutip as qt

# Parámetros físicos
N_c = 9
S = 0.5 #Spin
A_val, B_val, C_val, D_val, k_val = 0.01, 1.6, 0.005, 1.0, 1.0


N_Q = int(2*S + 1)

I_c = qt.qeye(N_c)
a = qt.destroy(N_c)
a_q = qt.tensor(a, I_c)
a_d_q = a_q.dag()
a_p = qt.tensor(I_c, a)
a_d_p = a_p.dag()

Q = 1/np.sqrt(2) * (a_q + a_d_q)
P = 1/np.sqrt(2) * (a_p + a_d_p)
Pi_Q = -1j/np.sqrt(2)*(a_q-a_d_q)
Pi_P = -1j/np.sqrt(2)*(a_p-a_d_p)

# 2. Descomposición Espectral de Q
evals, evecs = Q.eigenstates()

E_Q = qt.Qobj(np.zeros((N_c**2, N_c**2)), dims=Q.dims)
T_Q = qt.Qobj(np.zeros((N_c**2, N_c**2)), dims=Q.dims)

# Reconstruimos los operadores aplicando las funciones escalares a los autovalores
for val, vec in zip(evals, evecs):
    if val > 0:
        e_val = A_val * (1 - np.exp(-B_val * val))
    else:
        e_val = A_val * (np.exp(B_val * val) - 1)
        
    # Función T(q)
    t_val = C_val * np.exp(-D_val * val**2)
    
    # Proyector |v><v|
    proj = vec * vec.dag()
    
    E_Q += e_val * proj
    T_Q += t_val * proj


Id_dim = qt.qeye(N_c**2)
Id_dim.dims = [[N_c, N_c], [N_c, N_c]]

H_QP = np.zeros((4, 4, N_c**2, N_c**2), dtype='complex128')

# Término diagonal común
D_QP =  k_val * (P*Pi_Q - Q*Pi_P)

H_QP[0, 0] = (D_QP + 2*E_Q).full()
H_QP[1, 1] = D_QP.full()
H_QP[2, 2] = D_QP.full()
H_QP[3, 3] = (D_QP - 2*E_Q).full()

H_QP[0, 1] = H_QP[1, 0] = T_Q.full()
H_QP[0, 2] = H_QP[2, 0] = T_Q.full()
H_QP[1, 3] = H_QP[3, 1] = T_Q.full()
H_QP[2, 3] = H_QP[3, 2] = T_Q.full()

H_final = H_QP.transpose(0, 2, 1, 3).reshape((N_c**2 * N_Q**2, N_c**2 * N_Q**2))

np.save(f'molecular_model/hamiltonians/hamiltonian_k_{k_val}_{N_c}.npy', H_final)
print("Hamiltoniano construido y guardado con éxito.")