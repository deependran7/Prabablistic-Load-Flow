import numpy as np
import cmath

# Y-bus calculation
def calculate_ybus(bus, branch,nbus):
    Ybus = np.zeros((nbus, nbus), dtype=complex)
    for br in branch:
        fbus = int(br[0]) - 1
        tbus = int(br[1]) - 1
        r = br[2]
        x = br[3]
        b = br[4]
        z = r + 1j * x
        y = 1 / z
        b_shunt = 1j * b / 2
        Ybus[fbus, tbus] -= y
        Ybus[tbus, fbus] -= y
        Ybus[fbus, fbus] += y + b_shunt
        Ybus[tbus, tbus] += y + b_shunt
    for i in range(nbus):
        Ybus[i, i] += bus[i, 4] + 1j * bus[i, 5]
    return Ybus

# Power calculation
def calculate_power(V, Ybus, Pg, Qg, Pd, Qd,nbus):
    S_calc = np.zeros(nbus, dtype=complex)
    for i in range(nbus):
        for j in range(nbus):
            S_calc[i] += V[i] * np.conj(Ybus[i, j] * V[j])
    P_calc = np.real(S_calc)
    Q_calc = np.imag(S_calc)
    delta_P = Pg - Pd - P_calc
    delta_Q = Qg - Qd - Q_calc
    return delta_P, delta_Q

# Jacobian calculation
def calculate_jacobian(V, Ybus, bus_types,nbus):
    n = nbus
    npq = len([i for i in range(nbus) if bus_types[i] == 1])
    n_vars = nbus - 1 + npq
    J = np.zeros((n_vars, n_vars))
    Vmag = np.abs(V)
    Vang = np.angle(V)
    G = np.real(Ybus)
    B = np.imag(Ybus)
    
    pq_buses = [i for i in range(nbus) if bus_types[i] == 1]
    non_slack = [i for i in range(nbus) if bus_types[i] != 3]
    
    row_idx = 0
    delta_idx = {i: k for k, i in enumerate(non_slack)}
    V_idx = {i: k + len(non_slack) for k, i in enumerate(pq_buses)}
    
    for i in non_slack:
        for j in non_slack:
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    if k != i:
                        sum_term += Vmag[i] * Vmag[k] * (G[i, k] * np.sin(Vang[i] - Vang[k]) - B[i, k] * np.cos(Vang[i] - Vang[k]))
                J[row_idx, delta_idx[j]] = -sum_term
            else:
                J[row_idx, delta_idx[j]] = Vmag[i] * Vmag[j] * (G[i, j] * np.sin(Vang[i] - Vang[j]) - B[i, j] * np.cos(Vang[i] - Vang[j]))
        if bus_types[i] == 1:
            for j in pq_buses:
                if i == j:
                    sum_term = 0
                    for k in range(nbus):
                        if k != i:
                            sum_term += Vmag[k] * (G[i, k] * np.cos(Vang[i] - Vang[k]) + B[i, k] * np.sin(Vang[i] - Vang[k]))
                    J[row_idx, V_idx[j]] = 2 * Vmag[i] * G[i, i] + sum_term
                else:
                    J[row_idx, V_idx[j]] = Vmag[i] * (G[i, j] * np.cos(Vang[i] - Vang[j]) + B[i, j] * np.sin(Vang[i] - Vang[j]))
        row_idx += 1
    
    for i in pq_buses:
        for j in non_slack:
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    if k != i:
                        sum_term += Vmag[i] * Vmag[k] * (G[i, k] * np.cos(Vang[i] - Vang[k]) + B[i, k] * np.sin(Vang[i] - Vang[k]))
                J[row_idx, delta_idx[j]] = sum_term
            else:
                J[row_idx, delta_idx[j]] = Vmag[i] * Vmag[j] * (G[i, j] * np.cos(Vang[i] - Vang[j]) + B[i, j] * np.sin(Vang[i] - Vang[j]))
        for j in pq_buses:
            if i == j:
                sum_term = 0
                for k in range(nbus):
                    if k != i:
                        sum_term += Vmag[k] * (G[i, k] * np.sin(Vang[i] - Vang[k]) - B[i, k] * np.cos(Vang[i] - Vang[k]))
                J[row_idx, V_idx[j]] = -2 * Vmag[i] * B[i, i] + sum_term
            else:
                J[row_idx, V_idx[j]] = Vmag[i] * (G[i, j] * np.sin(Vang[i] - Vang[j]) - B[i, j] * np.cos(Vang[i] - Vang[j]))
        row_idx += 1
    
    return J

# Newton-Raphson solver (returns operating condition for linearization)
def newton_raphson(nbus,bus, gen, branch,Pg_pu,Qg_pu,Pd_pu,Qd_pu, tol=1e-4, max_iter=20):
    slack_bus = np.where(bus[:, 1] == 3)[0][0] + 1
    bus_types = bus[:, 1].astype(int)
    V = np.ones(nbus, dtype=complex)
    V[slack_bus-1] = 1.0
    for g in gen:
        bus_idx = int(g[0]) - 1
        if bus_types[bus_idx] == 2:
            V[bus_idx] = g[5] * np.exp(1j * np.angle(V[bus_idx]))
    
    Ybus = calculate_ybus(bus, branch,nbus)
    iteration = 0
    while iteration < max_iter:
        delta_P, delta_Q = calculate_power(V, Ybus, Pg_pu, Qg_pu, Pd_pu, Qd_pu,nbus)
        mismatches = []
        non_slack = [i for i in range(nbus) if bus_types[i] != 3]
        pq_buses = [i for i in range(nbus) if bus_types[i] == 1]
        for i in non_slack:
            mismatches.append(delta_P[i])
        for i in pq_buses:
            mismatches.append(delta_Q[i])
        mismatches = np.array(mismatches)
        
        if np.max(np.abs(mismatches)) < tol:
            print(f"Converged after {iteration} iterations")
            break
        
        J = calculate_jacobian(V, Ybus, bus_types,nbus)
        try:
            delta_x = np.linalg.solve(J, mismatches)
        except np.linalg.LinAlgError:
            print("Jacobian singular, cannot proceed")
            return None, None
        
        delta_idx = 0
        for i in non_slack:
            V[i] = cmath.rect(np.abs(V[i]), np.angle(V[i]) + delta_x[delta_idx])
            delta_idx += 1
        for i in pq_buses:
            V[i] = cmath.rect(np.abs(V[i]) + delta_x[delta_idx], np.angle(V[i]))
            delta_idx += 1
        
        iteration += 1
        if iteration == max_iter:
            print("Did not converge within maximum iterations")
    
    delta_P, delta_Q = calculate_power(V, Ybus, Pg_pu, Qg_pu, Pd_pu, Qd_pu,nbus)
    S_inj = Pg_pu - Pd_pu + 1j * (Qg_pu - Qd_pu)
    
    return V, S_inj, J

