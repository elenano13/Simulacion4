import numpy as np 
import matplotlib.pyplot as plt 
import numba 
from numba import njit 
from scipy.ndimage import convolve

N = 8
times = 10000 
temperaturas = np.linspace(0.1, 50, 10)  

def inicializar_spins(N):
    init_random = np.random.random((N, N))
    spinsini = np.zeros((N, N))
    spinsini[init_random >= 0.75] = 1
    spinsini[init_random < 0.75] = -1
    return spinsini

def energia(array_spins):
    kern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    arr = -array_spins * convolve(array_spins, kern, mode='wrap')
    return arr.sum()

@njit
def metropolis(spin_arr, times, beta, energia):
    spin_arr = spin_arr.copy()
    movimientos = 0  

    for t in range(times):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]  
        spin_f = spin_i * -1  

        E_i = 0
        E_f = 0
        
        E_i += -spin_i * spin_arr[(x-1) % N, y]
        E_f += -spin_f * spin_arr[(x-1) % N, y]
        E_i += -spin_i * spin_arr[(x+1) % N, y]
        E_f += -spin_f * spin_arr[(x+1) % N, y]
        E_i += -spin_i * spin_arr[x, (y-1) % N]
        E_f += -spin_f * spin_arr[x, (y-1) % N]
        E_i += -spin_i * spin_arr[x, (y+1) % N]
        E_f += -spin_f * spin_arr[x, (y+1) % N]

        dE = E_f - E_i
        if (dE > 0) and (np.random.random() < np.exp(-beta * dE)):
            spin_arr[x, y] = spin_f
            movimientos += 1  # se suma 1 si cambia el estado
        elif dE <= 0:
            spin_arr[x, y] = spin_f
            movimientos += 1  

    acceptance_ratio = movimientos / times
    return spin_arr, acceptance_ratio


razones_aceptacion = []

#se calcula para diferentes temperaturas
for T in temperaturas:
    spinsini = inicializar_spins(N)
    beta = 1 / T  # J/kB = 1
    energia_inicial = energia(spinsini)
    spins_final, razon_aceptacion = metropolis(spinsini, times, beta, energia_inicial)
    razones_aceptacion.append(razon_aceptacion)


plt.figure()
plt.plot(temperaturas, razones_aceptacion, marker='o')
plt.title('Razón de aceptación vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Razón de aceptación')
plt.grid()
plt.show()
