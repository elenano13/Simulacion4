import numpy as np 
import matplotlib.pyplot as plt 
import numba 
from numba import njit 
from scipy.ndimage import convolve

N = 8
Temperatura = 1  
times = 10000 

def inicializar_spins(N):
    init_random = np.random.random((N, N))
    spinsini = np.zeros((N, N))
    spinsini[init_random >= 0.75] = 1
    spinsini[init_random < 0.75] = -1
    return spinsini

spinsini = inicializar_spins(N)

# Visualizar configuración inicial 
plt.imshow(spinsini, cmap='summer') 
plt.title('Configuración inicial') 
plt.colorbar() 
plt.show()

def energia(array_spins):
    # Se utiliza wrap para simular las condiciones periódicas del lattice
    kern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    arr = -array_spins * convolve(array_spins, kern, mode='wrap')
    return arr.sum()

@njit
def metropolis(spin_arr, times, beta, energia):
    spin_arr = spin_arr.copy()
    spinsnetos = np.zeros(times-1)
    energianeta = np.zeros(times-1)
    
    for t in range(0, times-1):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]  
        spin_f = spin_i * -1  

        E_i = 0
        E_f = 0
        
        # Calcular la energía
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
        elif dE <= 0:
            spin_arr[x, y] = spin_f

        spinsnetos[t] = spin_arr.sum()  
        energianeta[t] = energia  

    return spinsnetos, energianeta, spin_arr

#beta para la t
beta = 1 / Temperatura  # J/kB = 1
energia_inicial = energia(spinsini)
spins, energias, spinsfin = metropolis(spinsini, times, beta, energia_inicial)

# Calcular la magnetización
pasos_equi = int(0.4 * len(spins))
magnetizacion_promedio = np.mean(spins[pasos_equi:])

# configuración final
plt.figure()
plt.imshow(spinsfin, cmap='summer')  


numerodespins = N * N
positivos = np.sum(spinsfin == 1)
negativos = np.sum(spinsfin == -1)
porcentajepositivo = (positivos / numerodespins) * 100
porcentajenegativo = (negativos / numerodespins) * 100

plt.title(f'Configuración final a T = {Temperatura:.2f}\n'
          f'Porcentaje de spins +1: {porcentajepositivo:.2f}%, '
          f'Porcentaje de spins -1: {porcentajenegativo:.2f}%')
plt.colorbar()
plt.show()

print(f"Resultados a T = {Temperatura:.2f}:")
print(f"Magnetización promedio: {magnetizacion_promedio:.4f}")

# cambio en el signo de la magnetización
magnetizacion_t = spins

# magnetización en diferentes tiempos 
print("\nValores de magnetización a lo largo del tiempo:")
for i in range(0, len(magnetizacion_t), 1000):  
    print(f"Paso {i}: Magnetización = {magnetizacion_t[i]}")

# identificar donde hay cambios de signo en la magnetizacion
cambios_signo = np.where(np.diff(np.sign(magnetizacion_t)))[0]

# magnetizacion vs t
plt.figure()
plt.plot(magnetizacion_t, label='Magnetización')
plt.scatter(cambios_signo, magnetizacion_t[cambios_signo], color='red', label='Cambio de signo', zorder=5)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title('Magnetización a lo largo del tiempo')
plt.xlabel('Tiempo')
plt.ylabel('Magnetización')
plt.legend()
plt.grid()
plt.show()
