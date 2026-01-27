import matplotlib.pyplot as plt
horas = list(range(24))
kWh = [0.5, 0.4, 0.3, 0.3, 0.2, 0.3, 0.6, 1.0, 1.5, 1.3, 1.2, 1.8, 
        2.5, 2.0, 1.7, 1.9, 2.2, 3.0, 3.5, 3.0, 2.5, 1.8, 1.0, 0.6]

plt.figure(figsize=(10, 4))
plt.plot(horas, kWh, 'b-o', linewidth=2, markersize=8)
plt.title("Consumo de Energía en un Día (kWh)", fontsize=14)
plt.xlabel("Hora del Día", fontsize=12)
plt.ylabel("Consumo (kWh)", fontsize=12)
plt.xticks(horas)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

import numpy as np
fft = np.fft.fft(kWh)
n = len(kWh)
freqs = np.fft.fftfreq(n, d=1.0)

plt.figure(figsize=(10, 4))
plt.stem(freqs[:n//2], np.abs(fft[:n//2])/n, 'r-', basefmt=" ", linefmt='r-', markerfmt='ro')
plt.title("FFT del Consumo - Espectro de Frecuencias", fontsize=14)
plt.xlabel("Frecuencia (ciclos/hora)", fontsize=12)
plt.ylabel("Amplitud Normalizada", fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

energia_fft = np.sum(np.abs(fft)**2) / n
frec_dominante = freqs[np.argmax(np.abs(fft[1:n//2])) + 1]

plt.figure(figsize=(8, 3))
plt.barh(["Energía FFT"], [energia_fft], color='teal', alpha=0.6)
plt.axvline(x=frec_dominante, color='red', linestyle='--', label=f'Frec. Dominante: {frec_dominante:.2f}')
plt.title("Energía FFT y Frecuencia Dominante", fontsize=14)
plt.xlabel("Valor", fontsize=12)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()