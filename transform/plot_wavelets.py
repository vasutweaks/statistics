import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def morlet_wavelet(t, w=5.0):
    """
    Manual implementation of the Morlet wavelet.
    psi(t) = exp(-t^2 / 2) * cos(w * t)
    """
    return np.exp(-t**2 / 2) * np.cos(w * t)

def mexican_hat_wavelet(t, sigma=1.0):
    """
    Manual implementation of the Mexican Hat (Ricker) wavelet.
    psi(t) = (1 - (t/sigma)^2) * exp(-t^2 / (2 * sigma^2))
    Normalized version:
    psi(t) = (2 / (sqrt(3*sigma) * pi**0.25)) * (1 - (t/sigma)**2) * exp(-t**2 / (2*sigma**2))
    """
    prefactor = 2 / (np.sqrt(3 * sigma) * np.pi**0.25)
    return prefactor * (1 - (t/sigma)**2) * np.exp(-t**2 / (2 * sigma**2))

def paul_wavelet(t, m=4):
    """
    Implementation of the Paul wavelet.
    psi(t) = (2^m * m! * i^m) / (sqrt(pi * (2m)!)) * (1 - i*t)**(-(m+1))
    """
    prefactor = (2**m * factorial(m) * 1j**m) / np.sqrt(np.pi * factorial(2*m))
    return prefactor * (1 - 1j*t)**(-(m+1))

def plot_wavelets():
    # Time axis
    t = np.linspace(-5, 5, 1000)
    
    plt.figure(figsize=(6, 10))
    
    # 1. Morlet Wavelet
    morlet = morlet_wavelet(t)
    plt.subplot(3, 1, 1)
    plt.plot(t, morlet, label='Morlet (w=5.0)', color='blue')
    plt.title('Morlet Wavelet')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 2. Mexican Hat (Ricker) Wavelet
    mexican_hat = mexican_hat_wavelet(t)
    plt.subplot(3, 1, 2)
    plt.plot(t, mexican_hat, label=r'Mexican Hat ($\sigma=1.0$)', color='red')
    plt.title('Mexican Hat (Ricker) Wavelet')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 3. Paul Wavelet
    paul = paul_wavelet(t, m=4)
    plt.subplot(3, 1, 3)
    plt.plot(t, paul.real, label='Real part', color='green')
    plt.plot(t, paul.imag, '--', label='Imaginary part', color='orange', alpha=0.6)
    plt.title('Paul Wavelet (m=4)')
    plt.xlabel('Time (t)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('wavelets_plot.png')
    print("Plot successfully saved as wavelets_plot.png")
    # plt.show() # Disabled to avoid interactive issues in non-GUI environment

if __name__ == "__main__":
    plot_wavelets()
