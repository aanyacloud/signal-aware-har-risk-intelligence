import numpy as np
import matplotlib.pyplot as plt

def analyze_signal(signal):

    plt.figure()
    plt.plot(signal)
    plt.title("Time Domain")
    plt.show()

    diff = np.diff(signal)
    plt.figure()
    plt.plot(diff)
    plt.title("Signal Change")
    plt.show()

    fft = np.abs(np.fft.fft(signal))
    plt.figure()
    plt.plot(fft[:100])
    plt.title("FFT")
    plt.show()