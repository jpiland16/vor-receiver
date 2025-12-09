import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt # type: ignore
from analyze_v01 import configure_plot

def determine_frequency_offset(x: npt.NDArray, Fs: float) -> float:

    X = np.fft.fft(x)
    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    X_shift_abs = np.fft.fftshift(np.abs(X))
    
    max_index = np.argmax(X_shift_abs)
    second_max_index = np.argsort(X_shift_abs)[-2]

    max_freq = frequencies[max_index]
    second_max_freq = frequencies[second_max_index]

    max_freq_amplitude = X_shift_abs[max_index]
    second_max_freq_amplitude = X_shift_abs[second_max_index]

    # Interpolate between the two maximum frequencies based on their relative amplitudes
    total = max_freq_amplitude + second_max_freq_amplitude
    interpolated_offset = max_freq_amplitude / total * max_freq + second_max_freq_amplitude / total * second_max_freq
    
    return interpolated_offset

def remove_frequency_offset(x: npt.NDArray):
    """
    Note: the sampling frequency is essentially unnecessary for this calculation
    """

    Fs = 1

    Fo = determine_frequency_offset(x, Fs)

    t_end = (len(x) - 1) / Fs
    T = np.linspace(0, t_end, len(x))

    return x * np.exp(-1j * T * Fo * 2 * np.pi)

def main():

    # TARGET: 6,549.43

    fname = "/home/jpiland/iq_recordings/gqrx_20251206_202931_111000000_1800000_fc.raw"

    Fs = 1.8e6

    start_crop=int(10 * Fs)
    end_crop=int(start_crop + 1 * Fs)

    data = np.fromfile(fname, dtype=np.float32)
    x = data.view(np.complex64)[start_crop:end_crop]

    X = np.fft.fft(x)
    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    X_shift_abs = np.fft.fftshift(np.abs(X))
    plt.plot(frequencies, X_shift_abs)
    configure_plot(plt)
    plt.xlim([-6590, -6510])
    
    max_index = np.argmax(X_shift_abs)
    second_max_index = np.argsort(X_shift_abs)[-2]

    max_freq = frequencies[max_index]
    second_max_freq = frequencies[second_max_index]

    max_freq_amplitude = X_shift_abs[max_index]
    second_max_freq_amplitude = X_shift_abs[second_max_index]

    # print(max_freq, second_max_freq)
    # print(max_freq_amplitude, second_max_freq_amplitude)

    # Interpolate between the two maximum frequencies based on their relative amplitudes
    total = max_freq_amplitude + second_max_freq_amplitude
    interpolated_offset = max_freq_amplitude / total * max_freq + second_max_freq_amplitude / total * second_max_freq
    print(interpolated_offset)

def main2():

    fname = "/home/jpiland/iq_recordings/gqrx_20251206_202931_111000000_1800000_fc.raw"

    Fs = 1.8e6

    start_crop=int(10 * Fs)
    end_crop=int(start_crop + 1 * Fs)

    data = np.fromfile(fname, dtype=np.float32)
    x = data.view(np.complex64)[start_crop:end_crop]
    x = remove_frequency_offset(x)
    
    X = np.fft.fft(x)
    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    X_shift_abs = np.fft.fftshift(np.abs(X))
    plt.plot(frequencies, X_shift_abs)
    configure_plot(plt)

    # plt.xlim([-6590, -6510])
    plt.xlim([-50, 50])
    plt.show()
    


if __name__ == "__main__":
    main2()