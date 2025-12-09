import numpy as np
from quick_spectrogram import plot_spectrogram
from determine_fo import determine_frequency_offset
import matplotlib.pyplot as plt # type: ignore
from dataclasses import dataclass

def analyze_file(fname: str, start_time_seconds: float, duration_seconds: float, do_plotting: bool = True, do_second_plot: bool = False):

    Fs = 1.8e6

    start_crop=int(start_time_seconds * Fs)
    end_crop=int(start_crop + duration_seconds * Fs)
    
    sample_rate_hz = Fs
    ref_level_db = 0
    window_size = 1024 * 32
    num_windows = 1024 // 8

    ##
    
    data = np.fromfile(fname, dtype=np.float32)
    x = data.view(np.complex64)[start_crop:end_crop]

    ###plot
    ax1, ax2 = plot_spectrogram(x, sample_rate_hz, ref_level_db, window_size, num_windows, "Signal envelope and spectrogram for 1-second segment of recording", do_plot=False, verbose=False)
    ax2.set_xlim(-0.02, 0.02)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    # plt.show()
    plt.savefig("out/img01_initial.png", dpi=300)
    ###

    ###plot
    plt.clf()
    from analyze_v01 import configure_plot
    X = np.fft.fft(x)
    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    X_shift_abs = np.fft.fftshift(np.abs(X))
    plt.plot(frequencies, X_shift_abs)
    configure_plot(plt)
    plt.xlim([-6500, -6410])
    
    max_index = np.argmax(X_shift_abs)
    second_max_index = np.argsort(X_shift_abs)[-2]

    max_freq = frequencies[max_index]
    second_max_freq = frequencies[second_max_index]

    max_freq_amplitude = X_shift_abs[max_index]
    second_max_freq_amplitude = X_shift_abs[second_max_index]

    plt.scatter([max_freq, second_max_freq], [max_freq_amplitude, second_max_freq_amplitude])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Section of spectrogram X(f) with maximum 2 frequencies identified")
    # plt.show()
    plt.savefig("out/img02_freq_offset.png", dpi=300)
    ###

    # Correct frequency offset
    Fo = determine_frequency_offset(x, Fs)
    print(f" * Frequency offset: {Fo:7.2f} Hz")
    t_end = (len(x) - 1) / Fs
    T = np.linspace(0, t_end, len(x))
    x = x * np.exp(-1j * T * Fo * 2 * np.pi)


    ###plot
    plt.clf()
    X = np.fft.fft(x)
    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    X_shift_abs = np.fft.fftshift(np.abs(X))
    plt.plot(frequencies, X_shift_abs)
    configure_plot(plt)
    plt.xlim([-100, 100])
    
    plt.scatter([max_freq, second_max_freq], [max_freq_amplitude, second_max_freq_amplitude])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"Center of spectrogram X(f) after changing frequency by {-Fo:7.2f} Hz")
    # plt.show()
    plt.savefig("out/img03_corrected_freq_offset.png", dpi=300)
    ###


    ###plot
    ax1, ax2 = plot_spectrogram(x, sample_rate_hz, ref_level_db, window_size, num_windows, "Signal envelope and spectrogram after correcting frequency offset", do_plot=False, verbose=False)
    ax2.set_xlim(-0.02, 0.02)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    # plt.show()
    plt.savefig("out/img03a_initial_fo.png", dpi=300)
    ###

    # Step 1: Retrieve the phase of the reference signal (center) by using LPF and comparing to 30 Hz reference
    N = 1000 # decimation to 1.8 kHz --> removes the 9.96-kHz-modulated variable signal
    drop = len(x) % N
    if drop == 0:
        data_to_average = x
    else:
        data_to_average = x[:-drop]

    average = np.average(np.reshape(data_to_average, [-1, N]), axis=1)
    times = np.linspace(0, ((len(data_to_average) - 1) / sample_rate_hz), round(len(data_to_average) / N))

    ###plot
    plt.clf()
    X = np.fft.fft(average)
    Fs_average = Fs / N
    df = Fs_average / len(X)
    frequencies = np.linspace(-Fs_average / 2, Fs_average / 2 - df, len(X))
    plt.plot(frequencies, np.fft.fftshift(X).real, label="real")
    plt.plot(frequencies, np.fft.fftshift(X).imag, label="imag")
    plt.title("Spectrogram X(f) after averaging with N=1000 samples")
    plt.legend(loc="upper right")
    plt.xlim(-100, 100)
    configure_plot(plt)
    # plt.show()
    plt.savefig("out/img04_spectrum_after_averaging.png", dpi=300)
    ###

    reference_phase = np.angle(np.fft.fft(np.abs(average))[30]) + np.pi / 2
    print(f" * Reference phase: {reference_phase*180/np.pi:.1f}\u00b0")

    ###plot
    plt.clf()
    F_ref = 30
    max_ = max(np.abs(average))
    min_ = min(np.abs(average))
    A = (max_ - min_) / 2
    c = (max_ + min_) / 2
    plt.plot(times, A * np.sin(F_ref * 2 * np.pi * times + reference_phase) + c, color="#f55", alpha=1, label="ideal sinusoid")
    plt.plot(times, np.abs(average), color="C0", label="signal with N=1000 averaging")
    
    configure_plot(plt)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.xlabel("time (s)")
    plt.ylabel("sample value")
    plt.title("Signal envelope with 30-Hz ideal sinusoid of detected phase")
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig("out/img05_reference_phase_detection.png", dpi=300)
    ###

    df = Fs / len(x)
    frequencies = np.linspace(-Fs / 2, Fs / 2 - df, len(x))

    FM_filter_min =  9.3e3
    FM_filter_max = 10.7e3

    fft_frequencies = np.fft.ifftshift(frequencies)
    frequency_domain_filter = np.logical_and(fft_frequencies <= FM_filter_max, fft_frequencies >= FM_filter_min)

    x_filtered = np.fft.ifft(np.fft.fft(x) * frequency_domain_filter)

    ###plot
    ax1, ax2 = plot_spectrogram(x_filtered, sample_rate_hz, ref_level_db, window_size, num_windows, "Envelope and spectrogram after filtering for the +9.96 kHz FM signal", do_plot=False, verbose=False)
    ax2.set_xlim(-0.02, 0.02)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    # plt.show()
    plt.savefig("out/img06_spectrum_after_filtering.png", dpi=300)
    ###

    ###plot
    plt.clf()
    plt.plot(T, np.abs(x_filtered))
    plt.xlabel("time (s)")
    plt.ylabel("sample value")
    plt.title("x(t) after filtering for +9.96 kHz")
    # plt.show()
    plt.savefig("out/img07_signal_after_filtering.png", dpi=300)
    ###

    F_fm = 9960 # Hz
    F_fm = 9960 - 480 # Hz
    x_filtered_shifted = x_filtered * np.exp(-1j * T * F_fm * 2 * np.pi)

    x_filtered_shifted_diff = np.diff(x_filtered_shifted, append=[x_filtered_shifted[-1]])

    ###plot
    ax1, ax2 = plot_spectrogram(x_filtered_shifted_diff, sample_rate_hz, ref_level_db, window_size, num_windows, "Envelope and spectrogram after shifting to 480 Hz and differentiating", do_plot=False, verbose=False)
    ax2.set_xlim(-0.02, 0.02)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    # plt.show()
    plt.savefig("out/img08_spectrum_after_shifting.png", dpi=300)
    ###

    ###plot
    plt.clf()
    plt.plot(T, np.abs(x_filtered_shifted_diff))
    plt.xlabel("time (s)")
    plt.ylabel("sample value")
    plt.title("x(t) after shifting to 480 Hz and differentiating")
    plt.savefig("out/img09_signal_after_shifting.png", dpi=300)
    plt.clf()
    ###


    N = 1000 # decimation to 1.8 kHz (again, but this time after filtering and isolating FM signal)
    drop = len(x_filtered_shifted_diff) % N
    if drop == 0:
        data_to_average = x_filtered_shifted_diff
    else:
        data_to_average = x_filtered_shifted_diff[:-drop]
    average_2 = np.average(np.reshape(data_to_average, [-1, N]), axis=1)
    times_2 = np.linspace(0, ((len(data_to_average) - 1) / sample_rate_hz), round(len(data_to_average) / N))

    ###plot
    plt.clf()
    plt.plot(times_2, np.abs(average_2))
    plt.xlabel("time (s)")
    plt.ylabel("sample value")
    plt.title("x(t) shifted to 480 Hz, differentiated, and low-pass filtered")
    plt.savefig("out/img10_signal_after_lpf.png", dpi=300)
    plt.clf()
    ###

    X3 = np.fft.fft(np.abs(average_2))

    variable_phase = np.angle(X3[30]) + np.pi / 2
    print(f" * Variable phase: {variable_phase*180/np.pi:.1f}\u00b0")
    diff = variable_phase - reference_phase
    print(f" * Difference in phase: {(diff*180/np.pi)%360:.1f}\u00b0")

    if do_plotting:
        ax1, ax2 = plot_spectrogram(x, sample_rate_hz, ref_level_db, window_size, num_windows, "Signal envelope and spectrogram, with sinusoids of detected phase overlaid", do_plot=False, verbose=False, red=True)
        F_ref = 30
        F_var = 30
        ref_amp = np.sin(F_ref * 2 * np.pi * times + reference_phase)
        var_freq = np.cos(F_var * 2 * np.pi * times + variable_phase)
        times_shifted = times - (window_size / 2 / Fs)
        times_to_use = times_shifted

        min_, max_ = ax1.get_ylim()
        ax1.plot(times * 1e3, ref_amp * (max_ - min_) / 2 + (max_ + min_) / 2, alpha=0.3, color="r")
        ax2.plot(var_freq * 6e-4 + 0.01, times_to_use * 1e3, color="b", alpha=0.4)
        # ax2.plot(-var_freq * 6e-4 - 0.01, times_to_use * 1e3, color="b", alpha=0.4)
        ax2.set_xlim(-0.012, 0.012)
        ax2.set_ylim(0, duration_seconds*1e3)

        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        plt.savefig("out/img11_big_plot.png", dpi=300)

        if do_second_plot:
            fig = plt.figure(2)
            plt.clf()
            var_freq_shifted = np.sin(F_var * 2 * np.pi * times + variable_phase)
            plt.plot(times, 0.9 * ref_amp + 1, label="Signal amplitude (reference)", color="#f55")
            plt.plot(times, 0.9 * var_freq_shifted - 1, label="Signal frequency (variable)", color="#55f")

            offset = np.pi / 2
            start_ref = ((-reference_phase + offset) % (2 * np.pi)) / (2 * np.pi) / F_ref
            start_var = ((-variable_phase + offset) % (2 * np.pi)) / (2 * np.pi) / F_var

            plt.scatter([start_ref], [1.9], color="#f55")
            plt.scatter([start_var], [-0.1], color="#55f")

            plt.vlines([start_ref], -2.5, 2.5, "#f55", linestyle="--", alpha=0.5)
            plt.vlines([start_var], -2.5, 2.5, "#55f", linestyle="--", alpha=0.5)

            # plt.xlim(0, 0.125)
            plt.xlim(0, 0.07)
            plt.ylim(-2.5, 2.5)
            plt.xlabel("Time (s)")
            plt.ylabel("(arbitrary units)")
            plt.legend(loc="upper right")


            if diff > 0:
                title = f"$\\phi_{{var}} - \\phi_{{ref}}$ = {((variable_phase - reference_phase) * 180 / np.pi) % 360:.1f}\u00b0"
            else:
                title = f"$\\phi_{{var}} - \\phi_{{ref}}$ = {((variable_phase - reference_phase) * 180 / np.pi):.1f}\u00b0 ({((variable_phase - reference_phase) * 180 / np.pi) % 360:.1f}\u00b0)"

            plt.title(title)

            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.savefig("out/img12_phases.png", dpi=300)


        # plt.show()
    
@dataclass
class AnalysisInfo:
    fname: str
    start_time_seconds: float
    duration_seconds: float

def main():

    # fname = "/home/jpiland/iq_recordings/gqrx_20251206_202931_111000000_1800000_fc.raw"

    ### NOTE: gqrx_20251206_202931_111000000_1800000_fc at 18.0 - 21.5 seconds has Morse code for DCA
    ### NOTE: gqrx_20251206_205810_111000000_1800000_fc at 23 seconds has some interesting AM data (perhaps)

    analysis_list = [
        AnalysisInfo(
            fname="/home/jpiland/iq_recordings/gqrx_20251206_211543_111000000_1800000_fc.raw", 
            start_time_seconds=25, 
            duration_seconds=1
        ),
    ]

    import re
    patt = re.compile(r"gqrx_20251206_(\d{2})(\d{2})(\d{2})_111000000_1800000_fc")

    for i, analysis in enumerate(analysis_list):

        h, m, s = patt.search(analysis.fname).groups()

        print(f"=== Evaluation {i + 1} ({int(h) - 17}:{m}pm) ===")
        analyze_file(analysis.fname, analysis.start_time_seconds, analysis.duration_seconds, do_plotting=True, do_second_plot=True)


if __name__ == "__main__":
    main()