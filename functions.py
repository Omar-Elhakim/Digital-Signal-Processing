import plotly.graph_objects as go
import numpy as np
import streamlit as st
from math import ceil


def readSignal(binF, uploaded_file):
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    # timeFlag = file_content[0]  # First line
    # periodicFlag = file_content[1]  # Second line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split()
        indices.append(float(values[0]) if not binF else values[0])
        amplitudes.append(float(values[1]))

    return indices, amplitudes


def draw(indices, amplitudes):
    fig_cont = go.Figure()
    fig_cont.add_trace(
        go.Scatter(x=indices, y=amplitudes, mode="lines", name="Continuous Signal")
    )

    fig_cont.update_layout(
        title="Continuous Signal",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )

    # fig_cont.show()  # For notebooks
    st.plotly_chart(fig_cont)  # For Streamlit

    # Discrete Signal
    fig_disc = go.Figure()

    fig_disc.add_trace(
        go.Scatter(x=indices, y=amplitudes, mode="markers", name="Discrete Signal")
    )

    # Set titles and labels
    fig_disc.update_layout(
        title="Discrete Signal",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )

    # fig_disc.show()  # For notebooks
    st.plotly_chart(fig_disc)  # For Streamlit


def generate_signal(sineFlag, A, f, fs, theta):
    signal = []
    indices = np.linspace(0, fs - 1, fs)

    if sineFlag:  # if sineFlag is True, then it's a sine wave
        for i in range(fs):
            signal.append(A * np.sin(2 * np.pi * f / fs * i + theta))
    else:
        for i in range(fs):
            signal.append(A * np.cos(2 * np.pi * f / fs * i + theta))
    return indices, signal


def addSignals(firstSignalFile, secondSignalFile):
    # Read the files
    indices1, amplitudes1 = readSignal(0, firstSignalFile)
    indices2, amplitudes2 = readSignal(0, secondSignalFile)
    if len(indices1) != len(indices2):
        print("The two files must be the same size")
        return
    addedAmplitudes = list(x + y for x, y in zip(amplitudes1, amplitudes2))
    draw(indices1, addedAmplitudes)
    return indices1, addedAmplitudes


def subtractSignals(firstSignalFile, secondSignalFile):
    # Read the files
    indices1, amplitudes1 = readSignal(0, firstSignalFile)
    indices2, amplitudes2 = readSignal(0, secondSignalFile)
    if len(indices1) != len(indices2):
        print("The two files must be the same size")
        return
    subtractedAmplitudes = list(x - y for x, y in zip(amplitudes1, amplitudes2))
    draw(indices1, subtractedAmplitudes)
    return indices1, subtractedAmplitudes


def multiplySignals(uploaded_file, constant):
    # Read the file
    indices, amplitudes = readSignal(0, uploaded_file)
    # Multiply each amplitude by the constant
    multipliedAmplitudes = [amp * constant for amp in amplitudes]
    draw(indices, multipliedAmplitudes)
    return indices, multipliedAmplitudes


def squareSignals(uploaded_file):
    # Read the file
    indices, amplitudes = readSignal(0, uploaded_file)
    # Square each amplitude
    squaredAmplitudes = [amp**2 for amp in amplitudes]
    draw(indices, squaredAmplitudes)
    return indices, squaredAmplitudes


def normalizeSignal0(signalFile):
    indices, signal = readSignal(0, signalFile)
    max_value = max(signal)
    min_v = min(signal)
    normalized_signal = list((x - min_v) / (max_value - min_v) for x in signal)
    draw(indices, normalized_signal)
    return indices, normalized_signal


def normalizeSignal1(signalFile):
    indices, signal = readSignal(0, signalFile)
    min_val = min(signal)
    max_val = max(signal)
    signal = list((i - min_val) / (max_val - min_val) * 2 - 1 for i in signal)
    draw(indices, signal)
    return indices, signal


def accumulate(signalFile):
    indices, signal = readSignal(0, signalFile)
    for i in range(len(signal) - 1):
        signal[i + 1] += signal[i]
    draw(indices, signal)
    return indices, signal


def SignalSamplesAreEqual(compareFile, indices, samples):
    expected_indices, expected_samples = readSignal(0, compareFile)

    if len(expected_samples) != len(samples):
        st.write(
            "Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write(
                "Test case failed, your signal have different values from the expected one"
            )
            return
    st.write("Test case passed successfully")


def quantize(noOfLevels, samples):
    minValue = np.min(samples)
    maxValue = np.max(samples)
    delta = (maxValue - minValue) / noOfLevels
    interval_index = []
    quantizedValues = []
    quantizationErrors = []
    encodedLevels = []

    # Quantization
    for sample in samples:
        quantized_level = int((sample - minValue) / delta)
        quantized_level = min(quantized_level, noOfLevels - 1)  # Avoid overflow
        quantized_value = minValue + quantized_level * delta + delta / 2

        # Rounding to three decimal places
        interval_index.append(quantized_level + 1)
        quantizedValues.append(round(quantized_value, 3))
        quantizationErrors.append(round(quantized_value - sample, 3))
        encodedLevels.append(f"{quantized_level:0{int(np.ceil(np.log2(noOfLevels)))}b}")

    return interval_index, encodedLevels, quantizedValues, quantizationErrors


def draw_quantization(indices, original, quantized, error, show_error):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=indices, y=original, mode="lines", name="Original Signal")
    )
    fig.add_trace(
        go.Scatter(x=indices, y=quantized, mode="lines", name="Quantized Signal")
    )

    if show_error:
        fig.add_trace(
            go.Scatter(x=indices, y=error, mode="lines", name="Quantization Error")
        )

    fig.update_layout(
        title="Quantized Signal and Quantization Error",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )

    st.plotly_chart(fig)


def QuantizationTest1(compareFile, Your_EncodedValues, Your_QuantizedValues):
    expectedEncodedValues, expectedQuantizedValues = readSignal(1, compareFile)
    if (len(Your_EncodedValues) != len(expectedEncodedValues)) or (
        len(Your_QuantizedValues) != len(expectedQuantizedValues)
    ):
        st.write(
            "QuantizationTest1 Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            st.write(
                "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one"
            )
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one"
            )
            return
    st.write("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(
    file_name,
    Your_IntervalIndices,
    Your_EncodedValues,
    Your_QuantizedValues,
    Your_SampledError,
):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(" ")) == 4:
                L = line.split(" ")
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (
        len(Your_IntervalIndices) != len(expectedIntervalIndices)
        or len(Your_EncodedValues) != len(expectedEncodedValues)
        or len(Your_QuantizedValues) != len(expectedQuantizedValues)
        or len(Your_SampledError) != len(expectedSampledError)
    ):
        st.write(
            "QuantizationTest2 Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_IntervalIndices)):
        if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
            st.write(
                "QuantizationTest2 Test case failed, your signal have different indicies from the expected one"
            )
            return
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            st.write(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one"
            )
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one"
            )
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest2 Test case failed, your SampledError have different values from the expected one"
            )
            return
    st.write("QuantizationTest2 Test case passed successfully")


def FourierTransform(check, indices, samples, samplingFrequency):
    N = len(samples)

    # DFT
    if check == 0:
        amplitude = []
        angle = []
        for k in range(N):
            real_part = 0
            imag_part = 0
            for n in range(N):
                exponent = (2 * np.pi * k * n) / N
                real_part += samples[n] * np.cos(exponent)
                imag_part -= samples[n] * np.sin(exponent)

            amplitude.append(np.sqrt((real_part * real_part) + (imag_part * imag_part)))
            angle.append(np.arctan2(imag_part, real_part))
        omega = (2 * np.pi) / (N / samplingFrequency)
        newIndices = [omega * i for i in range(1, N + 1)]
        return amplitude, angle, newIndices

    elif check == 1:
        frequencies = []
        for n in range(N):
            real_part = 0
            imag_part = 0
            for k in range(N):
                real_amplitude = indices[k] * np.cos(samples[k])
                imag_amplitude = indices[k] * np.sin(samples[k])

                exponent = (2 * np.pi * k * n) / N
                real_part += real_amplitude * np.cos(
                    exponent
                ) - imag_amplitude * np.sin(exponent)
                imag_part += real_amplitude * np.sin(
                    exponent
                ) + imag_amplitude * np.cos(exponent)

            frequencies.append(round((real_part + imag_part) / N))
        st.write(frequencies)
        return frequencies


def DCT(signal, m):
    N = len(signal)

    y = []

    for k in range(N):
        sum = 0
        for n in range(1, N + 1):
            sum += signal[n - 1] * np.cos(
                np.pi * (2 * (n - 1) - 1) * (2 * k - 1) / (4 * N)
            )

        y.append(np.sqrt(2 / N) * sum)

    with open("DCT_Output.txt", "w") as file:
        for value in y[:m]:
            file.write(f"{value}\n")

    return y


def delay_advance_signal(indices, amplitudes, k):
    indices_shifted = [i + k for i in indices]
    return indices_shifted, amplitudes


def fold_signal(indices, amplitudes):
    indices_folded = [-i for i in reversed(indices)]
    amplitudes_folded = list(reversed(amplitudes))
    return indices_folded, amplitudes_folded


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(" ")) == 2:
                L = line.split(" ")
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    st.write("Current Output Test file is: ")
    if (len(expected_samples) != len(Your_samples)) and (
        len(expected_indices) != len(Your_indices)
    ):
        st.write(
            "Shift_Fold_Signal Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            st.write(
                "Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one"
            )
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write(
                "Shift_Fold_Signal Test case failed, your signal have different values from the expected one"
            )
            return
    st.write("Shift_Fold_Signal Test case passed successfully")


def DerivativeSignal():
    InputSignal = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
    ]
    expectedOutput_first = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    expectedOutput_second = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    """
    Write your Code here:
    Start
    """
    FirstDrev = [
        InputSignal[n] - InputSignal[n - 1] if n else 0
        for n in range(1, len(InputSignal))
    ]
    SecondDrev = [
        InputSignal[n + 1] - 2 * InputSignal[n] + InputSignal[n - 1] if n else 0
        for n in range(1, len(InputSignal) - 1)
    ]

    """
    End
    """

    """
    Testing your Code
    """
    if (len(FirstDrev) != len(expectedOutput_first)) or (
        len(SecondDrev) != len(expectedOutput_second)
    ):
        st.write("mismatch in length")
        return
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            st.write("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            st.write("2nd derivative wrong")
            return
    if first and second:
        st.write("Derivative Test case passed successfully")
    else:
        st.write("Derivative Test case failed")
    return


def compute_moving_average(signal, window_size):
    smoothed_signal = []
    for i in range(len(signal) - (window_size - 1)):
        x = 0
        for j in range(window_size):
            x += signal[j + i]
        smoothed_signal.append(x / window_size)
    return smoothed_signal


def convolve_signals(
    signal1_indices, signal1_samples, signal2_indices, signal2_samples
):
    n = len(signal1_samples)
    m = len(signal2_samples)
    convolved_signal = [0] * (n + m - 1)

    for i in range(len(convolved_signal)):
        for j in range(max(0, i - m + 1), min(n, i + 1)):
            convolved_signal[i] += signal1_samples[j] * signal2_samples[i - j]

    convolved_indices = np.arange(
        signal1_indices[0] + signal2_indices[0],
        signal1_indices[-1] + signal2_indices[-1] + 1,
    )

    return convolved_signal, convolved_indices


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (
        len(expected_indices) != len(Your_indices)
    ):
        st.write(
            "Conv Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            st.write(
                "Conv Test case failed, your signal have different indicies from the expected one"
            )
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write(
                "Conv Test case failed, your signal have different values from the expected one"
            )
            return
    st.write("Conv Test case passed successfully")


def norm_cross_correlation(signal1, signal2):
    N = len(signal1)
    result = []
    for j in range(N):
        sum = 0
        firstSum = 0
        secondSum = 0

        for n in range(N):
            sum += signal1[n] * signal2[j + n - N]
            firstSum += signal1[n] ** 2
            secondSum += signal2[n] ** 2

        numerator = sum / N
        denominator = ((firstSum * secondSum) ** 0.5) / N
        result.append(numerator / denominator)
    return result


def Low_Pass(fc, n):
    if n == 0:
        return 2 * fc
    omega_c = 2 * np.pi * fc
    return 2 * fc * (np.sin(n * omega_c) / (n * omega_c))


def High_Pass(fc, n):
    if n == 0:
        return 1 - (2 * fc)
    omega_c = 2 * np.pi * fc
    return -2 * fc * (np.sin(n * omega_c) / (n * omega_c))


def Band_Pass(f, n):
    if n == 0:
        return 2 * (f[1] - f[0])
    omega_1 = 2 * np.pi * f[0]
    omega_2 = 2 * np.pi * f[1]
    term1 = 2 * f[1] * (np.sin(n * omega_2) / (n * omega_2))
    term2 = 2 * f[0] * (np.sin(n * omega_1) / (n * omega_1))
    return term1 - term2


def Band_Stop(f, n):
    if n == 0:
        return 1 - 2 * (f[1] - f[0])
    omega_1 = 2 * np.pi * f[0]
    omega_2 = 2 * np.pi * f[1]
    term1 = 2 * f[0] * (np.sin(n * omega_1) / (n * omega_1))
    term2 = 2 * f[1] * (np.sin(n * omega_2) / (n * omega_2))
    return term1 - term2


FilterType = {
    "Low_Pass": Low_Pass,
    "High_Pass": High_Pass,
    "Band_Pass": Band_Pass,
    "Band_Stop": Band_Stop,
}


def RectangularWindow():
    return 1


def HanningWindow(n, N):
    if n == 0:
        return 1
    return 0.5 + 0.5 * np.cos((2 * np.pi * n) / N)


def HammingWindow(n, N):
    if n == 0:
        return 1
    return 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)


def BlackmanWindow(n, N):
    if n == 0:
        return 1
    term2 = 0.5 * np.cos((2 * np.pi * n) / (N - 1))
    term3 = 0.08 * np.cos((4 * np.pi * n) / (N - 1))
    return 0.42 + term2 + term3


WindowType = {
    "RectangularWindow": RectangularWindow,
    "HanningWindow": HanningWindow,
    "HammingWindow": HammingWindow,
    "BlackmanWindow": BlackmanWindow,
}


def FIR_Filter(filterType, Fs, StopBandAttenuation, F, TransitionWidth):
    if StopBandAttenuation <= 21:
        window = "RectangularWindow"
        N = 0.9 / (TransitionWidth / Fs)
    elif StopBandAttenuation <= 44:
        window = "HanningWindow"
        N = 3.1 / (TransitionWidth / Fs)
    elif StopBandAttenuation <= 53:
        window = "HammingWindow"
        N = 3.3 / (TransitionWidth / Fs)
    elif StopBandAttenuation <= 74:
        window = "BlackmanWindow"
        N = 5.5 / (TransitionWidth / Fs)
    N = ceil(N)
    if N % 2 == 0:
        N += 1
    if filterType == "Low_Pass":
        F += TransitionWidth / 2
        F /= Fs
    elif filterType == "High_Pass":
        F -= TransitionWidth / 2
        F /= Fs
    elif filterType == "Band_Pass":
        f1, f2 = F
        f1 -= TransitionWidth / 2
        f2 += TransitionWidth / 2
        f1 /= Fs
        f2 /= Fs
        F = f1, f2
    elif filterType == "Band_Stop":
        f1, f2 = F
        f1 += TransitionWidth / 2
        f2 -= TransitionWidth / 2
        f1 /= Fs
        f2 /= Fs
        F = f1, f2

    h = []
    for n in range(ceil(N / 2)):
        h.append(FilterType[filterType](F, n) * WindowType[window](n, N))
    hReversed = h.copy()
    hReversed.reverse()
    hReversed.pop()
    return hReversed + h
