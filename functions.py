import plotly.graph_objects as go
import numpy as np
import streamlit as st


def readSignal(binF, uploaded_file):
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    # timeFlag = file_content[0]  # First line
    # periodicFlag = file_content[1]  # Second line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split(" ")
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
        "Test case failed, your signal have different length from the expected one"
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            "Test case failed, your signal have different values from the expected one"
            return
    "Test case passed successfully"


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
                real_part += real_amplitude * np.cos(exponent) - imag_amplitude * np.sin(exponent)
                imag_part += real_amplitude * np.sin(exponent) + imag_amplitude * np.cos(exponent)

            frequencies.append(round((real_part + imag_part) / N))
        st.write(frequencies)
        return frequencies
