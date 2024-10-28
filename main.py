import plotly.graph_objects as go
import numpy as np
import streamlit as st

from signals.task3.QuanTest1 import QuantizationTest1


def main():
    # Main page
    st.title("Digital Signal Processing")

    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox(
        "Select an option",
        ["Arithmetic Operations", "Signal Reading", "Signal Generation", "Quantization"],
        index=None,
    )

    # Read and draw a Signal from a file
    if menu == "Signal Reading":
        st.header("Read Signal Samples")

        uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
        read_button = st.button("Read Signal")

        if read_button:
            if uploaded_file is not None:
                indices, amplitudes = readSignal(uploaded_file)
                draw(indices, amplitudes)

    # Choose your inputs and generate the signal then draw it
    elif menu == "Signal Generation":
        st.header("Signal Generation")

        signal_type = st.selectbox("Choose Signal Type", ["Sine Wave", "Cosine Wave"])
        amplitude = st.number_input("Amplitude (A)", min_value=0.0, value=1.0)
        analogFreq = st.number_input("Analog Frequency (Hz)", min_value=0.0, value=1.0)
        samplingFreq = int(
            st.number_input("Sampling Frequency (Hz)", min_value=0.0, value=10.0)
        )
        phaseShift = st.number_input("Phase Shift ", min_value=0.0, value=0.0)

        generate_button = st.button("Generate Signal")
        if generate_button:
            sinFlag = signal_type == "Sine Wave"  # True for sine, False for cosine
            if samplingFreq < 2 * analogFreq:
                st.error(
                    "Sampling frequency must be at least twice the analog frequency ."
                )
            else:
                indices1, values1 = generate_signal(
                    sinFlag, amplitude, analogFreq, samplingFreq, phaseShift
                )
                draw(indices1, values1)

        # Choose your inputs and quantize the samples
    elif menu == "Quantization":
        st.header("Quantization")

        uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")

        quant_type = st.radio("Quantization Type", ["Levels", "Bits"])
        no_of_levels = st.number_input("Number of Levels:", min_value=1, value=2) if quant_type == "Levels" else (
                1 << st.number_input("Bits:", min_value=1, value=1))

        show_interval_index = st.checkbox("Interval Index", value=True)
        show_encoded = st.checkbox("Encoded", value=True)
        show_quantized = st.checkbox("Quantized", value=True)
        show_error = st.checkbox("Error", value=True)

        if st.button("Quantize Signal"):
            if uploaded_file is not None:
                indices, amplitudes = readSignal(uploaded_file)
                interval_index, encoded, quantized, error = quantize(no_of_levels, amplitudes)

                # Display chosen outputs
                if show_interval_index:
                    st.write("Interval Index:", interval_index)
                if show_encoded:
                    st.write("Encoded Signal:", encoded)
                if show_quantized:
                    st.write("Quantized Signal:", quantized)
                if show_error:
                    st.write("Quantization Error:", error)

                if show_quantized or show_error:
                    draw_quantization(indices, amplitudes, quantized, error, show_error)

                comparingFile = st.file_uploader("Upload the output txt file", type="txt")
                if comparingFile and encoded and quantized:
                    QuantizationTest1(comparingFile, encoded, quantized)

    elif menu == "Arithmetic Operations":

        if "button_pressed" not in st.session_state:
            st.session_state["button_pressed"] = None

        st.write("### Select an option:")
        (
            add,
            sub,
            mult,
            square,
            norm,
            accum,
        ) = st.columns([1, 1.5, 1.5, 1.2, 1.5, 1.6])

        with add:
            if st.button("Add"):
                st.session_state["button_pressed"] = "button_1"

        with sub:
            if st.button("Subtract"):
                st.session_state["button_pressed"] = "button_2"

        with mult:
            if st.button("Multiply"):
                st.session_state["button_pressed"] = "button_3"

        with square:
            if st.button("Square"):
                st.session_state["button_pressed"] = "button_4"

        with norm:
            if st.button("Normalize"):
                st.session_state["button_pressed"] = "button_6"

        with accum:
            if st.button("Accumulate"):
                st.session_state["button_pressed"] = "button_7"
        if st.session_state["button_pressed"] is None:
            pass

        # Display for Add button
        if st.session_state["button_pressed"] == "button_1":
            uploaded_file1 = st.file_uploader(
                "Upload the first signal txt file", type="txt"
            )

            uploaded_file2 = st.file_uploader(
                "Upload the second signal txt file", type="txt"
            )

            if uploaded_file1 and uploaded_file2:
                indecis, samples = addSignals(uploaded_file1, uploaded_file2)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indecis:
                SignalSamplesAreEqual(comparingFile, indecis, samples)

        # Display for Sub button
        elif st.session_state["button_pressed"] == "button_2":
            uploaded_file1 = st.file_uploader(
                "Upload the first signal txt file", type="txt"
            )

            uploaded_file2 = st.file_uploader(
                "Upload the second signal txt file", type="txt"
            )
            if uploaded_file1 and uploaded_file2:
                indecis, samples = subtractSignals(uploaded_file1, uploaded_file2)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indecis:
                SignalSamplesAreEqual(comparingFile, indecis, samples)

        # Display for Multiply button
        elif st.session_state["button_pressed"] == "button_3":
            uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
            constant = st.number_input("Enter a constant", min_value=-1.0, value=1.0)
            if uploaded_file:
                indecis, samples = multiplySignals(uploaded_file, constant)
                comparingFile = st.file_uploader(
                    "Upload the signal compare txt file", type="txt"
                )
                if comparingFile and samples and indecis:
                    SignalSamplesAreEqual(comparingFile, indecis, samples)

        # Display for Square button
        elif st.session_state["button_pressed"] == "button_4":
            uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
            if uploaded_file:
                indecis, samples = squareSignals(uploaded_file)
            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indecis:
                SignalSamplesAreEqual(comparingFile, indecis, samples)

        # Display for Normalize button
        elif st.session_state["button_pressed"] == "button_6":
            uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
            choice = st.radio("Normalize to:", ("[-1,1]", "[0,1]"), index=0)
            if uploaded_file:
                if choice == "[-1,1]":
                    indecis, samples = normalizeSignal1(uploaded_file)
                else:
                    indecis, samples = normalizeSignal0(uploaded_file)

                comparingFile = st.file_uploader(
                    "Upload the signal compare txt file", type="txt"
                )
                if comparingFile and samples and indecis:
                    SignalSamplesAreEqual(comparingFile, indecis, samples)

        # Display for accumulate button
        elif st.session_state["button_pressed"] == "button_7":
            uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
            if uploaded_file:
                indecis, samples = accumulate(uploaded_file)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indecis:
                SignalSamplesAreEqual(comparingFile, indecis, samples)

    else:
        st.markdown("*Choose an option from the side menu!*")


def readSignal(uploaded_file):
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    # timeFlag = file_content[0]  # First line
    # periodicFlag = file_content[1]  # Second line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3: 3 + nOfSamples]:
        values = line.strip().split(" ")
        indices.append(int(values[0]))
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
    indices1, amplitudes1 = readSignal(firstSignalFile)
    indices2, amplitudes2 = readSignal(secondSignalFile)
    if len(indices1) != len(indices2):
        print("The two files must be the same size")
        return
    addedAmplitudes = list(x + y for x, y in zip(amplitudes1, amplitudes2))
    draw(indices1, addedAmplitudes)
    return indices1, addedAmplitudes


def subtractSignals(firstSignalFile, secondSignalFile):
    # Read the files
    indices1, amplitudes1 = readSignal(firstSignalFile)
    indices2, amplitudes2 = readSignal(secondSignalFile)
    if len(indices1) != len(indices2):
        print("The two files must be the same size")
        return
    subtractedAmplitudes = list(x - y for x, y in zip(amplitudes1, amplitudes2))
    draw(indices1, subtractedAmplitudes)
    return indices1, subtractedAmplitudes


def multiplySignals(uploaded_file, constant):
    # Read the file
    indices, amplitudes = readSignal(uploaded_file)
    # Multiply each amplitude by the constant
    multipliedAmplitudes = [amp * constant for amp in amplitudes]
    draw(indices, multipliedAmplitudes)
    return indices, multipliedAmplitudes


def squareSignals(uploaded_file):
    # Read the file
    indices, amplitudes = readSignal(uploaded_file)
    # Square each amplitude
    squaredAmplitudes = [amp ** 2 for amp in amplitudes]
    draw(indices, squaredAmplitudes)
    return indices, squaredAmplitudes


def normalizeSignal0(signalFile):
    indices, signal = readSignal(signalFile)
    max_value = max(signal)
    min_v = min(signal)
    normalized_signal = list((x - min_v) / (max_value - min_v) for x in signal)
    draw(indices, normalized_signal)
    return indices, normalized_signal


def normalizeSignal1(signalFile):
    indices, signal = readSignal(signalFile)
    min_val = min(signal)
    max_val = max(signal)
    signal = list((i - min_val) / (max_val - min_val) * 2 - 1 for i in signal)
    draw(indices, signal)
    return indices, signal


def accumulate(signalFile):
    indices, signal = readSignal(signalFile)
    for i in range(len(signal) - 1):
        signal[i + 1] += signal[i]
    draw(indices, signal)
    return indices, signal


def SignalSamplesAreEqual(compareFile, indices, samples):
    expected_indices, expected_samples = readSignal(compareFile)

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
    fig.add_trace(go.Scatter(x=indices, y=original, mode="lines", name="Original Signal"))
    fig.add_trace(go.Scatter(x=indices, y=quantized, mode="lines", name="Quantized Signal"))

    if show_error:
        fig.add_trace(go.Scatter(x=indices, y=error, mode="lines", name="Quantization Error"))

    fig.update_layout(
        title="Quantized Signal and Quantization Error",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )

    st.plotly_chart(fig)


main()
