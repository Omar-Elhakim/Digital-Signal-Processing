import streamlit as st

from functions import *

# Main page
st.title("Digital Signal Processing")

st.sidebar.title("Menu")
menu = st.sidebar.selectbox(
    "Select an option",
    [
        "Arithmetic Operations",
        "Signal Reading",
        "Signal Generation",
        "Quantization",
        "Frequency Domain",
        "FIR Filters",
        "Time Domain",
        "Sharpening",
        "smoothing",
        "Convolution",
        "Correlation",
    ],
    index=None,
)

# Read and draw a Signal from a file
if menu == "Signal Reading":
    st.header("Read Signal Samples")

    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
    read_button = st.button("Read Signal")

    if read_button:
        if uploaded_file is not None:
            indices, amplitudes = readSignal(0, uploaded_file)
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
            st.error("Sampling frequency must be at least twice the analog frequency .")
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
    no_of_levels = (
        st.number_input("Number of Levels:", min_value=1, value=2)
        if quant_type == "Levels"
        else (1 << st.number_input("Bits:", min_value=1, value=1))
    )

    show_interval_index = st.checkbox("Interval Index", value=True)
    show_encoded = st.checkbox("Encoded", value=True)
    show_quantized = st.checkbox("Quantized", value=True)
    show_error = st.checkbox("Error", value=True)

    # if st.button("Quantize Signal"):
    if uploaded_file is not None:
        indices, amplitudes = readSignal(0, uploaded_file)
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

        draw_quantization(indices, amplitudes, quantized, error, 0)

    comparingFile = st.file_uploader("Upload the output txt file", type="txt")
    if comparingFile and encoded and quantized:
        if uploaded_file.name == "Quan1_input.txt":
            QuantizationTest1(comparingFile, encoded, quantized)
        else:
            QuantizationTest2(
                f"signals/task3/{comparingFile.name}",
                interval_index,
                encoded,
                quantized,
                error,
            )

elif menu == "Frequency Domain":
    st.header("Frequency Domain")
    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")

    check = st.radio(
        "Choose Transform:",
        (0, 1, 2),
        format_func=lambda x: "DFT" if x == 0 else "IDFT" if x == 1 else "DCT",
    )
    if uploaded_file:
        indices, amplitudes = readSignal(0, uploaded_file)

    if check == 0:  # DFT requires sampling frequency
        samplingFrequency = st.number_input(
            "Enter sampling frequency in Hz", min_value=1
        )
    elif check == 2:  # DCT
        m = st.number_input("Enter number of coefficients :", value=0, step=1)
        x = DCT(amplitudes, m)

    if st.button("Perform Transform"):

        if check == 0:  # DFT
            amp, angles, newIndices = FourierTransform(
                check, indices, amplitudes, samplingFrequency
            )
            draw(newIndices, amp)
            draw(newIndices, angles)
            amp
            angles

        elif check == 1:  # IDFT
            FourierTransform(1, indices, amplitudes, 0)

        elif check == 2:  # DCT
            x

    if check == 2:  # DCT
        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile:
            SignalSamplesAreEqual(comparingFile, indices, x)

    if st.button("Remove DC component"):  # DFT
        amp, angles, newIndices = FourierTransform(
            check, indices, amplitudes, samplingFrequency
        )
        amp[0] = 0
        draw(newIndices, amp)
        draw(newIndices, angles)
        amp
        angles

elif menu == "Time Domain":
    st.header("Time Domain Operations")

    operation = st.selectbox(
        "Choose Operation",
        [
            "Delay/Advance Signal by k Steps",
            "Fold Signal",
            "Delay/Advance Folded Signal by k Steps",
            "remove DC",
        ],
    )

    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
    if uploaded_file:
        indices, amplitudes = readSignal(0, uploaded_file)
        indices_folded, amplitudes_folded = fold_signal(indices, amplitudes)
        amplitudes_shifted = None
        indices_shifted = None

        if operation == "Delay/Advance Signal by k Steps":
            k = st.number_input(
                "Enter k (positive for delay, negative for advance):", value=0, step=1
            )
            if st.button("Apply"):
                indices_shifted, amplitudes_shifted = delay_advance_signal(
                    indices, amplitudes, k
                )
                draw(indices_shifted, amplitudes_shifted)

        elif operation == "Fold Signal":
            if st.button("Apply"):
                draw(indices_folded, amplitudes_folded)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and amplitudes_folded and indices_folded:
                SignalSamplesAreEqual(comparingFile, indices_folded, amplitudes_folded)

        elif operation == "Delay/Advance Folded Signal by k Steps":
            k = st.number_input(
                "Enter k (positive for delay, negative for advance):", value=0, step=1
            )
            if st.button("Apply"):
                indices_shifted, amplitudes_shifted = delay_advance_signal(
                    indices_folded, amplitudes_folded, k
                )
                draw(indices_shifted, amplitudes_shifted)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile:
                if indices_shifted is None or amplitudes_shifted is None:
                    st.error(
                        "Please perform the required operation first to generate a shifted signal."
                    )
                else:
                    Shift_Fold_Signal(
                        f"signals/task5/Shifting_and_folding/Shifting and Folding/{comparingFile.name}",
                        indices_shifted,
                        amplitudes_shifted,
                    )

        elif operation == "remove DC":
            mean_amplitude = sum(amplitudes) / len(amplitudes)
            dc_removed_amplitudes = [amp - mean_amplitude for amp in amplitudes]
            draw(indices, dc_removed_amplitudes)
            st.write(dc_removed_amplitudes)
            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and dc_removed_amplitudes and indices:
                SignalSamplesAreEqual(comparingFile, indices, dc_removed_amplitudes)

elif menu == "Sharpening":
    DerivativeSignal()

elif menu == "smoothing":
    st.header("Smoothing")

    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
    if uploaded_file is not None:
        indices, amplitudes = readSignal(0, uploaded_file)
        window_size = st.number_input(
            "Enter the window size for moving average:", min_value=1, value=3
        )
        smoothed_amplitudes = compute_moving_average(amplitudes, window_size)

        if st.button("Apply Smoothing"):
            draw(indices, smoothed_amplitudes)
            st.write("Smoothed Signal (Moving Average):", smoothed_amplitudes)
        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile and smoothed_amplitudes and indices:
            SignalSamplesAreEqual(comparingFile, indices, smoothed_amplitudes)

elif menu == "Convolution":
    st.header("Signal Convolution")
    uploaded_file1 = st.file_uploader(
        "Upload the first signal txt file", type="txt", key="file1"
    )
    uploaded_file2 = st.file_uploader(
        "Upload the second signal txt file", type="txt", key="file2"
    )

    if uploaded_file1 and uploaded_file2:
        indices1, amplitudes1 = readSignal(0, uploaded_file1)
        indices2, amplitudes2 = readSignal(0, uploaded_file2)

        if st.button("Perform Convolution"):
            convolved_amplitudes, convolved_indices = convolve_signals(
                indices1, amplitudes1, indices2, amplitudes2
            )
            draw(convolved_indices, convolved_amplitudes)
            st.write(convolved_indices, convolved_amplitudes)
            ConvTest(convolved_indices, convolved_amplitudes)

elif menu == "Correlation":
    st.header("Normalized Cross-Correlation")
    uploaded_file1 = st.file_uploader(
        "Upload the first signal txt file", type="txt", key="file1"
    )
    uploaded_file2 = st.file_uploader(
        "Upload the second signal txt file", type="txt", key="file2"
    )

    if uploaded_file1 and uploaded_file2:
        indices1, amplitudes1 = readSignal(0, uploaded_file1)
        indices2, amplitudes2 = readSignal(0, uploaded_file2)
        result = norm_cross_correlation(amplitudes1, amplitudes2)
        st.write(result)

        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        SignalSamplesAreEqual(comparingFile, indices1, result)

elif menu == "FIR Filters":
    st.header("Filtering")
    menu2 = st.selectbox(
        "Select a Filter ",
        [
            "Low_Pass",
            "High_Pass",
            "Band_Pass",
            "Band_Stop",
        ],
        index=None,
    )
    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
    if uploaded_file is not None:
        indices, amplitudes = readSignal(0, uploaded_file)
    h = None
    fs = st.number_input("Sampling Freq (Hz)", value=1)
    stopA = st.number_input("Stopband attenuation (dB)", value=1)
    transitionBand = st.number_input("Transition Band (Hz)", value=1)

    if menu2 in ["Low_Pass", "High_Pass"]:
        fc = st.number_input("Cutoff Freq (Hz)", value=1)
        h = FIR_Filter(menu2, fs, stopA, fc, transitionBand)

    elif menu2 in ["Band_Pass", "Band_Stop"]:
        f1 = st.number_input("F1 (Hz)", value=1)
        f2 = st.number_input("F2 (Hz)", value=1)
        h = FIR_Filter(menu2, fs, stopA, (f1, f2), transitionBand)

    if h:
        st.write(h)
        amp, ind = convolve_signals([0], h, indices, amplitudes)

    comparingFile = st.file_uploader("Upload the signal compare txt file", type="txt")

    SignalSamplesAreEqual(comparingFile, ind, amp)


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
            indices, samples = addSignals(uploaded_file1, uploaded_file2)

        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile and samples and indices:
            SignalSamplesAreEqual(comparingFile, indices, samples)

    # Display for Sub button
    elif st.session_state["button_pressed"] == "button_2":
        uploaded_file1 = st.file_uploader(
            "Upload the first signal txt file", type="txt"
        )

        uploaded_file2 = st.file_uploader(
            "Upload the second signal txt file", type="txt"
        )
        if uploaded_file1 and uploaded_file2:
            indices, samples = subtractSignals(uploaded_file1, uploaded_file2)

        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile and samples and indices:
            SignalSamplesAreEqual(comparingFile, indices, samples)

    # Display for Multiply button
    elif st.session_state["button_pressed"] == "button_3":
        uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
        constant = st.number_input("Enter a constant", min_value=-1.0, value=1.0)
        if uploaded_file:
            indices, samples = multiplySignals(uploaded_file, constant)
            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indices:
                SignalSamplesAreEqual(comparingFile, indices, samples)

    # Display for Square button
    elif st.session_state["button_pressed"] == "button_4":
        uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
        if uploaded_file:
            indices, samples = squareSignals(uploaded_file)
        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile and samples and indices:
            SignalSamplesAreEqual(comparingFile, indices, samples)

    # Display for Normalize button
    elif st.session_state["button_pressed"] == "button_6":
        uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
        choice = st.radio("Normalize to:", ("[-1,1]", "[0,1]"), index=0)
        if uploaded_file:
            if choice == "[-1,1]":
                indices, samples = normalizeSignal1(uploaded_file)
            else:
                indices, samples = normalizeSignal0(uploaded_file)

            comparingFile = st.file_uploader(
                "Upload the signal compare txt file", type="txt"
            )
            if comparingFile and samples and indices:
                SignalSamplesAreEqual(comparingFile, indices, samples)

    # Display for accumulate button
    elif st.session_state["button_pressed"] == "button_7":
        uploaded_file = st.file_uploader("Upload the signal txt file", type="txt")
        if uploaded_file:
            indices, samples = accumulate(uploaded_file)

        comparingFile = st.file_uploader(
            "Upload the signal compare txt file", type="txt"
        )
        if comparingFile and samples and indices:
            SignalSamplesAreEqual(comparingFile, indices, samples)

else:
    st.markdown("*Choose an option from the side menu!*")
