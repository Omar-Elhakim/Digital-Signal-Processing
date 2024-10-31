import streamlit as st

from functions import *

# Main page
st.title("Digital Signal Processing")

st.sidebar.title("Menu")
menu = st.sidebar.selectbox(
    "Select an option",
    ["Arithmetic Operations", "Signal Reading", "Signal Generation", "Quantization", "Frequency Domain"],
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

    check = st.radio("Choose Transform:", (0, 1), format_func=lambda x: "DFT" if x == 0 else "IDFT"
                     )

    if st.button("Perform Transform"):

        if check == 0:
            indices, amplitudes = readSignal(0, uploaded_file)
            amp, angle = FourierTransform(check, amplitudes)
            draw(indices,amp)
            amp
            angle

        # Hakim: Add the second condition code for the IDTF
        # else:

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
