import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# Second task function
def generate_signal(sineFlag, A, f, theta, fs):
    t = np.linspace(0, 1, int(fs))  # Corrected to generate based on sampling frequency
    if sineFlag:  # if sineFlag is True, then it's a sine wave
        signal = A * np.sin(2 * 180 * f * t/fs + theta)
    else:
        signal = A * np.cos(2 * 180 * f * t/fs + theta)
    return t, signal


st.title("Signal Processing")

st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select an option", ["Signal Reading", "Signal Generation"], index=None)

if menu == "Signal Reading":
    st.header("Read Signal Samples")

    uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
    read_button = st.button("Read Signal")

    if read_button:
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8").splitlines()

            timeFlag = file_content[0]  # First line
            periodicFlag = file_content[1]  # Second line
            nOfSamples = int(file_content[2])  # Third line

            indices = []
            amplitudes = []
            for line in file_content[3:3 + nOfSamples]:  # Process the next nOfSamples lines
                values = line.strip().split(" ")
                indices.append(int(values[0]))
                amplitudes.append(float(values[1]))

            # Plotting continuous signal
            figCont, axCont = plt.subplots(figsize=(50, 5))
            axCont.plot(indices, amplitudes)
            st.pyplot(figCont)  # continuous

            # Plotting discrete signal
            figDis, axDis = plt.subplots(figsize=(10, 5))
            axDis.stem(indices, amplitudes)
            st.pyplot(figDis)  # discrete


elif menu == "Signal Generation":
    st.header("Signal Generation")

    signal_type = st.selectbox("Choose Signal Type", ["Sine Wave", "Cosine Wave"])
    amplitude = st.number_input("Amplitude (A)", min_value=0.0, value=1.0)
    phaseShift = st.number_input("Phase Shift ", min_value=0.0, value=0.0)
    analogFreq = st.number_input("Analog Frequency (Hz)", min_value=0.0, value=1.0)
    samplingFreq = st.number_input("Sampling Frequency (Hz)", min_value=0.0, value=10.0)

    generate_button = st.button("Generate Signal")
    if generate_button:
        sinFlag = signal_type == "Sine Wave"  # True for sine, False for cosine
        if samplingFreq < 2 * analogFreq:
            st.error("Sampling frequency must be at least twice the analog frequency .")
        else:
            t, wave = generate_signal(sinFlag, amplitude, analogFreq, phaseShift, samplingFreq)

            # Plotting Continuous Signal
            fig_cont, ax_cont = plt.subplots(figsize=(10, 5))
            ax_cont.plot(t, wave)
            ax_cont.set_title("Continuous Signal")
            ax_cont.set_xlabel("Time (s)")
            ax_cont.set_ylabel("Amplitude")
            st.pyplot(fig_cont)

            # Plotting Discrete Signal
            fig_disc, ax_disc = plt.subplots(figsize=(10, 5))
            ax_disc.scatter(t, wave)
            ax_disc.set_title("Discrete Signal")
            ax_disc.set_xlabel("Time (s)")
            ax_disc.set_ylabel("Amplitude")
            st.pyplot(fig_disc)

else:
    st.markdown("*Choose an option from the side menu!*")