import plotly.graph_objects as go
import numpy as np
import streamlit as st


def main():

# Main page
    st.title("Digital Signal Processing")

    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox(
        "Select an option", ["Arithmetic Operations", "Signal Reading", "Signal Generation"], index=None
    )

# Read and draw a Signal from a file
    if menu == "Signal Reading":
        st.header("Read Signal Samples")

        uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
        read_button = st.button("Read Signal")

        if read_button:
            if uploaded_file is not None:
                indices ,amplitudes= readSignal(uploaded_file)
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

    elif menu == "Signal Generation":
            pass

    else:
        st.markdown("*Choose an option from the side menu!*")

def readSignal(uploaded_file):
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    # timeFlag = file_content[0]  # First line
    # periodicFlag = file_content[1]  # Second line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split(" ")
        indices.append(int(values[0]))
        amplitudes.append(float(values[1]))
        
    return indices,amplitudes

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


# Second task function
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

def addSignals(firstSignalFile,secondSignalFile):
    # Read the files
    indices1,amplitudes1=readSignal(firstSignalFile)
    indices2,amplitudes2=readSignal(secondSignalFile)
    if len(indices1)!=len(indices2):
        print('The two files must be the same size')
        return
    addedAmplitudes=list(x+y for x,y in zip(amplitudes1,amplitudes2))
    draw(indices1,addedAmplitudes)
    return indices1,addedAmplitudes

def subtractSignals(firstSignalFile,secondSignalFile):
    # Read the files
    indices1,amplitudes1=readSignal(firstSignalFile)
    indices2,amplitudes2=readSignal(secondSignalFile)
    if len(indices1)!=len(indices2):
        print('The two files must be the same size')
        return
    subtractedAmplitudes=list(x-y for x,y in zip(amplitudes1,amplitudes2))
    draw(indices1,subtractedAmplitudes)
    return indices1,subtractedAmplitudes

def normalizeSignal0(signal):
    max_value = max(np.abs(signal))
    normalized_signal = list(x / max_value for x in signal)
    return normalized_signal

def normalize_signal1(signal):
    min_val = min(signal)
    max_val = max(signal)
    signal = list((i - min_val) / (max_val - min_val) for i in signal)
    return signal
main()
