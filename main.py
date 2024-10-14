import plotly.graph_objects as go
import numpy as np
import streamlit as st


def main():
    st.title("Signal Processing")

    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox(
        "Select an option", ["Signal Reading", "Signal Generation"], index=None
    )

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
                for line in file_content[3 : 3 + nOfSamples]:
                    values = line.strip().split(" ")
                    indices.append(int(values[0]))
                    amplitudes.append(float(values[1]))

                draw(indices, amplitudes)

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

    else:
        st.markdown("*Choose an option from the side menu!*")


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


main()
