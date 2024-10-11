import matplotlib.pyplot as plt
import numpy as np


# Second task function
def generate_signal(sineFlag, A, f, theta, fs):
    t = np.linspace(0, 1)

    if sineFlag:  # if its equal to 1 , then its a sine wave
        signal = A * np.sin(2 * np.pi * f * t + theta)
    else:
        signal = A * np.cos(2 * np.pi * f * t + theta)

    return t, signal


# First Task
# TODO: replace fileName with the file you receive from the user
with open(fileName, "r") as f:
    timeFlag = f.readline()
    periodicFlag = f.readline()

    nOfSamples = int(f.readline())
    indices = []
    amplitudes = []
    for _ in range(nOfSamples):
        line = f.readline().strip().split(" ")
        indices.append(int(line[0]))
        amplitudes.append(float(line[1]))

figCont, axCont = plt.subplots(figsize=(10, 5))
axCont.plot(indices, amplitudes)
# Plotting the signal

st.pyplot(figCont)  # continuous

figDis, axDis = plt.subplots(figsize=(10, 5))
axDis.stem(indices, amplitudes)

st.pyplot(figDis)  # discrete


# Second Task
# TODO fill the variables from the user
sinFlag =   # expects a boolean
amp =   # expects a float or integer
freq =   # expects a float or integer
theta =   # expects a float
samplingFreq =   # expects a float or integer
if samplingFreq < 2 * freq:
    # TODO show an error to the user because the sampling frequency must be
    # at least twice the analog frequency
    
t, wave = generate_signal(sinFlag, amp, freq, theta, samplingFreq)

# Plotting

# Continuous Signal
fig_cont, ax_cont = plt.subplots(figsize=(10, 5))
ax_cont.plot(t, sine_wave)
ax_cont.set_title(f"Continuous Signal")
ax_cont.set_xlabel("Time (s)")
ax_cont.set_ylabel("Amplitude")

st.pyplot(fig_cont)

# Discrete Signal
fig_disc, ax_disc = plt.subplots(figsize=(10, 5))
ax_disc.scatter(t, sine_wave)
ax_disc.set_title(f"Discrete Signal")
ax_disc.set_xlabel("Time (s)")
ax_disc.set_ylabel("Amplitude")

st.pyplot(fig_disc)
