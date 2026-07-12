# Digital Signal Processing

An interactive Streamlit app that implements classic digital signal processing operations from scratch in Python.

## Overview

Every DSP operation here — DFT, DCT, convolution, correlation, quantization, FIR filter design — is written directly from its mathematical definition using plain Python and NumPy arrays. No `scipy.signal`, no `numpy.fft`. The point is to show the math rather than call a library that hides it.

The app wraps those implementations in a Streamlit GUI: pick an operation from the sidebar, upload a signal file, and see the result plotted (continuous and discrete) with Plotly. Most operations also accept an expected-output file and report whether the result matches it, so the test cases bundled under `signals/` can be verified from the UI.

## Features

- **Signal reading & plotting** — upload a signal file, view it as continuous and discrete plots.
- **Signal generation** — sine or cosine waves from amplitude, analog frequency, sampling frequency, and phase shift, with a Nyquist check (`fs >= 2f`).
- **Arithmetic operations** — add, subtract, multiply by a constant, square, normalize (to `[-1,1]` or `[0,1]`), and accumulate.
- **Quantization** — by number of levels or number of bits; reports interval index, encoded level, quantized value, and quantization error.
- **Frequency domain** — DFT (amplitude and phase over a real frequency axis), IDFT, DCT, and DC-component removal.
- **Time domain** — delay/advance by *k* steps, folding, delay/advance of a folded signal, and DC removal.
- **Sharpening** — first and second derivative.
- **Smoothing** — moving average with a configurable window size.
- **Convolution** — direct convolution of two signals.
- **Correlation** — normalized cross-correlation.
- **FIR filters** — low-pass, high-pass, band-pass, and band-stop. The window (Rectangular, Hanning, Hamming, or Blackman) and the filter order are derived automatically from the requested stopband attenuation and transition width.
- **Built-in test comparison** — compare a result against an expected-output file and get a pass/fail message.

## Tech stack

- Python
- [Streamlit](https://streamlit.io/) — the GUI
- [NumPy](https://numpy.org/) — numerics
- [Plotly](https://plotly.com/python/) — plotting

## Getting started

Requires Python 3.

```bash
git clone https://github.com/Omar-Elhakim/Digital-Signal-Processing.git
cd Digital-Signal-Processing
pip install -r requirements.txt
```

## Usage

Run the app:

```bash
streamlit run main.py
```

Streamlit opens the app in your browser. Choose an operation from the sidebar menu, then upload a signal file from `signals/`.

For example, to try an FIR low-pass filter: select **FIR Filters → Low_Pass**, upload `signals/task7/FIR/Testcase 2/ecg400.txt`, and enter the values from that test case's `Filter Specifications.txt` (sampling frequency, stopband attenuation, cutoff frequency, transition band).

### Signal file format

Signal files are plain text:

```
0        <- time flag
0        <- periodic flag
1001     <- number of samples
0 0      <- index amplitude
1 1
2 2
...
```

## Project structure

```
main.py           Streamlit app — the UI and the menu for each operation
functions.py      All DSP implementations (transforms, filters, comparison tests)
main.ipynb        Development notebook the implementations were prototyped in
signals/          Input signals and expected outputs, grouped by task
requirements.txt
```
