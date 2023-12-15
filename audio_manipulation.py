import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import butter, lfilter
from io import BytesIO

st.set_option('deprecation.showPyplotGlobalUse', False)

def perform_audio_manipulation():
    # Initialize variables
    fs = 44100  # Default sample rate
    audio_data = None
    filter_type = "None"
    lower_cutoff = 0
    upper_cutoff = 0
    cutoff_frequency = 0
    
    st.title("Audio Manipulation")
    
    # Sidebar navigation
    display_waveform = st.sidebar.checkbox("Display Audio Waveform")
    display_filtered_waveform = st.sidebar.checkbox("Display Filtered Audio Waveform")
    display_spectrogram = st.sidebar.checkbox("Display Spectrogram")
    display_audio_info = st.sidebar.checkbox("Display Audio Info")
    
    # Filtering options
    filter_options = st.sidebar.checkbox("Filtering Options")
    if filter_options:
        filter_type = st.sidebar.radio("Select Filter Type", ["None", "Low-pass", "High-pass", "Band-pass"])
        if filter_type == "Band-pass":
            lower_cutoff = st.sidebar.slider("Lower Cutoff Frequency (Hz)", min_value=0, max_value=int(fs / 2), value=20, step=10)
            upper_cutoff = st.sidebar.slider("Upper Cutoff Frequency (Hz)", min_value=lower_cutoff, max_value=int(fs / 2), value=200, step=10)
        else:
            cutoff_frequency = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=0, max_value=int(fs / 2), value=200, step=10)
    
    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if audio_file is not None:
        # Read audio file
        fs, audio_data = read(BytesIO(audio_file.read()))

        # Apply filtering if selected
        if filter_type != "None":
            filtered_audio = apply_filter(audio_data, fs, filter_type, cutoff_frequency, lower_cutoff, upper_cutoff)
        else:
            filtered_audio = audio_data

        # Display audio waveform if checkbox is selected
        if display_waveform:
            plot_audio_waveform(audio_data)

        # Display filtered audio waveform if checkbox is selected
        if display_filtered_waveform:
            plot_audio_waveform(filtered_audio, title="Filtered Audio Waveform")

        # Display audio information if checkbox is selected
        if display_audio_info:
            display_audio_information(fs, filtered_audio)

        # Display spectrogram if checkbox is selected
        if display_spectrogram:
            plot_spectrogram(filtered_audio, fs)

def apply_filter(audio_data, fs, filter_type, cutoff_frequency, lower_cutoff, upper_cutoff):
    # Design the filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist

    if filter_type == "Low-pass":
        b, a = butter(6, normal_cutoff, btype='low', analog=False)
    elif filter_type == "High-pass":
        b, a = butter(6, normal_cutoff, btype='high', analog=False)
    elif filter_type == "Band-pass":
        low = lower_cutoff / nyquist
        high = upper_cutoff / nyquist
        b, a = butter(6, [low, high], btype='band', analog=False)
    else:
        return audio_data  # No filtering

    # Apply the filter
    filtered_audio = lfilter(b, a, audio_data)
    return filtered_audio

def plot_audio_waveform(audio_data, title="Audio Waveform"):
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    st.pyplot()

def plot_spectrogram(audio_data, fs):
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.specgram(audio_data, Fs=fs, cmap='viridis')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    st.pyplot()

def display_audio_information(fs, audio_data):
    # Display audio information
    st.subheader("Audio Information")
    st.text(f"Sampling frequency (fs): {fs}")
    st.text(f"Duration: {len(audio_data) / fs:.2f} seconds")
    st.text(f"Number of channels: {audio_data.shape[1] if len(audio_data.shape) == 2 else 1}")
    st.text(f"Bit depth: {audio_data.dtype.itemsize * 8} bits")
    st.text(f"Shape of the audio data: {audio_data.shape}")

    # If stereo, extract mono channel
    if len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
        data_mono = np.array([audio_data[i][0] for i in range(len(audio_data))])
        st.text(f"Shape of the mono channel: {data_mono.shape}")

# Run the Streamlit app
perform_audio_manipulation()
