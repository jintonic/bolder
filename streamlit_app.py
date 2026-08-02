import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import struct
import io

st.title("BOLDER")
st.subheader("[BEARD](https://physino.xyz/beard) Output Loader & Data ExploreR")

uploaded_file = st.file_uploader("Upload your binary file")

if uploaded_file is None:
    st.info("Please upload a binary file to begin.")
    st.stop()

binary_bytes = uploaded_file.read()

if not binary_bytes:
    st.error("The uploaded file is empty.")
    st.stop()

f = io.BytesIO(binary_bytes)

header_n = f.read(2)
if len(header_n) < 2:
    st.error("File is too short to read the header.")
    st.stop()

n = struct.unpack('<H', header_n)[0]
st.success(f"File loaded successfully! (n={n} samples per waveform)")

SAMPLE_INTERVAL_US = 2
timestamps = []
waveforms = []
max_heights = []

while True:
    ts_data = f.read(8)
    if not ts_data:
        break
    
    wf_data = f.read(n)
    if len(wf_data) < n:
        st.warning("Warning: Partial waveform found at end of file. Stopping.")
        break
    
    ms = struct.unpack('<Q', ts_data)[0]
    wf = np.frombuffer(wf_data, dtype=np.uint8)
    
    timestamps.append(ms)
    waveforms.append(wf)
    max_heights.append(np.max(wf))

if not timestamps:
    st.error("Error: No valid events found in the file.")
    st.stop()

ts = np.array(timestamps, dtype=np.uint64)
samples = np.array(waveforms, dtype=np.uint8)
max_heights = np.array(max_heights, dtype=np.uint8)
sample_times = np.arange(n, dtype=np.float32) * SAMPLE_INTERVAL_US

st.header("Energy Spectrum (Pulse Heights)")

fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
ax_hist.hist(max_heights, bins=50, color='tab:blue', edgecolor='black')
ax_hist.set_xlabel("Height (h)")
ax_hist.set_ylabel("Entries")
st.pyplot(fig_hist)

st.header("Waveform Inspector")

total_events = len(samples)
default_window = min(10, total_events)

# Range slider lets the user select a slice of events [start, end]
selected_range = st.slider(
    "Select Event Range",
    min_value=0,
    max_value=total_events - 1,
    value=(0, default_window - 1),
    step=1
)

start_event, end_event = selected_range
# Inclusive range for plotting
end_event_inclusive = min(end_event, total_events - 1)

fig_wave, ax_wave = plt.subplots(figsize=(8, 4))

for i in range(start_event, end_event_inclusive + 1):
    ax_wave.plot(sample_times, samples[i], label=f"Event {i} (ms={ts[i]})")

ax_wave.set_xlabel("Time ($\mu$s)")
ax_wave.set_ylabel("ADC Counts (s)")
ax_wave.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_wave)