import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import struct
import io

st.title("BOLDER")
st.subheader("[BEARD](https://physino.xyz/beard) Output Loader & Data ExploreR")

num_files = st.radio("Number of files to compare", options=[1, 2], horizontal=True)

datasets = []

for i in range(num_files):
    st.markdown(f"### Dataset {i+1}")
    default_name = "Dataset 1" if i == 0 else "Dataset 2"
    ds_name = st.text_input(f"Name for Dataset {i+1}", value=default_name, key=f"name_{i}")
    
    uploaded_file = st.file_uploader(f"Upload binary file for {ds_name}", max_upload_size=1, key=f"file_{i}")
    
    if uploaded_file is not None:
        binary_bytes = uploaded_file.read()
        if not binary_bytes:
            st.error(f"The uploaded file for {ds_name} is empty.")
            continue
        
        f = io.BytesIO(binary_bytes)
        header_n = f.read(2)
        if len(header_n) < 2:
            st.error(f"File for {ds_name} is too short to read the header.")
            continue
        
        n = struct.unpack('<H', header_n)[0]
        
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
                break
            ms = struct.unpack('<Q', ts_data)[0]
            wf = np.frombuffer(wf_data, dtype=np.uint8)
            timestamps.append(ms)
            waveforms.append(wf)
            max_heights.append(np.max(wf))
            
        if not timestamps:
            st.error(f"Error: No valid events found in {ds_name}.")
            continue
            
        ts = np.array(timestamps, dtype=np.uint64)
        samples = np.array(waveforms, dtype=np.uint8)
        max_heights = np.array(max_heights, dtype=np.uint8)
        sample_times = np.arange(n, dtype=np.float32) * SAMPLE_INTERVAL_US
        
        total_time_sec = (ts[-1] - ts[0]) / 1000.0
        
        datasets.append({
            "name": ds_name,
            "n": n,
            "ts": ts,
            "samples": samples,
            "max_heights": max_heights,
            "sample_times": sample_times,
            "total_time_sec": total_time_sec
        })

if not datasets:
    st.info("Please upload at least one valid binary file to begin.")
    st.stop()

st.header("Energy Spectrum (Pulse Heights)")

fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
colors = ['tab:blue', 'tab:green']
markers = ['o', '^']

for idx, ds in enumerate(datasets):
    if ds["total_time_sec"] <= 0:
        st.error(f"Error: Total data-taking time for {ds['name']} must be greater than zero to calculate rate.")
        st.stop()
        
    counts, bin_edges = np.histogram(ds["max_heights"], bins=64, range=(0, 256))
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2.0
    
    rate_heights = counts / (ds["total_time_sec"] * bin_widths)
    rate_errors = np.sqrt(counts) / (ds["total_time_sec"] * bin_widths)
    
    # Unfilled histogram with thinner solid line (no label to exclude from legend)
    ax_hist.hist(
        bin_edges[:-1], 
        bins=bin_edges, 
        weights=rate_heights, 
        histtype='step', 
        color=colors[idx % len(colors)], 
        linewidth=0.8,
        linestyle='-'
    )
    
    # Data points with error bars and legend label
    ax_hist.errorbar(
        bin_centers, 
        rate_heights, 
        yerr=rate_errors, 
        fmt=markers[idx % len(markers)], 
        color=colors[idx % len(colors)], 
        ecolor=colors[idx % len(colors)], 
        elinewidth=1, 
        capsize=2,
        label=ds['name']
    )

ax_hist.set_xlabel("Height (ADC counts)")
ax_hist.set_ylabel("Rate (Hz/ADC count)")
ax_hist.legend()
st.pyplot(fig_hist)

for idx, ds in enumerate(datasets):
    st.header(f"Trigger Rate — {ds['name']}")
    
    ts_seconds = ds["ts"] / 1000.0
    ts_minutes = ts_seconds / 60.0
    
    fig_rate, ax_rate = plt.subplots(figsize=(8, 4))
    
    counts, bin_edges_min, patches = ax_rate.hist(
        ts_minutes, 
        bins=50, 
        color=colors[idx % len(colors)], 
        edgecolor='black'
    )
    
    bin_widths_sec = np.diff(bin_edges_min) * 60.0
    
    ax_rate.clear()
    ax_rate.bar(
        bin_edges_min[:-1], 
        counts / bin_widths_sec, 
        width=np.diff(bin_edges_min), 
        align='edge', 
        color=colors[idx % len(colors)], 
        edgecolor='black'
    )
    
    ax_rate.set_xlabel("Time (minutes)")
    ax_rate.set_ylabel("Trigger Rate (Hz)")
    st.pyplot(fig_rate)

for idx, ds in enumerate(datasets):
    st.header(f"Waveform Inspector — {ds['name']}")
    
    total_events = len(ds["samples"])
    default_window = min(10, total_events)
    
    selected_range = st.slider(
        f"Select Event Range ({ds['name']})",
        min_value=0,
        max_value=total_events - 1,
        value=(0, default_window - 1),
        step=1,
        key=f"slider_{idx}"
    )
    
    start_event, end_event = selected_range
    end_event_inclusive = min(end_event, total_events - 1)
    
    fig_wave, ax_wave = plt.subplots(figsize=(8, 4))
    
    for i in range(start_event, end_event_inclusive + 1):
        ax_wave.plot(ds["sample_times"], ds["samples"][i], label=f"Event {i} (ms={ds['ts'][i]})")
        
    ax_wave.set_xlabel(r"Time ($\mu$s)")
    ax_wave.set_ylabel("ADC Counts")
    ax_wave.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_wave)