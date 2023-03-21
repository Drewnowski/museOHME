from math import nan
from pylsl import StreamInlet, resolve_byprop
from threading import Thread
import numpy as np
from scipy.signal import butter,filtfilt,lfilter,detrend,iirnotch,welch,find_peaks

import numpy as np

def findStreams():

    print("Looking for an LSL stream...")
    # discover the available LSL streams
    streams = resolve_byprop('type', 'EEG', timeout=5)
    ppg_streams = resolve_byprop("type", "PPG", timeout=5)

    if len(streams) == 0:
        raise(RuntimeError("Can't find LSL stream."))

    print("Start acquiring data.")

    # Init stream if BlueMuse is streaming, we take the first streming device
    stream = StreamCleanDataOSC(streams[0],ppg_streams[0])
    # stream.update_OSCstream()
    stream.start()
    

class StreamCleanDataOSC():
    def __init__(self, stream,ppg_stream):
        """Init"""
        self.stream = stream
        self.fs_speed = 32
        # connect to the streams
        self.inlet = StreamInlet(stream, max_chunklen=256)
        self.ppg_inlet = StreamInlet(ppg_stream)
        info = self.inlet.info()
        # self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.preprocess_every = int(256/self.fs_speed)
        self.data = np.zeros((256, self.n_chan))
        self.times = np.zeros(self.fs_speed)
        self.times256 = []

        self.connection_quality_channels = [3,3,3,3]
        # Initialize the heart rate buffer and timestamps
        self.hr_buffer = []
        self.hr_timestamps = []


    def updateMainLoop(self):
        try:
            nb_pull = 0
            ppg_buffer = []
            # Main program loop
            while True:
                # Receive PPG data and calculate heart rate every 5 seconds
                ppg_chunk, ppg_timestamp = self.ppg_inlet.pull_chunk(timeout=1.0,max_samples=65)
                if ppg_timestamp: 
                    ppg_buffer.extend(ppg_chunk)
                    nb_pull += 1
                    if nb_pull >= 5:
                        nb_pull = 0
                        ppg_signal = np.array(ppg_buffer).T
                        filtred_ppg_signal = ppg_data_filter(ppg_signal)

                        heart_rate = HR_generator(filtred_ppg_signal)
                        SpO2 = calc_SpO2(filtred_ppg_signal)
                        print('Heart rate:', heart_rate)
                        print('SpO2:', SpO2)
                        ppg_buffer = []
                            

        except RuntimeError as e:
            raise

    def start(self):
        self.started = True
        self.thread = Thread(target=self.updateMainLoop)
        # self.thread.daemon = True
        self.thread.start()
    
def ppg_data_filter(ppg_signal,lowcut=0.5, highcut=4.0, fs=65):
    # Define the filter parameters
    nyquist_freq =  fs / 2.0
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Apply the bandpass filter
    b, a = butter(3, [low, high], btype='bandpass')
    filtered_signal = filtfilt(b, a, ppg_signal)

    return filtered_signal

def HR_generator(data):
    # Convert the PPG signal to a 1D array by taking the mean across all channels
    ppg_signal = np.mean(data, axis=0)

    
    # Find the peaks in the heart rate signal using a peak detection algorithm
    peaks, _ = find_peaks(ppg_signal, height=0)
    return len(peaks)

def calc_SpO2(filtered_signal):
    filtered_signal = np.mean(filtered_signal, axis=0)

    # Define the absorption coefficients for oxygenated and deoxygenated hemoglobin
    alpha_oxy = 0.15
    alpha_deoxy = 0.17

    # Calculate the ratio of the AC component to the DC component of the PPG signal
    ratio = calculate_ratio(filtered_signal)

    # Use the Beer-Lambert law to estimate the SpO2 value
    spo2 = (alpha_oxy * ratio) / (alpha_oxy * ratio + alpha_deoxy)
    return spo2

def calculate_ratio(signal):
    """
    Calculate the ratio of the AC component to the DC component of the PPG signal.

    :param signal: The PPG signal with the AC component extracted.
    :return: The ratio of the AC component to the DC component of the signal.
    """
    ac_component = signal - np.mean(signal)
    dc_component = np.mean(signal)
    ratio = np.max(ac_component) / dc_component

    return ratio

findStreams()