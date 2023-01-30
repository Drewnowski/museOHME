from math import nan
from pylsl import StreamInlet, resolve_byprop
from threading import Thread
import numpy as np
from scipy.signal import butter,filtfilt,detrend,iirnotch,welch
from time import sleep,time


#osc import
from pythonosc import udp_client
from pythonosc import osc_message_builder

from utils import *

# Create UDP client
client = udp_client.UDPClient('192.168.1.255',9998,  True)

current_delta_max = 0.0
current_theta_max = 0.0
current_alpha_max = 0.0
current_beta_max = 0.0
current_gamma_max = 0.0

def streamCleanData():

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))

    print("Start acquiring data.")

    # Init stream if BlueMuse is streaming, we take the first streming device
    stream = StreamCleanDataOSC(streams[0])
    # stream.update_OSCstream()
    stream.start()
    

class StreamCleanDataOSC():
    def __init__(self, stream, dejitter=True):
        """Init"""
        self.stream = stream
        self.dejitter = dejitter
        self.fs_speed = 32
        self.inlet = StreamInlet(stream, max_chunklen=256)
        info = self.inlet.info()
        # self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        self.preprocess_every = int(256/self.fs_speed)
        self.data = np.zeros((256, self.n_chan))

        self.connection_quality_channels = [3,3,3,3]


    def update_OSCstream(self):
        nb_pull = 0
        signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10 = [[],[],[],[]]
        try:
            while self.started:

                samples, timestamps = self.inlet.pull_chunk(timeout=1.0, max_samples=self.fs_speed)#256
                
                if timestamps:
                    self.data = np.vstack([self.data, samples])
                    self.data = self.data[-256:]
                    [signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10] = self.preprocess(self.data)
                    self.process(signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10)

                    #MODIF
                    record(np.column_stack([signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10]),timestamps)
                    # ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 256 and the array at index 3 has size 0
                    nb_pull +=1
                    if nb_pull == self.preprocess_every:
                        print('Full message')
                        # [signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10] = self.preprocess(self.data)
                        nb_pull = 0
                    else:
                        sleep(0.08)
                

        except RuntimeError as e:
            raise

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update_OSCstream)
        # self.thread.daemon = True
        self.thread.start()
    
    def connectionQuality(self, signal):
        # The standard deviation is the square root of the average of the squared deviations from the mean
        val = np.std(signal)
        if val < 20:
            return 1
        elif val < 30:
            return 2
        else:
            return 3
    def preprocessing(self,signal,artifact_detection):
        # Cancellation of the DC component
        signal = signal - np.mean(signal)

        # filtering signal with Butterworth passband filter 0.1 - 50Hz
        fs = 256 # Sample frequency (Hz)
        nyq = fs/2
        order = 4
        b,a = butter(order,[0.1/nyq,50.0/nyq],btype='band')
        signal = filtfilt(b,a,signal)
        

        # Detrending
        # The function detrend subtracts the mean or a best-fit line (in the least-squares sense)
        # from your data. If your data contains several data columns, detrend
        # treats each data column separately.
        # https://nl.mathworks.com/help/matlab/data_analysis/detrending-data.html
        signal = detrend(signal) # detrend removes the linear trend
        

        # Notch filter (just in case)
        # f0 = 50
        # q = 35 # arbitrary
        # wo = f0/nyq
        # bw = wo/q
        notch_freq = 50.0  # Frequency to be removed from signal (Hz)
        quality_factor = 35.0  # Quality factor (arbitrary)
        b_notch, a_notch = iirnotch(notch_freq, quality_factor,fs)
        signal = filtfilt(b_notch,a_notch,signal)

        # blinks detection
        if artifact_detection:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if np.max(signal) > signal_mean + 3*signal_std or np.min(signal) < signal_mean - 3*signal_std:
                return nan

        return signal

    def frequency_bands_separation(self, signal_clean):
        # Frequency analysis of the signal. Returns the bandpowers of the 5
        # main frequency bands (delta, theta, alpha, beta, gamma)
        fs = 256
        # fs = 32
        # calcule de power spectral density by Welch method
        f,pxx = welch(signal_clean,fs)
        # Separation of PSD in bandpower
        delta = np.trapz(pxx[np.argmax(f > 0.2) - 1 : np.argmax(f > 4) - 1]) # 0-4Hz
        theta = np.trapz(pxx[np.argmax(f > 4) - 1 : np.argmax(f > 8) - 1]) # 4-8Hz
        alpha = np.trapz(pxx[np.argmax(f > 8) - 1 : np.argmax(f > 14) - 1]) # 8-14Hz
        beta = np.trapz(pxx[np.argmax(f > 14) - 1 : np.argmax(f > 30) - 1]) # 14-30Hz
        gamma = np.trapz(pxx[np.argmax(f > 30) - 1 : np.argmax(f > 80) - 1]) # 30-80Hz

        return [delta,theta,alpha,beta,gamma]



    def preprocess(self, matrix_data):
        # Data preparation
        data = np.reshape(matrix_data, (256, 4)) # bcz 5 different channel of data (change 4 -> 5 like in OpenVibe) 4 channels for csv files from Louise
        TP9_data = data[:,0]
        AF7_data = data[:,1]
        AF8_data = data[:,2]
        TP10_data = data[:,3]

        # Create a list of connection quality of each channel
        self.connection_quality_channels = [self.connectionQuality(TP9_data),self.connectionQuality(AF7_data),self.connectionQuality(AF8_data),self.connectionQuality(TP10_data)]
        # Check if at least one sensor is good

        if self.connection_quality_channels[0] < 3: # TP9
            signal_clean_TP9 = self.preprocessing(TP9_data,False) # Preprocessing
        else:
            # signal_clean_TP9 = np.zeros(256)
            signal_clean_TP9 = []

        if self.connection_quality_channels[1] < 3: # AF7
            signal_clean_AF7 = self.preprocessing(AF7_data,False) # Preprocessing
        else:
            # signal_clean_AF7 = np.zeros(256)
            signal_clean_AF7 = []

        if self.connection_quality_channels[2] < 3: # AF8
            signal_clean_AF8 = self.preprocessing(AF8_data,False) # Preprocessing
        else:
            # signal_clean_AF8 = np.zeros(256)
            signal_clean_AF8 = []

        if self.connection_quality_channels[3] < 3: # TP10
            signal_clean_TP10 = self.preprocessing(TP10_data,False) # Preprocessing
        else:
            # signal_clean_TP10 = np.zeros(256)
            signal_clean_TP10 = []
        
        return signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10

    def process(self,signal_clean_TP9,signal_clean_AF7,signal_clean_AF8,signal_clean_TP10):

        nb_good_sensors = 0
        if signal_clean_TP9 != []: # TP9
            [deltaTP9, thetaTP9, alphaTP9, betaTP9, gammaTP9] = self.frequency_bands_separation(signal_clean_TP9)# frequency bands separation
            nb_good_sensors += 1
        else:
            [deltaTP9, thetaTP9, alphaTP9, betaTP9, gammaTP9] = [0,0,0,0,0]

        if signal_clean_AF7 != []: # AF7
            [deltaAF7, thetaAF7, alphaAF7, betaAF7, gammaAF7] = self.frequency_bands_separation(signal_clean_AF7)# frequency bands separation
            nb_good_sensors += 1
        else:
            [deltaAF7, thetaAF7, alphaAF7, betaAF7, gammaAF7] = [0,0,0,0,0]

        if signal_clean_AF8 != []: # AF8
            [deltaAF8, thetaAF8, alphaAF8, betaAF8, gammaAF8] = self.frequency_bands_separation(signal_clean_AF8)# frequency bands separation
            nb_good_sensors += 1
        else:
            [deltaAF8, thetaAF8, alphaAF8, betaAF8, gammaAF8] = [0,0,0,0,0]

        if signal_clean_TP10 != []: # TP10
            [deltaTP10, thetaTP10, alphaTP10, betaTP10, gammaTP10] = self.frequency_bands_separation(signal_clean_TP10)# frequency bands separation
            nb_good_sensors += 1
        else:
            [deltaTP10, thetaTP10, alphaTP10, betaTP10, gammaTP10] = [0,0,0,0,0]
        
        if nb_good_sensors != 0:
            delta = (deltaTP9+deltaAF7+deltaAF8+deltaTP10)/nb_good_sensors
            theta = (thetaTP9 +thetaAF7+thetaAF8+thetaTP10)/nb_good_sensors 
            alpha = (alphaTP9 + alphaAF7+alphaAF8+alphaTP10)/nb_good_sensors
            beta = (betaTP9+betaAF7+betaAF8+betaTP10)/nb_good_sensors
            gamma = (gammaTP9+gammaAF7+gammaAF8+gammaTP10)/nb_good_sensors
        else:
            delta, theta, alpha, beta, gamma = 0,0,0,0,0
        
        # find the maximum
        global current_delta_max
        global current_theta_max 
        global current_alpha_max
        global current_beta_max 
        global current_gamma_max
        
        current_delta_max = highest_value(delta, current_delta_max)
        current_theta_max = highest_value(theta, current_theta_max)
        current_alpha_max = highest_value(alpha, current_alpha_max)
        current_beta_max = highest_value(beta, current_beta_max)
        current_gamma_max = highest_value(gamma, current_gamma_max)


        # Send data on UDP port (protocol OSC)
        # print ([delta,theta,alpha,beta,gamma])
        message_array = osc_message_builder.OscMessageBuilder(address = '/mean')
        message_array2 = osc_message_builder.OscMessageBuilder(address = '/allwaves')

        message_array.add_arg(delta)
        message_array.add_arg(theta)
        message_array.add_arg(alpha)
        message_array.add_arg(beta)
        message_array.add_arg(gamma)

        for i in self.connection_quality_channels:
            message_array.add_arg(i)
        for i in [current_delta_max, current_theta_max, current_alpha_max, current_beta_max, current_gamma_max]:
            message_array.add_arg(i)

        for i in [deltaTP9, thetaTP9, alphaTP9, betaTP9, gammaTP9]:
            message_array2.add_arg(i)
        for i in [deltaAF7, thetaAF7, alphaAF7, betaAF7, gammaAF7]:
            message_array2.add_arg(i)
        for i in [deltaAF8, thetaAF8, alphaAF8, betaAF8, gammaAF8]:
            message_array2.add_arg(i)
        for i in [deltaTP10, thetaTP10, alphaTP10, betaTP10, gammaTP10]:
            message_array2.add_arg(i)
       
        message_array.build()
        message_array2.build()
        client.send(message_array.build())
        client.send(message_array2.build())
        # print(time())



# Run teh stream
streamCleanData()