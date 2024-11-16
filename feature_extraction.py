import os
import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import savemat
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from mne.preprocessing import ICA, create_eog_epochs
from mne_features.feature_extraction import FeatureExtractor
from sklearn.preprocessing import StandardScaler
import pywt
from tqdm import tqdm

folder_path = r"E:"
result_dir = r"E:"

edf_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]
edf_files.sort()
print(edf_files)

target_length = 50 * 60 * 128

interpolation_method = 'linear'

Time_Features = []
Frequency_Features = []
Spatial_Features = []

def interpolate_eeg_data(eeg_data, target_length):
    num_channels, original_length = eeg_data.shape
    original_times = np.linspace(0, original_length - 1, original_length)
    new_times = np.linspace(0, original_length - 1, target_length)
    interpolated_data = np.zeros((num_channels, target_length))

    for channel in range(num_channels):
        interp_func = interp1d(original_times, eeg_data[channel, :], kind='linear', fill_value="extrapolate")
        interpolated_data[channel, :] = interp_func(new_times)

    return interpolated_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_data(data, sample_rate=128, segment_lengths=[2,8]*5):
    result = []
    for length in segment_lengths:
        points = length * 60 * sample_rate

        if length ==2:
            data = data[:, points:]
        if length ==8:
            result.append(data[:, :points])
            data = data[:, points:]


    return np.concatenate(result, axis=1)

def log_specgram(sample, sample_rate, window_size=20, step_size=10, eps=1e-10):
    freqs, times, spec = signal.spectrogram(sample,
                                        fs=sample_rate,
                                        nperseg=window_size,
                                        noverlap=step_size)
    return freqs, times, 10 * np.log(spec.T.astype(np.float32) + eps)


def discrete_wavelet_transform(data):
    wavelet = 'db4'
    level = 3

    transformed_data = []
    for trial in tqdm(range(data.shape[0]), desc='Transforming data'):
        transformed_trial = []
        for channel in range(data.shape[1]):
            coeffs = pywt.wavedec(data[trial, channel, :], wavelet, level=level)
            max_len = max(map(len, coeffs))
            coeffs_padded = [np.pad(c, (0, max_len - len(c)), mode='constant') for c in coeffs]
            transformed_trial.append(coeffs_padded)
        transformed_data.append(transformed_trial)

    return np.array(transformed_data)


# 数据处理流程
for edf_file in edf_files:
    edf_file_path = os.path.join(folder_path, edf_file)
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    raw.set_eeg_reference('average')
    af3_data = raw.copy().pick_channels(['AF3'])
    af3_data.filter(0.1, 4.0)

    exclude_channels = ['']
    data_channels = [ch for ch in raw.ch_names if ch not in exclude_channels]
    other_data = raw.copy().pick_channels(data_channels)

    combined_data = np.vstack([af3_data.get_data(), other_data.get_data()])

    combined_ch_names = other_data.ch_names + ['EOG']
    combined_ch_types = other_data.get_channel_types() + ['eog']

    combined_info = mne.create_info(ch_names=combined_ch_names, sfreq=raw.info['sfreq'], ch_types=combined_ch_types)
    combined_raw = mne.io.RawArray(combined_data, combined_info)

    montage = mne.channels.make_standard_montage('standard_1020')
    combined_raw.set_montage(montage, match_case=False, on_missing='ignore')

    combined_raw.filter(4.0, 45.0)

    # ICA
    ica = ICA(n_components=14, random_state=98, max_iter=800)
    ica.fit(combined_raw)

    eog_indices, eog_scores = ica.find_bads_eog(combined_raw, ch_name='EOG', threshold=0.9,measure='correlation')
    ica.exclude = eog_indices
    ica.apply(combined_raw)

    emg_indices, emg_scores = ica.find_bads_muscle(combined_raw, )

    ica.exclude += emg_indices

    ica.apply(combined_raw)

    raw_without_eog =combined_raw.drop_channels(['EOG'])
    print("Remaining channels after removing EOG:", raw_without_eog.ch_names)
    raw_without_eog.set_eeg_reference('average', projection=True)


    selected_channels = raw_without_eog.info['ch_names'][2:16]
    print("选取的通道有：", selected_channels)

    data, times = raw_without_eog[2:16, :]
    # zero_count = data.size - np.count_nonzero(data)
    # nan_count = np.isnan(data).sum()
    df = pd.DataFrame(data)

    df_interpolated = df.interpolate(method=interpolation_method, axis=0)
    data = df_interpolated.to_numpy()
    data = interpolate_eeg_data(data, target_length)
    print(data.shape)
    data = data[:, :50 * 60 * 128]
    print(data.shape)
    data = process_data(data)

    data = data.reshape(14, 40 * 5, 128 * 12)
    data = np.transpose(data, (1, 0, 2))
    filter_data = np.empty([200, 14, 128 * 12])
    for trial in range(200):
        for channel in range(14):
            filter_data[trial, channel, :] = butter_bandpass_filter(data[trial, channel, :], 4, 45, 128, order=5)



    fe = FeatureExtractor(sfreq=128,
                          selected_funcs=['mean', 'variance', 'ptp_amp', 'kurtosis', 'zero_crossings', 'skewness',
                                          'line_length'])
    time_features = fe.fit_transform(X=filter_data)

    wavelet_data = discrete_wavelet_transform(filter_data)


    max_diff_features = []
    min_diff_features = []

    for trial in range(200):
        for channel in range(14):

            coeffs = wavelet_data[trial, channel]
            detail_coeffs = coeffs[-1]


            diff = np.diff(detail_coeffs)


            max_diff = np.max(diff)
            min_diff = np.min(diff)


            max_diff_features.append(max_diff)
            min_diff_features.append(min_diff)


    max_diff_features = np.array(max_diff_features).reshape(-1, 14)
    min_diff_features = np.array(min_diff_features).reshape(-1, 14)


    time_features = np.hstack((time_features, max_diff_features))
    time_features = np.hstack((time_features, min_diff_features))
    Time_Features.append(time_features)



    bands = [(4, 8), (8, 15), (15, 30), (30, 45)]
    # [alias_feature_function]__[optional_param]
    params = dict({
        'pow_freq_bands__log': True,
        'pow_freq_bands__normalize': False,
        'pow_freq_bands__freq_bands': bands
    })
    fe = FeatureExtractor(sfreq=128, selected_funcs=['pow_freq_bands'], params=params)
    frequency_features = fe.fit_transform(X=filter_data)
    Frequency_Features.append(frequency_features)


    sample_rate = 128
    sample = data[0, 0, :]

    window_size = int(sample_rate)  # 1s
    step_size = sample_rate * 0.5  # 0.5s
    freqs, times, spectrogram = log_specgram(sample, sample_rate, window_size, step_size)
    #  200 14 23 65
    all_spec_data = np.zeros((data.shape[0], data.shape[1], spectrogram.shape[0], spectrogram.shape[1]))

    # loop each trial
    for i, each_trial in enumerate(filter_data):
        #     print(each_trial.shape) # (channel, seq len) (e.g., 32, 8064)
        for j, each_trial_channel in enumerate(each_trial):
            #         print(each_trial_channel.shape) # (seq len) (e.g., 8064, )
            freqs, times, spectrogram = log_specgram(each_trial_channel, sample_rate, window_size, step_size)
            all_spec_data[i, j, :, :] = spectrogram
    Spatial_Features.append(all_spec_data)

time_features = np.array(Time_Features).reshape(-1, 126)   # -1=20*200  14*9
print("时间形状:", time_features.shape)
time_scaler = StandardScaler()
time_features_standardized = time_scaler.fit_transform(time_features)


frequency_features = np.array(Frequency_Features).reshape(-1, 56) # -1= 20*200 14*4
print("频率形状:", frequency_features.shape)
frequency_scaler = StandardScaler()
frequency_features_standardized = frequency_scaler.fit_transform(frequency_features)


spatial_features = np.array(Spatial_Features).reshape(-1, 14, 23, 65)  # -1=20*200
print("空间形状:", spatial_features.shape)
# spatial_scaler = StandardScaler()
# spatial_features_standardized = spatial_scaler.fit_transform(spatial_features)

standardized_features_path = os.path.join(result_dir, 'standardized_features.mat')
savemat(standardized_features_path, {"time_features": time_features_standardized, "frequency_features": frequency_features_standardized, "spatial_features": Spatial_Features})



