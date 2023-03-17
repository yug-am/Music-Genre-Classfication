import json
import os
import math
import librosa



def mfcc_process(dataset_output_path, json_output_path, sampling_per_track, sampling_rate, duration, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    data_set = {
        "genre_dir": [],
        "label": [],
        "mfcc": []
    }

    sampling_segment = int(sampling_per_track / num_segments)
    expected_mfcc_count = math.ceil(sampling_segment / hop_length)

    for i, (dirpath, dirnames, tracknames) in enumerate(os.walk(dataset_output_path)):

        if dirpath is not dataset_output_path:
            genre_label = dirpath.split("/")[-1]
            data_set["genre_dir"].append(genre_label)
            for f in tracknames:
                curr_file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(curr_file_path, sr=sampling_rate)
                for d in range(num_segments):
                    start = sampling_segment * d
                    finish = start + sampling_segment
                    temp_signal = signal[start:finish]
                    mfcc = librosa.feature.mfcc(y=temp_signal, sr= sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc_t = mfcc.T
                    if len(mfcc_t) == expected_mfcc_count:
                        data_set["mfcc"].append(mfcc_t.tolist())
                        data_set["label"].append(i - 1)
                        print("{}, segment:{}".format(curr_file_path, d + 1))
    with open(json_output_path, "w") as fp:
        json.dump(data_set, fp, indent=4)


