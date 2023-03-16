# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from audioprocessor.processor import mfcc_process
def genre_classifier():
    # Use a breakpoint in the code line below to debug your script.
    dataset_output_path = "data/raw_data"
    json_output_path = "data/json_data/data.json"
    sample_rate = 22050
    audio_duration = 30  # GTZAN project
    samples_per_track = sample_rate * audio_duration
    mfcc_process(dataset_output_path, json_output_path, samples_per_track, sample_rate, audio_duration, num_segments = 10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Main function call")
    #   genre_classifier()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
