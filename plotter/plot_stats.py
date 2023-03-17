import matplotlib.pyplot as plt
import csv
import pandas as pd
def decode_csv(data_path):

    epoch = []
    training_accuracy = []
    test_accuracy = []
    training_loss = []
    test_loss = []
    print(data_path)
    with open(data_path) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=';')
        next(csvReader, None)
        for i, row in enumerate(csvReader):
            #print(row[0])
            epoch.append(int(row[0]))
            training_accuracy.append(float(row[1]))
            test_accuracy.append(float(row[2]))
            training_loss.append(float(row[3]))
            test_loss.append(float(row[4]))
            #   print(test_loss[i])
    return [epoch, training_accuracy, test_accuracy, training_loss, test_loss]

def plot_stats(data_location, title):
    epoch, training_accuracy, test_accuracy, training_loss, test_loss = decode_csv(data_location)
    fig, axs = plt.subplots(2)
    axs[0].plot(training_accuracy, label="train accuracy")
    axs[0].plot(test_accuracy, label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[0].legend(loc="upper right")
    axs[0].set_title(str(title)+" Accuracy stats")

    axs[1].plot(training_loss, label="train error")
    axs[1].plot(test_loss, label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title(str(title)+" Error stats")

    plt.show()
