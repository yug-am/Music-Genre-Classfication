import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from keras.callbacks import CSVLogger
from contextlib import redirect_stdout
def basic_neural_network(data_path ):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["label"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    csv_logger = CSVLogger('logs/basicnn_log.csv', append=True, separator=';')
    model.fit(X_train, y_train, callbacks=[csv_logger], validation_data=(X_test, y_test), batch_size=32, epochs=50)

