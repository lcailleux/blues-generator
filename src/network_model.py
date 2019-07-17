import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from helper import constant


class NetworkModel:
    def create(self, network_input, vocab_length):
        """
        :return model: keras.models.Model
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                256,
                input_shape=(network_input.shape[1], network_input.shape[2]),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(vocab_length, activation='softmax')
        ])

        model.compile(
            loss=constant.MODEL_LOSS,
            optimizer='adam',
            metrics=['acc']
        )

        return model

    def plot_loss_and_accuracy(self, args, history):
        """
        :param args: dictionary of arguments (dict)
        :param history:
        :return:
        """
        nb_epochs = len(history.history['loss'])

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, nb_epochs), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, nb_epochs), history.history["acc"], label="train_acc")
        plt.title("Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(args["plot"])
        plt.show()

    def generate_notes(self, model, network_input, pitches, vocab_length):
        start = np.random.randint(0, len(network_input) - 1)

        int_to_note = dict((number, note) for number, note in enumerate(pitches))

        pattern = network_input[start]

        prediction_output = []

        for note_index in range(constant.OUTPUT_LENGTH):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(vocab_length)

            prediction = model.predict(prediction_input, verbose=0)

            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)

            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

