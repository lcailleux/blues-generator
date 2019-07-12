import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from helper import constant


class NetworkModel:
    def create(self, args, vocab_length):
        """
        :return model: keras.models.Model
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(constant.SEQUENCE_LENGTH, vocab_length)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
            tf.keras.layers.Dense(vocab_length, activation='softmax')
        ])

        model.compile(
            loss=constant.MODEL_LOSS,
            optimizer=tf.keras.optimizers.Adam(lr=constant.LEARNING_RATE),
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

    def generate_music(self, model, input_notes, note_dict, vocab_length):
        backward_dict = dict()
        for note in note_dict.keys():
            index = note_dict[note]
            backward_dict[index] = note

        n = np.random.randint(0, len(input_notes) - 1)
        sequence = input_notes[n]
        start_sequence = sequence.reshape(1, constant.SEQUENCE_LENGTH, vocab_length)
        output = []

        for i in range(0, 100):
            new_note = model.predict(start_sequence, verbose=0)
            index = np.argmax(new_note)
            encoded_note = np.zeros(vocab_length)
            encoded_note[index] = 1
            output.append(encoded_note)
            sequence = start_sequence[0][1:]
            start_sequence = np.concatenate((sequence, encoded_note.reshape(1, vocab_length)))
            start_sequence = start_sequence.reshape(1, constant.SEQUENCE_LENGTH, vocab_length)

        return output, backward_dict

