import tensorflow as tf
import pickle
from helper import constant
from data_handler import DataHandler
from network_model import NetworkModel


class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get('acc') and logs.get('acc') > 0.99:
            print("\n[INFO] Reached 98% accuracy, stopping training.")
            self.model.stop_training = True


class NeuralNetwork:
    def __init__(self):
        self.data_handler = DataHandler()
        self.network_model = NetworkModel()

    def train(self, args):
        # Loading dataset
        network_input, network_output, vocab_length = self.data_handler.load_dataset(args)
        model = self.network_model.create(network_input, vocab_length)

        # callbacks
        stop_training = StopTrainingCallback()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            constant.MODEL_PATH,
            monitor="acc",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq='epoch'
        )

        history = model.fit(
            network_input,
            network_output,
            epochs=constant.EPOCHS,
            callbacks=[checkpoint, stop_training]
        )

        self.network_model.plot_loss_and_accuracy(args, history)
        return model

    def run(self, args):
        model = tf.keras.models.load_model(args["model"])

        with open(args["notes"], 'rb') as notes_path:
            notes = pickle.load(notes_path)
            pitches = sorted(set(item for item in notes))
            vocab_length = len(set(notes))

            network_input, network_output = self.data_handler.prepare_sequences(notes, vocab_length)
            prediction_output = self.network_model.generate_notes(model, network_input, pitches, vocab_length)

            self.data_handler.save_midi(args, prediction_output)


