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
        input_notes, output_notes, vocab_length = self.data_handler.load_dataset(args)
        model = self.network_model.create(args, vocab_length)

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
            input_notes,
            output_notes,
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

            input_notes, output_notes, vocab_length, note_dict = self.data_handler.get_neural_network_notes(
                notes,
                pitches
            )

            output, backward_dict = self.network_model.generate_music(
                model,
                input_notes,
                note_dict,
                vocab_length
            )

            self.data_handler.save_midi(args, output, backward_dict)


