import tensorflow as tf
from helper import constant
from data_handler import DataHandler
from network_model import NetworkModel


class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get('acc') and logs.get('acc') > 0.99:
            print("\n[INFO] Reached 99% accuracy, stopping training.")
            self.model.stop_training = True


class NeuralNetwork:
    def __init__(self):
        self.data_handler = DataHandler()
        self.network_model = NetworkModel()

    def train_and_run(self, args):
        """

        :param args: dictionary of arguments (dict)
        :return model: keras.models.Model
        """
        # Loading dataset
        input_notes, output_notes, note_dict, vocab_length = self.data_handler.load_dataset(args)

        # Training model
        model = self.train(args, input_notes, output_notes, vocab_length)

        # Generating new music
        output, backward_dict = self.run(model, input_notes, note_dict, vocab_length)

        self.data_handler.save_midi(args, output, backward_dict)

    def train(self, args, input_notes, output_notes, vocab_length):
        model = self.network_model.create(args, vocab_length)
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
            callbacks=[checkpoint]
        )

        self.network_model.plot_loss_and_accuracy(args, history)
        return model

    def run(self, model, input_notes, note_dict, vocab_length):

        output, backward_dict = self.network_model.generate_music(
            model,
            input_notes,
            note_dict,
            vocab_length
        )

        return output, backward_dict


