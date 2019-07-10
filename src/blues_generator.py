import os
import argparse
from helper import constant
from neural_network import NeuralNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=constant.MODEL_PATH, help="path to output model")
    ap.add_argument("-d", "--dataset", default=constant.DATASET_PATH, help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default=constant.PLOT_PATH, help="path to output accuracy/loss plot")
    ap.add_argument(
        "-i",
        "--instrument",
        type=str,
        choices=['Electric Guitar', 'Piano'],
        default=constant.DESIRED_INSTRUMENT,
        help="instrument for the new song"
    )
    ap.add_argument(
        "-l",
        "--sequence_length",
        type=int,
        default=constant.SEQUENCE_LENGTH,
        help="number of notes in the new song"
    )

    args = vars(ap.parse_args())

    neural_network = NeuralNetwork()
    neural_network.train_and_run(args)
