import os
import argparse
from helper import constant
from neural_network import NeuralNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def unsigned_greater_than_thero(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("Minimum length is 1")
    return x


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, type=str, help="the MIDI file from which the new song will me generated")
ap.add_argument("-p", "--plot", type=str, default=constant.PLOT_PATH, help="path to output accuracy/loss plot")
ap.add_argument(
    "-s",
    "--sequence_length",
    type=unsigned_greater_than_thero,
    default=constant.SEQUENCE_LENGTH,
    help="number of notes in music to take in account when generating a new note"
)

args = vars(ap.parse_args())

neural_network = NeuralNetwork()
neural_network.train(args)
