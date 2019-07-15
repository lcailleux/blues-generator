import os
import argparse
from helper import constant
from neural_network import NeuralNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default=constant.DATASET_PATH, help="path to input dataset")
ap.add_argument("-f", "--file", required=True, type=str, help="the MIDI file from which the new song will me generated")
ap.add_argument("-p", "--plot", type=str, default=constant.PLOT_PATH, help="path to output accuracy/loss plot")

args = vars(ap.parse_args())

neural_network = NeuralNetwork()
neural_network.train(args)
