import main
import random


def score_func(network: main.NeuralNetwork):
    x = (random.random() * 4) + 1
    y = (random.random() * 4) + 1
    network.network[0][0].value = x
    network.network[0][1].value = y
    network.run()
    return abs(network.network[-1][0].value - (x + y)), network
