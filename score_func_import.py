import main
import random


def score_func(network: main.NeuralNetwork):
    x = (random.random() * 4) + 1
    y = (random.random() * 4) + 1
    network.network[0][0].value = x
    network.network[0][1].value = y
    network.run()
    return abs(network.network[-1][0].value - (x + y)), network


def score_func2(network: main.NeuralNetwork):
    x = random.randint(1, 10)
    network.network[0][0].value = x
    network.run()
    return abs(network.network[-1][0].value - (2*((x % 2)-.5))), network


def score_func3(network: main.NeuralNetwork):
    x = 0
    for neuron in network.network[0]:
        x_ = random.uniform(-10, 10)
        x += x_
        neuron.value = x_
    network.run()
    return abs((1*(x>0)+-1*(x<0)-network.network[-1][0].value)), network