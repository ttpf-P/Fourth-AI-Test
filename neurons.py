import random


class Neuron:
    def __init__(self):
        self.predecessors = []   # [[predecessor neuron, weight]]
        self.value = 0

    def update(self):
        value_new = 0
        length = len(self.predecessors)
        for synapse in range(length):
            value_new += self.predecessors[synapse][0].value*self.predecessors[synapse][1]
        self.value = value_new / length
        del value_new
        del length
        del synapse

    def mutate(self, lr, score):
        for synapse in range(len(self.predecessors)):
            rand = random.uniform(-lr * score, lr * score)
            self.predecessors[synapse][1] += rand
            del rand
        del synapse
        del score


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        del self.predecessors

    def update(self):
        pass

    def mutate(self, lr, score):
        pass
