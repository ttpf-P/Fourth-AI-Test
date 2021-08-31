import neurons
import random
import copy


class NeuralNetwork:
    def __init__(self, structure, lr=0.05):
        self.network = self.generate(structure)
        self.lr = lr

    def generate(self, structure):
        network = [[neurons.InputNeuron()], [neurons.Neuron()]]
        network[0][0].value = 1
        network[1][0].predecessors = [[network[0][0], 1]]
        return network

    def run(self):
        for layer in self.network:
            for neuron in layer:
                neuron.update()

    def mutate(self, score=1):
        for layer in self.network:
            for neuron in layer:
                neuron.mutate(self.lr, score)

    def __lt__(self, other):
        return True


class NetworkBatch:
    def __init__(self, structure, lr, gen_size):
        self.networks = []
        for _ in range(gen_size):
            self.networks.append(NeuralNetwork(structure, lr))
        self.lr = lr
        self.gen_size = gen_size

    def train(self, score_func, generations, survivor_cut=2):
        for gen in range(generations):
            scores = []
            for network in self.networks:
                scores.append((score_func(network), network))
            scores.sort()
            networks_new = []
            scores_cut = scores[:int(self.gen_size/survivor_cut)]
            for i in range(self.gen_size):
                chosen = random.choice(scores_cut)
                networks_new.append(copy.deepcopy(chosen[1]))
                # networks_new[i].mutate(chosen[0])
                networks_new[i].mutate()
            del self.networks
            self.networks = networks_new
            if gen % 100 == 0:
                print(scores[0][0], "@", scores[0][1].network[-1][0].predecessors[0][1], "  \t",
                      scores[-1][0], "@", scores[0][1].network[-1][0].predecessors[0][1])
            del networks_new
            del scores


if __name__ == "__main__":
    """NN = NeuralNetwork(())
    NN.run()
    print(NN.network[1][0].value)
    NN.mutate()
    NN.run()
    print(NN.network[1][0].value)"""

    def score_func(network: NeuralNetwork):
        network.network[0][0].value = 1
        network.run()
        return abs(network.network[-1][0].value)

    NB = NetworkBatch((), 0.001, 20)
    NB.train(score_func, 20000)
