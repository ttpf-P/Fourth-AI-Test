import neurons
import random
import copy
import functools


# multiprocessing stuff
import multiprocessing


class NeuralNetwork:
    def __init__(self, structure, lr=0.05):
        self.network = self.generate(structure)
        self.lr = lr

    @staticmethod
    @functools.lru_cache
    def generate(structure) -> list:
        """network = [[neurons.InputNeuron()], [neurons.Neuron()]]
        network[0][0].value = 1
        network[1][0].predecessors = [[network[0][0], 1]]"""
        network = []
        for layer in structure:
            network.append([])
            for _ in range(layer[0]):
                network[-1].append(layer[1]())
                try:
                    for predecessor in network[-2]:
                        network[-1][-1].predecessors.append([predecessor, random.uniform(-10, 10)])
                except IndexError:
                    pass
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
    def __init__(self, structure=((1, neurons.InputNeuron), (1, neurons.Neuron)), lr=0.001, gen_size=200):
        self.networks = []
        for _ in range(gen_size):
            self.networks.append(NeuralNetwork(structure, lr))
        self.lr = lr
        self.gen_size = gen_size

    def train(self, score_func, generations, survivor_cut=2, parallel=False):
        for gen in range(generations):
            scores = []
            if parallel:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                scores = pool.map(score_func, self.networks)
                #pool.close()
            else:
                for network in self.networks:
                    scores.append(score_func(network))

            scores.sort(reverse=True)
            networks_new = []
            scores_cut = scores[:int(self.gen_size/survivor_cut)]
            start_ = time.time_ns()
            """for i in range(self.gen_size):
                chosen = random.choice(scores_cut)
                networks_new.append(copy.deepcopy(chosen[1]))
                networks_new[i].mutate(chosen[0])"""
            if parallel:
                randoms = [random.choice(scores_cut)[1] for _ in range(self.gen_size)]
                """networks_new = pool.map(lambda x: copy.deepcopy(random.choice(x)),
                                        [scores_cut for _ in range(self.gen_size)])"""
                networks_new = pool.map(copy.deepcopy, randoms)
                pool.close()
            else:
                for i in range(self.gen_size):
                    chosen = random.choice(scores_cut)
                    networks_new.append(NeuralNetwork((), lr=self.lr))
                    networks_new[i].network = copy.deepcopy(chosen[1].network)
            print((time.time_ns() - start_) / 10 ** 9)
            for i in range(self.gen_size):
                networks_new[i].mutate(scores_cut[0][0])
            print((time.time_ns() - start_) / 10 ** 9)
            del self.networks
            self.networks = networks_new
            if gen % 1 == 0:
                print(gen, ":", scores[0][0], "@", scores[0][1].network[-1][0].predecessors[0][1], "  \t",
                      scores[-1][0], "@", scores[-1][1].network[-1][0].predecessors[0][1])
            del networks_new
            del scores


if __name__ == "__main__":
    import time
    import score_func_import
    """NN = NeuralNetwork(())
    NN.run()
    print(NN.network[1][0].value)
    NN.mutate()
    NN.run()
    print(NN.network[1][0].value)"""

    """def score_func(network: NeuralNetwork):
        network.network[0][0].value = 1
        network.run()
        return abs(network.network[-1][0].value)"""

    """def score_func(network: NeuralNetwork):
        score = 0
        for x in range(5):
            x = (random.random() * 4) + 1
            y = (random.random() * 4) + 1
            network.network[0][0].value = x
            network.network[0][1].value = y
            network.run()
            score += abs(network.network[-1][0].value - (x + y))
        return score/5"""

    startbatch = time.time_ns()
    NB = NetworkBatch(((4, neurons.InputNeuron), (2, neurons.Neuron), (1, neurons.Neuron)), 0.1, 200000)
    print(NB.networks[0].network)
    print(NB.networks[1].network)
    start = time.time_ns()
    NB.train(score_func_import.score_func3, 1, parallel=True)
    print("training:", (time.time_ns()-start)/10**9, "s")
    print("creating:", (start-startbatch)/10**9, "s")
