import ludopy
import numpy as np

class GeneticAlgorithm:
    def __init__(self):
        self.g = ludopy.Game()
        self.there_is_a_winner = False
        # Bounds the weights for each variable
        self.bounds =  [-100, 100]
        # Number of vars for the objective function
        self.vars = 5
        # define the total iterations
        self.n_iter = 100
        # bits per variable
        self.n_bits = 16
        # define the population size
        self.n_pop = 100
        # crossover rate
        self.r_cross = 0.9
        # mutation rate
        self.r_mut = 1.0 / (float(self.n_bits) * self.vars)

    def play(self):
        # initial population of random bitstring
	    self.pop = [np.random.randint(0, 2, self.n_bits*self.vars).tolist() for _ in range(self.n_pop)]
        # keep track of best solution
	    self.best, self.best_eval = 0, self.objective(self.pop[0])
        while not self.there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = self.g.get_observation()
            print(player_i, dice)
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, _, self.there_is_a_winner = self.g.answer_observation(piece_to_move)
    
    # ref code: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # ref paper: http://airccse.org/journal/ijaia/papers/3112ijaia11.pdf
    def objective(self, x):
        '''
        inputs:
            x is "weights"
            s is states
        outputs:
            fitness value
        '''
        return x

    def decode(self, bitstring):
        decoded = list()
        largest = 2**self.n_bits
        for i in range(self.vars):
            # extract the substring
            start, end = i * n_bits, (i * n_bits)+n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert bitstring to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[0] + (integer/largest) * (bounds[1] - bounds[0])
            # store
            decoded.append(value)
        return decoded

    # tournament selection
    def selection(self, pop, scores, k=3):
        # first random selection
        selection_ix = np.random.randint(len(pop))
        for ix in np.random.randint(0, len(pop), k-1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(self, p1, p2):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < self.r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1)-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(bitstring):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < self.r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.play()
    