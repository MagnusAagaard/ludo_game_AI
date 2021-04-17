import ludopy
import numpy as np
from copy import deepcopy
import logging
from timeit import default_timer as timer

class GeneticAlgorithm:
	def __init__(self):
		# Bounds the weights for each variable
		self.bounds =  [-128, 128]
		# Number of vars for the objective function
		self.vars = 10
		# define the total iterations
		self.n_iter = 100
		# bits per variable
		self.n_bits = 8
		# define the population size
		self.n_pop = 50
		# crossover rate
		self.r_cross = 0.9
		# mutation rate
		self.r_mut = 1.0 / (float(self.n_bits) * self.vars)
		self.init_pop()

	def init_pop(self):
		# initial population of random bitstring
		self.pop = [np.random.randint(0, 2, self.n_bits*self.vars).tolist() for _ in range(self.n_pop)]

	def evolve(self):
		self.best = 0
		self.best_weights = self.decode(self.pop[0])
		# enumarate generations
		for gen in range(self.n_iter):
			# decode population
			decoded = [self.decode(p) for p in self.pop]
			# evaluate all candidates in the population
			scores = [self.objective(d) for d in decoded]
			# check for new best solution
			for i in range(self.n_pop):
				if scores[i] > self.best:
					self.best, self.best_weights = scores[i], self.decode(self.pop[i])
					print('Gen: {}, new best: {}/100 wins, weights: {}'.format(gen, self.best, self.best_weights))
			logging.info('Gen: {}, best: {}/100 wins, weights: {}'.format(gen, self.best, self.best_weights))
			# select parents
			selected = [self.selection(self.pop, scores) for _ in range(self.n_pop)]
			# create next generation
			children = list()
			for i in range(0, self.n_pop, 2):
				# get selected parents in pairs
				p1, p2 = selected[i], selected[i+1]
				# crossover and mutation
				for c in self.crossover(p1, p2):
					# mutation
					self.mutation(c)
					# store for next generation
					children.append(c)
			# replace population
			self.pop = children
		return (self.best, self.best_weights)

	def get_safe_pieces(self, player_pieces, enemy_pieces):
		safe_spots = [0, 1, 9, 14, 22, 27, 35, 48, 53, 54, 55, 56, 57, 58, 59]
		safe_pieces = 0
		for i in range(len(player_pieces)-1):
			for j in range(i+1,len(player_pieces)):
				if player_pieces[i] == player_pieces[j] and player_pieces[i] not in safe_spots:
					safe_spots.append(player_pieces[i])
		
		for piece in player_pieces:
			for spot in safe_spots:
				# if piece is on safe spot which is not home
				if piece == spot and spot != 0:
					safe_pieces += 1
		return safe_spots, safe_pieces
	
	def get_danger_pieces(self, player_pieces, enemy_pieces, safe_spots):
		danger_pieces = 0
		danger_spots = []
		invalid_enemy = [0, 53, 54, 55, 56, 57, 58, 59]
		# correct enemy positions to be as seen from player 1...
		i = 0
		enemy_pieces_corrected = []
		for enemies in enemy_pieces:
			i += 1
			tmp = []
			for enemy in enemies:
				if enemy not in invalid_enemy:
					val = (enemy + i*13) % 52
					if val == 0:
						val = 52
					tmp.append(val)
				else:
					tmp.append(enemy)
			enemy_pieces_corrected.append(tmp)

		for piece in player_pieces:
			for enemies in enemy_pieces_corrected:
				for enemy in enemies:
					if piece - enemy <= 6 and piece - enemy > 0 or piece - enemy >= -50 and piece - enemy <= -46:
						if piece not in safe_spots and enemy not in invalid_enemy and piece not in danger_spots:
							danger_spots.append(piece)
		return len(danger_spots)

	def get_attack_spots(self, player_pieces, enemy_pieces):
		attack_spots = []
		invalid_enemy = [0, 53, 54, 55, 56, 57, 58, 59]
		# correct enemy positions to be as seen from player 1...
		i = 0
		enemy_pieces_corrected = []
		for enemies in enemy_pieces:
			i += 1
			tmp = []
			for enemy in enemies:
				if enemy not in invalid_enemy:
					val = (enemy + i*13) % 52
					if val == 0:
						val = 52
					tmp.append(val)
				else:
					tmp.append(enemy)
			enemy_pieces_corrected.append(tmp)

		for piece in player_pieces:
			for enemies in enemy_pieces_corrected:
				for enemy in enemies:
					if enemy - piece <= 6 and enemy - piece > 0:
						attack_spots.append(piece)
		return attack_spots

	def check_star_hit(self, prior_pieces, player_pieces):
		for i in range(len(prior_pieces)):
			if player_pieces[i] - prior_pieces[i] > 6:
				return True
		return False

	# ref code: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
	# ref paper: http://airccse.org/journal/ijaia/papers/3112ijaia11.pdf
	def util_func(self, w, game, dice, move_pieces_in, player_pieces_in, enemy_pieces_in):
		n_enemies_home_prior = 12 - np.count_nonzero(enemy_pieces_in)
		n_pieces_home_prior = 4 - np.count_nonzero(player_pieces_in)
		pieces_at_goal_offset = [p - 59 for p in player_pieces_in]
		n_pieces_goal_prior = 4 - np.count_nonzero(pieces_at_goal_offset)
		safe_spots, n_pieces_safe_prior = self.get_safe_pieces(player_pieces_in, enemy_pieces_in)
		n_pieces_danger_spot_prior = self.get_danger_pieces(player_pieces_in, enemy_pieces_in, safe_spots)
		n_pieces_in_attack_spot_prior = self.get_attack_spots(player_pieces_in, enemy_pieces_in)
		prior_pieces = player_pieces_in
		# remember which piece to move
		piece_to_move = move_pieces_in[0]
		max_score = 0

		for piece in move_pieces_in:
			score = 0
			# create deepcopy of game instance in order to do a move and see what happens
			g = deepcopy(game)
			_, _, _, _, _, there_is_a_winner = g.answer_observation(piece)
			# check what happened for player 0
			player_pieces, enemy_pieces = g.get_pieces(0)
			# calculate score
			n_enemies_home_posterio = 12 - np.count_nonzero(enemy_pieces)
			n_pieces_home_posterio = 4 - np.count_nonzero(player_pieces)
			pieces_at_goal_offset = [p - 59 for p in player_pieces]
			n_pieces_goal_posterio = 4 - np.count_nonzero(pieces_at_goal_offset)
			safe_spots, n_pieces_safe_posterio = self.get_safe_pieces(player_pieces, enemy_pieces)
			n_pieces_danger_spot_posterio = self.get_danger_pieces(player_pieces, enemy_pieces, safe_spots)
			n_pieces_in_attack_spot_posterio = self.get_attack_spots(player_pieces, enemy_pieces)
			hit_star = self.check_star_hit(prior_pieces, player_pieces)

			if n_enemies_home_prior < n_enemies_home_posterio:
				# enemy piece hit home, score += weight[0]
				score += w[0]
				#print("enemy hit home: {},{}".format(n_enemies_home_prior, n_enemies_home_posterio))
			if n_pieces_home_posterio < n_pieces_home_prior:
				# own piece moved out from start, score += weight[1]
				score += w[1]
				#print("own piece out of goal: {},{}".format(n_pieces_home_prior, n_pieces_home_posterio))
			if n_pieces_home_posterio > n_pieces_home_prior:
				# own piece hit home..., score += weight[2]
				score += w[2]
				#print("hit myself home: {},{}".format(n_pieces_home_prior, n_pieces_home_posterio))
			if n_pieces_goal_posterio > n_pieces_goal_prior:
				# piece moved to goal, score += weight[3]
				score += w[3]
				#print("piece moved to goal: {},{}".format(n_pieces_goal_prior, n_pieces_goal_posterio))
			if n_pieces_safe_prior < n_pieces_safe_posterio:
				# piece moved to safe spot, score += weight[4]
				score += w[4]
				#print("piece in safe spot: {},{}".format(n_pieces_safe_prior, n_pieces_safe_posterio))
			if n_pieces_safe_prior > n_pieces_safe_posterio:
				# piece moved out of safe spot, score += weight[5]
				score += w[5]
				#print("piece out of safe spot: {},{}".format(n_pieces_safe_prior, n_pieces_safe_posterio))
			if n_pieces_danger_spot_prior > n_pieces_danger_spot_posterio:
				# piece moved out of danger, score += weight[6]
				score += w[6]
				#print("piece out of danger spot: {},{}".format(n_pieces_danger_spot_prior, n_pieces_danger_spot_posterio))
			if n_pieces_danger_spot_prior < n_pieces_danger_spot_posterio:
				# piece moved into dangerous spot, score += weight[7]
				score += w[7]
				#print("piece into danger spot: {},{}".format(n_pieces_danger_spot_prior, n_pieces_danger_spot_posterio))
			if hit_star:
				# piece landed on star, score += weight[8]
				score += w[8]
				#print("Hit star: {}".format(hit_star))
			if n_pieces_in_attack_spot_prior < n_pieces_in_attack_spot_posterio:
				# piece moved to attack spot
				print("ATTACK")
				score += w[9]

			if score > max_score:
				piece_to_move = piece
				max_score = score

		return piece_to_move


	def objective(self, x):
		# play game 100 times, using the given weights
		# times won = fitness value
		times_won = 0
		start = timer()
		for i in range(100):
			game = ludopy.Game()
			player_is_a_winner = False
			there_is_a_winner = False
			while not there_is_a_winner:
				(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
				# only do moves for player 0, all other players will move randomly
				piece_to_move = -1
				if player_i == 0:
					if len(move_pieces):
						piece_to_move = self.util_func(x, deepcopy(game), dice, move_pieces, player_pieces, enemy_pieces)
				else:
					if len(move_pieces):
						piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
				_, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
			# game done, if first winnner was player 0, increment times_won
			if game.first_winner_was == 0:
				times_won += 1
		end = timer()
		logging.info('Done playing 100 games for a single member in the population. Games won: {}, time taken: {}'.format(times_won, end-start))
		return times_won

	def decode(self, bitstring):
		decoded = list()
		largest = 2**self.n_bits
		for i in range(self.vars):
			# extract the substring
			start, end = i * self.n_bits, (i * self.n_bits)+self.n_bits
			substring = bitstring[start:end]
			# convert bitstring to a string of chars
			chars = ''.join([str(s) for s in substring])
			# convert bitstring to integer
			integer = int(chars, 2)
			# scale integer to desired range
			value = self.bounds[0] + (integer/largest) * (self.bounds[1] - self.bounds[0])
			# store
			decoded.append(value)
		return decoded

	# tournament selection
	def selection(self, pop, scores, k=3):
		# first random selection
		selection_ix = np.random.randint(len(pop))
		for ix in np.random.randint(0, len(pop), k-1):
			# check if better (e.g. perform a tournament)
			if scores[ix] > scores[selection_ix]:
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
	def mutation(self, bitstring):
		for i in range(len(bitstring)):
			# check for a mutation
			if np.random.rand() < self.r_mut:
				# flip the bit
				bitstring[i] = 1 - bitstring[i]


if __name__ == "__main__":
	logging.basicConfig(filename='run.log', format='%(asctime)s %(message)s', level=logging.INFO)
	ga = GeneticAlgorithm()
	best, best_weights = ga.evolve()
	logging.info('Done! Best score: {}/100 wins! Weights: {}'.format(best, best_weights))
	print('Done!')
	print('Best score: {}/100 wins'.format(best))
	print('Weights: {}'.format(best_weights))