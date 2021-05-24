import ludopy
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer
import logging
import os
from multiprocessing import Pool

class QLearningPlayer():
    def __init__(self, filepath):
        self.Q = self.loadQTable(filepath)
        self.numOfPieces = 4
        self.numOfStates = 6 * self.numOfPieces
        self.numOfActions = 4

    def loadQTable(self,filepath):
        print("Loading Q Table.")
        return np.load(filepath)

    def getQValue(self, state, diceIdx, action):
        idx = state+[diceIdx]+[action]
        return self.Q[idx[0],idx[1],idx[2],idx[3],idx[4],idx[5],idx[6],idx[7],idx[8],idx[9],idx[10],idx[11],idx[12],idx[13],idx[14],idx[15],idx[16],idx[17],idx[18],idx[19],idx[20],idx[21],idx[22],idx[23],idx[24],idx[25]]

    def getState(self, playerPieces, enemyPieces):
        def distanceBetweenTwoPieces(piece, enemy, i):
            if enemy == 0 or enemy >= 53 or piece == 0 or piece >= 53:
                return 1000
            enemy_relative_to_piece = (enemy + 13 * i) % 52
            if enemy_relative_to_piece == 0: enemy_relative_to_piece = 52
            distances = [enemy_relative_to_piece - piece, (enemy_relative_to_piece - 52) - piece]
            return distances[np.argmin(list(map(abs,distances)))]

        HOME = 0
        SAFE = 1
        VULNERABLE = 2
        ATTACKING = 3
        FINISHLINE = 4
        FINISHED = 5

        home = [0]
        globes = [1, 9, 14, 22, 27, 35, 40, 48]
        unsafe_globes = [14, 27, 40]

        state = []
        for playerPiece in playerPieces:
            pieceState = [0] * (int)(self.numOfStates / self.numOfPieces)

            #Calculating the relative distance of all the enemy pieces to the players piece
            distanceToEnemy = []
            for i, enemy in enumerate(enemyPieces):
                for enemyPiece in enemy:
                    distanceToEnemy.append(distanceBetweenTwoPieces(playerPiece, enemyPiece, i + 1))

            if playerPiece in home:
                pieceState[HOME] = 1

            if playerPiece in globes:
                pieceState[SAFE] = 1

            vulnerable = any([-6 <= relativePosition < 0 for relativePosition in distanceToEnemy])
            if (vulnerable and playerPiece not in globes) or playerPiece in unsafe_globes: pieceState[VULNERABLE] = 1

            attacking = any([0 < relativePosition <= 6 for relativePosition in distanceToEnemy])
            if attacking: pieceState[ATTACKING] = 1

            if playerPiece >= 53:
                pieceState[FINISHLINE] = 1

            if playerPiece == 59:
                pieceState[FINISHED] = 1

            state += pieceState
        return state

    def getNextAction(self, state, dice, movePieces):
        diceIdx = dice - 1
        bestAction = movePieces[0]
        bestQValue = self.getQValue(state, diceIdx, bestAction)
        for action in movePieces:
            if self.getQValue(state,diceIdx, action) > bestQValue:
                bestAction = action
                bestQValue = self.getQValue(state,diceIdx,action)

        return bestAction

player = QLearningPlayer('BestQTable.npy')

def get_safe_pieces(player_pieces, enemy_pieces):
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

def get_danger_pieces(player_pieces, enemy_pieces, safe_spots):
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

def get_attack_spots(player_pieces, enemy_pieces):
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

def check_star_hit(prior_pieces, player_pieces):
    for i in range(len(prior_pieces)):
        if player_pieces[i] - prior_pieces[i] > 6:
            return True
    return False

def util_func(w, game, dice_in, move_pieces_in, player_pieces_in, enemy_pieces_in):
    n_enemies_home_prior = 12 - np.count_nonzero(enemy_pieces_in)
    n_pieces_home_prior = 4 - np.count_nonzero(player_pieces_in)
    pieces_at_goal_offset = [p - 59 for p in player_pieces_in]
    n_pieces_goal_prior = 4 - np.count_nonzero(pieces_at_goal_offset)
    enemies_at_goal_offset = [[p[0] - 59, p[1] - 59, p[2] - 59, p[3] - 59] for p in enemy_pieces_in]
    n_enemy_pieces_at_goal = [4 - np.count_nonzero(enemies) for enemies in enemies_at_goal_offset]
    safe_spots, n_pieces_safe_prior = get_safe_pieces(player_pieces_in, enemy_pieces_in)
    n_pieces_danger_spot_prior = get_danger_pieces(player_pieces_in, enemy_pieces_in, safe_spots)
    n_pieces_in_attack_spot_prior = get_attack_spots(player_pieces_in, enemy_pieces_in)
    prior_pieces = player_pieces_in
    # remember which piece to move
    #piece_to_move = move_pieces_in[np.random.randint(0, len(move_pieces_in))]
    piece_to_move = move_pieces_in[0]
    max_score = np.min(w)

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
        safe_spots, n_pieces_safe_posterio = get_safe_pieces(player_pieces, enemy_pieces)
        n_pieces_danger_spot_posterio = get_danger_pieces(player_pieces, enemy_pieces, safe_spots)
        n_pieces_in_attack_spot_posterio = get_attack_spots(player_pieces, enemy_pieces)
        hit_star = check_star_hit(prior_pieces, player_pieces)
        
        if n_enemies_home_prior < n_enemies_home_posterio:
            # enemy piece hit home, score += weight[0]
            # Find out which player it was, and how many pieces that player has in goal position
            n_hit_home = n_enemies_home_posterio - n_enemies_home_prior
            pieces_diff = 0
            i = -1
            for idx, p in enumerate(enemy_pieces_in):
                piece_diff = [p[0] - enemy_pieces[idx][0], p[1] - enemy_pieces[idx][1], p[2] - enemy_pieces[idx][2], p[3] - enemy_pieces[idx][3]]
                diff = np.count_nonzero(piece_diff)
                if diff != 0:
                    i = idx
                    pieces_diff += diff
            if pieces_diff > 0 and pieces_diff < 3 and i != -1:
                #score += w[0]
                score += w[0]*n_hit_home*(1+n_enemy_pieces_at_goal[i])
                #score += w[0]*n_hit_home*(max(1,1 + n_enemy_pieces_at_goal[i] - n_pieces_goal_posterio))
            else:
                logging.info("ERROR, more than two enemy pieces hit home? : {}".format(pieces_diff))
                logging.info("Prior pieces: {}, posterio pieces: {}".format(enemy_pieces_in, enemy_pieces))
                #score += w[0]
                score += w[0]*n_hit_home
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
            #print("ATTACK")
            score += w[9]

        if score > max_score:
            piece_to_move = piece
            max_score = score

    return piece_to_move

def evaluate(x, n_games):
    # play game 100 times, using the given weights
    # times won = fitness value
    times_won = 0
    print('Starting eval of {} games...'.format(n_games))
    print('Weights: {}'.format(x))
    logging.info('Starting eval of {} games...'.format(n_games))
    logging.info('Weights: {}'.format(x))
    start = timer()
    for i in range(n_games):
        game = ludopy.Game()
        player_is_a_winner = False
        there_is_a_winner = False
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
            # only do moves for player 0, all other players will move randomly
            piece_to_move = -1
            if player_i == 0:
                if len(move_pieces):
                    piece_to_move = util_func(x, deepcopy(game), dice, move_pieces, player_pieces, enemy_pieces)
            else:
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
        # game done, if first winnner was player 0, increment times_won
        if game.first_winner_was == 0:
            times_won += 1
        if i % 10000 == 0:
            logging.info('Done playing {} games. Won so far: {}'.format(i, times_won))
    end = timer()
    print('Done playing {} games. Games won: {}, time taken: {}'.format(n_games, times_won, end-start))
    logging.info('Done playing {} games. Games won: {}, time taken: {}'.format(n_games, times_won, end-start))
    return times_won

def evaluate_multiprocessing(x):
    times_won = 0
    game = ludopy.Game()
    player_is_a_winner = False
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
        # only do moves for player 0, all other players will move randomly
        piece_to_move = -1
        if player_i == 0:
            if len(move_pieces):
                piece_to_move = util_func(x, deepcopy(game), dice, move_pieces, player_pieces, enemy_pieces)
                #piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
    # game done, if first winnner was player 0, increment times_won
    return game.first_winner_was

def evaluate_qlearning_multiprocessing(i):
    times_won = 0
    game = ludopy.Game()
    player_is_a_winner = False
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
        # only do moves for player 0, all other players will move randomly
        piece_to_move = -1
        if player_i == 0:
            if len(move_pieces):
                piece_to_move = player.getNextAction(player.getState(player_pieces, enemy_pieces), dice, move_pieces)
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
    # game done, if first winnner was player 0, increment times_won
    return game.first_winner_was

def evaluate_qlearning_vs_ga_multiprocessing(i):
    weights = [104.0, 118.0, -80.0, 57.0, 94.0, -19.0, 98.0, -58.0, 69.0, 5.0]
    game = ludopy.Game()
    player_is_a_winner = False
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
        # only do moves for player 0, all other players will move randomly
        piece_to_move = -1
        if player_i == 0:
            if len(move_pieces):
                piece_to_move = util_func(weights, deepcopy(game), dice, move_pieces, player_pieces, enemy_pieces)
        elif player_i == 2:
            if len(move_pieces):
                piece_to_move = player.getNextAction(player.getState(player_pieces, enemy_pieces), dice, move_pieces)
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
    # game done, if first winnner was player 0, increment times_won
    return game.first_winner_was


def main():
    logging.basicConfig(filename='eval_winrates.log', format='%(message)s', level=logging.INFO)
    # With attack positive: WR = 67.181%
    # With attack and scaled enemy hit home: WR = 68.765%
    weights = [104.0, 118.0, -80.0, 57.0, 94.0, -19.0, 98.0, -58.0, 69.0, 5.0]
    # Without attack: WR = 66.973%
    #weights = [104.0, 118.0, -80.0, 57.0, 94.0, -19.0, 98.0, -58.0, 69.0, 0]
    # Basic weights: WR = 63.669%
    #weights = [104.0, 118.0, -80.0, 57.0, 94.0, -19.0, 0, 0, 69.0, 0]
    # Baseline: Random moves: WR = 25.891%
    # Baseline: Move first piece only: WR = 25.585%
    #weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n_games = 500000
    #_ = evaluate(weights, n_games)

    print('Starting eval of {} games...'.format(n_games))
    print('Weights: {}'.format(weights))
    logging.info('Starting eval of {} games...'.format(n_games))
    logging.info('Weights: {}'.format(weights))
    start = timer()
    # Q-learning:
    #with Pool(os.cpu_count()-1) as pool:
    #    dataout = pool.map(evaluate_qlearning_vs_ga_multiprocessing, [i for i in range(n_games)])
    # GA:
    with Pool(os.cpu_count()-1) as pool:
        dataout = pool.map(evaluate_multiprocessing, [weights for i in range(n_games)])
    winrates = [[dataout[:i].count(0)/i*100, dataout[:i].count(1)/i*100, dataout[:i].count(2)/i*100, dataout[:i].count(3)/i*100] for i in range(1,len(dataout)+1)]
    end = timer()
    #print(end-start)
    logging.info('Done playing {} games. Games won: {}, time taken: {}'.format(n_games, dataout.count(0), end-start))
    for line in winrates:
        logging.info(line)
    print('Done playing {} games. Games won: {}, time taken: {}'.format(n_games, dataout.count(0), end-start))
    

if __name__ == "__main__":
    main()
