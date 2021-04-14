import ludopy
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer

def get_safe_pieces(player_pieces, enemy_pieces):
    safe_spots = []
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

def get_danger_pieces(player_pieces, enemy_pieces, safe):
    danger_pieces = 0
    danger_spots = []
    safe_spots = [0, 1, 9, 14, 22, 27, 35, 40, 48, 54, 55, 56, 57, 58, 59]
    invalid_enemy = [0, 54, 55, 56, 57, 58, 59]
    # add detected safe spots from get_safe_pieces() to list
    for spot in safe:
        if spot not in safe_spots:
            safe_spots.append(spot)
    for piece in player_pieces:
        for enemies in enemy_pieces:
            for enemy in enemies:
                if piece - enemy <= 6 and piece - enemy > 0 or piece - enemy >= -50 and piece - enemy <= -46:
                    if piece not in safe_spots and enemy not in invalid_enemy and piece not in danger_spots:
                        danger_spots.append(piece)
    return len(danger_spots)

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
    safe_spots, n_pieces_safe_prior = get_safe_pieces(player_pieces_in, enemy_pieces_in)
    n_pieces_danger_spot_prior = get_danger_pieces(player_pieces_in, enemy_pieces_in, safe_spots)
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
        safe_spots, n_pieces_safe_posterio = get_safe_pieces(player_pieces, enemy_pieces)
        n_pieces_danger_spot_posterio = get_danger_pieces(player_pieces, enemy_pieces, safe_spots)
        hit_star = check_star_hit(prior_pieces, player_pieces)
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

        if score > max_score:
            piece_to_move = piece
            max_score = score

    return piece_to_move

def evaluate(x):
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
                    piece_to_move = util_func(x, deepcopy(game), dice, move_pieces, player_pieces, enemy_pieces)
            else:
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
        # game done, if first winnner was player 0, increment times_won
        if game.first_winner_was == 0:
            times_won += 1
    end = timer()
    print('Done playing 100 games. Games won: {}, time taken: {}'.format(times_won, end-start))
    return times_won

def main():
    weights = [10, 20, -50, 15, 5, -5, 2, -2, 8]
    evaluate(weights)

if __name__ == "__main__":
    main()