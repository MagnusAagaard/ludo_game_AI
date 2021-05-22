import numpy as np
from timeit import default_timer as timer
import logging

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

player_pieces = [1,2,3,0]
enemy_pieces = [[0,0,0,32],[49,59,50,59],[59,59,48,59]]
piece_at_goal_offset = [p - 59 for p in player_pieces]
player_pieces_posterio = [1,2,10,0]
#print(4 -np.count_nonzero(piece_at_goal_offset))
#print(get_safe_pieces(player_pieces, enemy_pieces))
safe_spots, pieces = get_safe_pieces(player_pieces, enemy_pieces)
print(get_danger_pieces(player_pieces, enemy_pieces, safe_spots))
print(check_star_hit(player_pieces, player_pieces_posterio))
enemies_at_goal_offset = [[p[0] - 59, p[1] - 59, p[2] - 59, p[3] - 59] for p in enemy_pieces]
n_enemy_pieces_at_goal_prior = [4 - np.count_nonzero(enemies) for enemies in enemies_at_goal_offset]
enemy_pieces_now = [[0,0,0,32],[0,59,0,59],[59,59,48,59]]
pieces_diff = 0
i = -1
for idx, p in enumerate(enemy_pieces):
    piece_diff = [p[0] - enemy_pieces_now[idx][0], p[1] - enemy_pieces_now[idx][1], p[2] - enemy_pieces_now[idx][2], p[3] - enemy_pieces_now[idx][3]]
    diff = np.count_nonzero(piece_diff)
    if diff != 0:
        i = idx
        pieces_diff += diff
    
if pieces_diff == 1:
    print("WOrks")
    print(i)
else:
    print("Error")
    print(i)

print(np.random.randint(0, 50, 2))
#start = timer()
#dataout = [i%4 for i in range(100000)]
#winrates = [[dataout[:i].count(0)/i*100, dataout[:i].count(1)/i*100, dataout[:i].count(2)/i*100, dataout[:i].count(3)/i*100] for i in range(1,len(dataout)+1)]
#end = timer()
#logging.basicConfig(filename='eval_winrates.log', format='%(message)s', level=logging.INFO)
#for line in winrates:
#    logging.info(line)
#print(end-start)