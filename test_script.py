import numpy as np

player_pieces = [0,53,1,59]
enemy_pieces = [[0,0,0,32],[48,59,41,59],[59,59,54,59]]
piece_at_goal_offset = [p - 59 for p in player_pieces]
print(4 -np.count_nonzero(piece_at_goal_offset))