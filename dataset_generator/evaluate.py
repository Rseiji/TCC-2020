import chess
import chess.engine
import chess.pgn

import math

engine = chess.engine.SimpleEngine.popen_uci("stockfish_20090216_x64_bmi2")

pgn = open("game1.txt")

game = chess.pgn.read_game(pgn)
board = game.board()
limit = chess.engine.Limit(depth = 23)
move_number = 1
scores = []

for move in game.mainline_moves():
    board.push(move)
    info =  engine.analyse(board, limit)

    print(board)
    print(f'Move: {math.floor(move_number)}')
    print(f'Score: {info["score"]}')
    print('\n')

    scores.append(f'{math.floor(move_number)}. {move} - cp {info["score"]}')
    
    move_number += 0.5

for score in scores:
    print(score)