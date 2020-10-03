import os
import math
import re

import chess
import chess.engine
import chess.pgn

engine = chess.engine.SimpleEngine.popen_uci("stockfish_20090216_x64_bmi2")

game_init = int(input("number of the first pgn to eval: "))
game_end = int(input("number of the last pgn to eval: "))

for i in range(game_init, game_end + 1):

    game_path = f'../pgn_files/pgn_{i}.txt'
    if(os.path.isfile(game_path)):
        print(f'Avaliando pgn {i}')

        pgn = open(game_path)
        title = pgn.readline()
        title = re.findall(r'"(.*?)"', title)[0]

        game = chess.pgn.read_game(pgn)
        board = game.board()
        limit = chess.engine.Limit(depth = 15)
        move_number = 1
        scores = []

        turn = 'W'

        for node in game.mainline():
            board.push(node.move)
            info =  engine.analyse(board, limit)

            if move_number == math.floor(move_number):
                turn = 'W'
            else:
                turn = 'B'

            scores.append(f'{title}, {math.floor(move_number)}{turn}, {node.move}, {info["score"].white()}, {node.comment}')

            #print(f'Analisado movimento {math.floor(move_number)}')
            
            move_number += 0.5

        pgn.close()

        save_path = f'../pgn_scores/pgn_{i}_scores.txt'
        scores_file = open(save_path, "w")

        scores_file.write("Event, Move Number, Move, Score, Comment\n")
        for score in scores:
            scores_file.write(score)
            scores_file.write("\n")


    else:
        print(f'Arquivo pgn {i} não existe')