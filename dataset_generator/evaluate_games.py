import os
import math
import re

import chess
import chess.engine
import chess.pgn

from ponto_virada import pontosVirada

#faz import da engine para um objeto da biblioteca python chess
engine = chess.engine.SimpleEngine.popen_uci("stockfish_20090216_x64_bmi2")

#intervalo dos pgns a serem analisados
game_init = int(input("number of the first pgn to eval: "))
game_end = int(input("number of the last pgn to eval: "))

#procura arquivo com os scores salvos. Se ele existe, escreve no fim dele. Caso contrário, o arquivo é criado
if os.path.isfile("scores.csv"):
    scores_file = open("scores.csv", "a")
else:
    scores_file = open("scores.csv", "w")
    scores_file.write("Id, Pgn Number, Event, Move Number, Move, Score, Comment, Label\n")

#checa se existe um arquivo contendo o primeiro id dos movimentos a serem inseridos no dataset. Caso contrário, ele é criado
if os.path.isfile("next_id.txt"):
    next_id = open("next_id.txt")
    move_id = int(next_id.readline())
    next_id.close()
else:
    move_id = 0

#loop principal de avaliação das partidas
for i in range(game_init, game_end + 1):

    game_path = f'../../pgn_files/pgn_{i}.txt'
    #checa se pgn i existe
    try:
        if(os.path.isfile(game_path)):
            print(f'Avaliando pgn {i}')

            #abre e lê arquivo pgn
            pgn = open(game_path)
            title = pgn.readline()
            title = re.findall(r'"(.*?)"', title)[0] #titulo da partida, que será inserido no dataset

            #cria objeto para representar a partida partindo do arquivo pgn
            game = chess.pgn.read_game(pgn)
            board = game.board()
            limit = chess.engine.Limit(depth = 15)      #limite de busca da engine
            move_number = 1
            scores = []     #lista com as linhas a serem inseridos no dataset
            score_values = []   #lista com os scores numericos de cada jogada

            turn = 'W'

            #loop de avaliação de uma partida
            for node in game.mainline():
                #insere próximo movimento da partida no tabuleiro
                board.push(node.move)
                info =  engine.analyse(board, limit)

                #tag para diferenciar jogada das brancas e das negras
                if move_number == math.floor(move_number):
                    turn = 'W'
                else:
                    turn = 'B'

                #insere linha no dataset
                scores.append(f'{move_id}, {i}, "{title}", {math.floor(move_number)}{turn}, {node.move}, {info["score"].white()}, "{node.comment}"')
                
                #insere score numerico na lista
                #score_values.append(info["score"].white().score())
                if info["score"].white().score() == None:   #para casos de jogadas proximos a um cheque mate, o score e substituido por um contador
                    mate_score = f'{info["score"].white()}'     #extrai sinal + (brancas) ou - (negras) do contador de jogadas ate mate
                    score_normalized = int(f'{mate_score[1]}1000')  #score numerico para esta situacao vira +1000 ou -1000
                    score_values.append(score_normalized)
                else:
                    score_values.append(info["score"].white().score())

                #print(f'Analisado movimento {math.floor(move_number)}')
                
                move_number += 0.5
                move_id += 1

            pgn.close()

            #cria lista contendo as labels de ponto de virada
            pontos_virada = pontosVirada(score_values)

            #escreve no dataset todas as linhas de avaliação desta partida
            j = 0
            for score in scores:
                scores_file.write(f'{score}, {pontos_virada[j]}')
                scores_file.write("\n")
                j += 1


        else:
            print(f'Arquivo pgn {i} não existe')
    
    except:
        print("Pgn com caracteres nao reconhecidos")
        log = open("log.txt", "a")
        log.write(f'{i}\n')
        log.close()

scores_file.close()

#salva o id do último movimento analisado
next_id = open("next_id.txt", "w")
next_id.write(f'{move_id}')
next_id.close()
