
def pontosVirada (scores):
    pontos_virada = [0]

    for i in range(1, len(scores)):
        if type(scores[i]) == int and type(scores[i-1]) == int:
            if abs(scores[i]) >= (abs(scores[i - 1]) + 150) and abs(scores[i - 1]) < 150:
                pontos_virada.append(1)
            else:
                pontos_virada.append(0)
        else:
            pontos_virada.append(0)
    
    return pontos_virada