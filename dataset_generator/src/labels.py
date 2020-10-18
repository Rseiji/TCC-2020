
ADVANTAGE_THRESHOLD = 100


class Advantage:
    WHITES = 1
    BLACKS = 2
    BALANCED = 3


def get_advantage(score: int) -> int:
    if score < - ADVANTAGE_THRESHOLD:
        return Advantage.BLACKS
    elif score > ADVANTAGE_THRESHOLD:
        return Advantage.WHITES
    else:
        return Advantage.BALANCED


def generate_labels(scores: list) -> list:
    ''' given a list of scores of each play on the chess game generates
    labels 0 (is not a turning point) or 1 (is a turning point) for each play
    '''

    # first label is 0 since the first play has no previous score to comparte
    labels = [0]

    for i in range(1, len(scores)):

        # there's some non numeric values on score list
        if type(scores[i]) == int and type(scores[i-1]) == int:
            previous_score = scores[i-1]
            previous_advantage = get_advantage(previous_score)
            current_score = scores[i]
            current_advantage = get_advantage(current_score)

            if abs(current_score - previous_score) > ADVANTAGE_THRESHOLD and previous_advantage != current_advantage:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)

    return labels
