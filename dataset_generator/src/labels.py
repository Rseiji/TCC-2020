
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


def is_int(score) -> bool:
    try:
        int(score)
        return True
    except:
        return False


def convert_score_to_int(score) -> int:
    try:
        return int(score)
    except:
        return int(f'{score[1]}1000')


def generate_labels(scores: list) -> list:
    ''' given a list of scores of each play on the chess game generates
    labels 0 (is not a turning point) or 1 (is a turning point) for each play
    '''

    # first label is 0 since the first play has no previous score to comparte
    labels = [0]

    for i in range(1, len(scores)):

        previous_score = convert_score_to_int(scores[i-1])
        current_score = convert_score_to_int(scores[i])

        previous_advantage = get_advantage(previous_score)
        current_advantage = get_advantage(current_score)

        if abs(current_score - previous_score) > ADVANTAGE_THRESHOLD and previous_advantage != current_advantage:
            labels.append(1)
        else:
            labels.append(0)

    return labels
