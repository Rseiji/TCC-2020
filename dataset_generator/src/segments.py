def generate_segment_ids(comments: list) -> list:
    ''' given a list of comments of each play on the chess game, it generates
    segment ids that groups certain plays on the game
    '''

    segment_ids = [0]
    current_id = 1 if comments[0] else 0

    for comment in comments[1:]:
        if(comment):
            segment_ids.append(current_id)
            current_id += 1

        else:
            segment_ids.append(current_id)

    return segment_ids
