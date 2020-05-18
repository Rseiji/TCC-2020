#Lê o .csv e faz outro, com as linhas devidamente separadas e identificadas por jogo
import csv
from csv import writer
import sys 

path = sys.argv[1]
file_name = sys.argv[2]

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)



with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        comments = row[0]
        moves = row[1]
        game_number = row[2]

        comments = comments.split('\n')
        comments.pop()
        
        moves = moves.split(',')

        game_number = [game_number for i in range(0,len(moves))]
        

        for i in range(0,len(moves)):
            gn = game_number[i]
            m = moves[i]

            if(i < len(comments)):
                c = comments[i]
            else:
                c = "No Comments"
            
            append_list_as_row(file_name, [gn,m,c])
