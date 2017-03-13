import re
import sys

# regex to get alphanumeric text and puntuation
get_con_regex = '[^+$A-Z?0-9]$'
get_line_id_regex = 'L0-9'
get_text_regex = '[^a-zA-Z0-9_.,;:!?"\'/$]'

# open movie conversations to get chronlogical order of conversations
movie_conversations = open('movie_conversations.txt', 'r')
conv_temp = open('con_temp.text', 'w+')
movie_lines = open('movie_lines.txt', 'r')
# open file to write cleaned data to
ml_temp = open('ml_temp.txt', 'w+')

for line in movie_conversations:
    for line_id in line.split():
        if re.search(get_con_regex, line_id):  # Check regex match
            conv_temp.write(line_id)
    conv_temp.write('\n')

for line in movie_lines:
    for word in line.split():
        if not re.search(get_text_regex, word):  # Check regex match
            ml_temp.write(word)        # Add match to file
            ml_temp.write(" ")         # Add space to text
    ml_temp.write("\n")                # Skip to next line

conv_temp.close()
ml_temp.close()

conv_temp = open('con_temp.text', 'aw+')
ml_temp = open('ml_temp.txt', 'r')
sorted_lines = open('sorted_lines.txt', 'a')
# save movie lines to dictionary
movie_lines_dictionary = {}
for mline in ml_temp:
    line_id = mline.partition(" ")[0]
    mline1 = mline.split(' ', 1)[1]
    movie_lines_dictionary[line_id] = mline1

# find movie lines an sort them in chronlogical order
for line in conv_temp:
    for lid in line.split():
        lid = lid.replace("[", "")
        lid = lid.replace("]", "")
        lid = lid.replace("\'", "")
        ids = lid.split(",")
        for i in ids:
            sorted_lines.write(i + " ")
            sorted_lines.write(movie_lines_dictionary[i])
            sorted_lines.write("\n")
# Gracefully close files
movie_conversations.close()
conv_temp.close()
movie_lines.close()
sorted_lines.close()
ml_temp.close()
