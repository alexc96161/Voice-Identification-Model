# Imports
import csv
import random
import os

# Set envourment path
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Seed for Random to make it consitant, but changeable shuffling
Seed = 42

# Get the csv of all training data, and get the unique sentences

ignored = 0
sentences = []
with open('cv-valid-train.csv') as csv_file:
    # Make a reader for the file
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # List the columns
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # Get all of the sentences not already in the sentences list, and tally up ignored sentances
            if not row[1] in sentences:
                sentences.append(row[1])
            else:
                ignored=ignored+1
            line_count += 1
    print(f'Processed {line_count} lines.')

print(str(ignored) + " ignored lines ; " + str(len(sentences)) + " added lines")

# Write a new csv with all of the unque sentences shuffled.

random.Random(Seed).shuffle(sentences)
new_file = open('unique_sentences.csv', 'w')
writer = csv.writer(new_file)
writer.writerow(sentences)
new_file.close()
