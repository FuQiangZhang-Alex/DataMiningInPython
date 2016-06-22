
with open(file='training_set.csv', mode='w') as output:
    for i in range(1, 15):
        filename = str(i) + '.csv'
        csv = open(file=filename, mode='r')
        output.write(csv.read())
