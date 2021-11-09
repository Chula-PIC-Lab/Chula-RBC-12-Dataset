import glob
from os import linesep
listfile = glob.glob('Label/*.txt', recursive=True)
for file in listfile:
    s = ''
    with open(file) as f:
        line = f.readline()
        while line:
            print(file)
            x,y, typee = line.strip().split(" ")
            if typee == "9" or typee == "13" or typee == "14"or typee == "15":
                line = f.readline()
                continue
            if int(typee) <9:
                s += line
            else:
                s += x+' '+y+' '+str(int(typee)-1)+'\n'

            line = f.readline()
    with open(file[:5]+'1'+file[5:], 'w') as f:
        f.write(s)