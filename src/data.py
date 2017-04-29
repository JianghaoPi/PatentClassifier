with open('../data/train.txt') as file:
    x_train=[]
    y_train=[]
    label_train=[]
    for line in file:
        line = line.split(None, 1)
        if len(line) == 1:
            line += ['']
        lb, ft = line
        xi = {}
        for e in ft.split():
            pos, pos_x = e.split(":")
            xi[int(pos)] = float(pos_x)
        if lb[0] == 'A':
            y_train += [1]
        else:
            y_train += [-1]
        x_train += [xi]
        label_train+=[lb.split(',')[0]]
print('Read training data with %d samples.'%(len(x_train)))
with open('../data/test.txt') as file:
    x_test=[]
    y_test=[]
    label_test=[]
    for line in file:
        line = line.split(None, 1)
        if len(line) == 1:
            line += ['']
        lb, ft = line
        xi = {}
        for e in ft.split():
            pos, pos_x = e.split(":")
            xi[int(pos)] = float(pos_x)
        if lb[0] == 'A':
            y_test += [1]
        else:
            y_test += [-1]
        x_test += [xi]
        label_test+=[lb.split(',')[0]]
print('Read testing data with %d samples.'%(len(x_test)))