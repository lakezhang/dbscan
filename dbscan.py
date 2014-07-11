#encoding utf-8
import math
import sys
import random
import copy
from collections import defaultdict

def distance(a, b, gauss=True):
    """
        Euclidean distance or gauss kernel
    """
    dim = len(a)
    
    _sum = 0.0
    for dimension in xrange(dim):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq

    if gauss :
        dis = 1 - math.exp(_sum * -1/2)
    else :
        dis = math.sqrt(_sum)

    return dis

def load_file(file_name) :
    """
        Load dataset, for iris dataset
        @return data, count
        count : the records num 
        data : {'f' : feature list , 'l' : origin cluster label}
    """
    data = {}

    count = 0
    with open(file_name) as fp :
        for line in fp :
            line = line.strip()
            if len(line) < 1 :
                continue

            tokens = line.split(',')
            if len(tokens) < 2 :
                continue
            #this last column is label
            data[count] = {'f' : [float(x) for x in tokens[:-1]], 'l' : tokens[-1]}
            count += 1

    return data, count

def neighbour(data, count, eps, gauss=True) :
    """
        Calculate all pair distances between records
        'c' : -2 not visited, -1 : noise, >=0 cluster id
    """
    neighbours = defaultdict(lambda : {'n' : set(), 'l' : '', 'c' : -2, 'm' : 0})

    length = count

    for i in xrange(length - 1) :
        for j in xrange(i + 1, length) :
            dis = distance(data[i]['f'], data[j]['f'], gauss)
            if dis <= eps :
                neighbours[i]['n'].add(j)
                neighbours[j]['n'].add(i)

    for i in xrange(length) :
        neighbours[i]['l'] = data[i]['l']
        neighbours[i]['m'] = len(neighbours[i]['n'])

        # NOISE
        if neighbours[i]['m'] == 0 :
            neighbours[i]['c'] = -1 

        #print i, neighbours[i]['m']

    return neighbours

def select(neighbours) :
    """
        select the not visited record with maximum neighbours
    """
    maxsub = -1
    max_neighbours_num = 0

    for k, v in neighbours.iteritems() :
        if v['c'] == -2 :
            if v['m'] > max_neighbours_num :
                maxsub = k
                max_neighbours_num = v['m']

    #print "SELECT:", maxsub, max_neighbours_num
    return maxsub

def dbscan(file_name, gauss, eps, minpts) :
    """
        dbscan
    """
    data, count = load_file(file_name)
    neighbours = neighbour(data, count, eps, gauss)

    while True :  
        point = select(neighbours)
        if point < 0 :
            break

        if neighbours[point]['m'] >= minpts :
            neighbours[point]['c'] = point
            expand_dbscan(neighbours, point, minpts)
        else :
            neighbours[point]['c'] = -1
            for k, v in neighbours.iteritems() :
                if v['c'] == -2 :
                    neighbours[k]['c'] = -1

    evaluate(neighbours)

def expand_dbscan(neighbours, point, minpts) :
    """
        expand the core point
    """
    candidate = copy.deepcopy(neighbours[point]['n'])

    while len(candidate) > 0:
        p = candidate.pop()
        if neighbours[p]['c'] > -1 :
            continue

        neighbours[p]['c'] = point
        candidate.union(neighbours[p]['n'])


def evaluate(data) :
    """
        evaluate precision and recall
    """
    stat = defaultdict(lambda : defaultdict(lambda : 0))
    for k, v in data.iteritems() :
        stat[v['c']][v['l']] += 1

    wholecount = 0
    wholecorrect = 0
    norecall = 0
    for k, v in stat.iteritems() :
        allcount = 0
        maxj = 0
        maxi = ''
        for i, j in v.iteritems() :
            allcount += j
            if j > maxj :
                maxj = j
                maxi = i


        if k != -1 :
            wholecorrect += maxj
            wholecount += allcount
        else :
            norecall += allcount

        print "CLUSTER: %d ALLNUM: %d CORRECT: %d PRECISION: %.4f LABEL: %s" % (k, allcount, maxj, float(maxj) / allcount, maxi)

    print "ALLNUM: %d CORRECT: %d PRECISION: %.4f RECALL: %.4f" % (wholecount + norecall, wholecorrect, float(wholecorrect) / wholecount, 1 - float(norecall) / (wholecount + norecall))


if __name__ == '__main__':
    
    dbscan('data/iris.data', True, 0.2, 20)
