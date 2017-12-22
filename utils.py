import numpy as np
import pdb

def trim(w, wvs):
    if w not in wvs:
        if w[:-1] in wvs: return w[:-1]
        elif w.replace('-','') in wvs: return w.replace('-','')
        elif w[:-4] + 'lize' in wvs: return w[:-4] + 'lize'
    return w

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#def readDataset(filename, wvs=[], no_skip=False, sort=True, printout=False):
def readDataset(filename, no_skip=False, sort=True, printout=False):
    pairs = []
    scores = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            split = line.split()
            #if no_skip or not wvs or (split[0].lower() in wvs and split[1].lower() in wvs):
            pairs.append([ split[0].lower(), split[1].lower(), float(split[2]) ])
    if sort:
        pairs = sorted(pairs, key=lambda x: x[2])
    if printout:
        for pair in pairs:
            print("  %+14s %+14s :  %.2f" % (pair[0], pair[1], pair[2]))
    for pair in pairs:
        scores.append(pair[2])
    return (pairs, scores)

def readWordVecs(filename):
    wvs = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
      line = line.strip()
      if line != '':
        split = line.split()
        word = split[0]
        vec = []
        #try:
        wvs[word] = np.array([float(split[i]) for i in range(1, len(split))], dtype='float64')
        #except:
            # load word vector error, replace with zeros
            #continue
            #wvs[word] = np.array([float(0.0) for i in range(1, len(split))], dtype='float64')
            #pdb.set_trace()
            #print('error load word vector')
    return wvs


def readWordVecsList(filename):
    words = []
    vecs = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line != '':
            split = line.split()
            words.append(split[0])
            vec = []
            for i in range(1, len(split)):
                vec.append(float(split[i]))
            vecs.append(np.array(vec, dtype='float64'))
    vecs = np.asarray(vecs, dtype='float64')
    return (words, vecs)
