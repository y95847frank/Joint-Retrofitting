import numpy as np
import sys
import utils
import os

from collections import defaultdict
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from scipy.stats import spearmanr, pearsonr
from utils import trim
import pdb

"""
    Sense embedding format: see https://github.com/sjauhar/SenseRetrofit
    Use ',' to seperate Datasets
"""
def run(path, fname):
    '''
    if len(sys.argv) != 3:
        print("Usage: python we_sensesim.py SenseEmbedding Datasets")
        exit(0)
    '''
    wvs = utils.readWordVecs(sys.argv[1])
    print("Finish reading vector!")
    wvssen = {}
    s_list = defaultdict(list)
    for sense in wvs:
        wvssen[sense.split("%")[0]] = ''
        s_list[sense.split("%")[0]].append(sense)
    mean_vector = np.mean(wvs.values(), axis=0)
    
    spear_score_max = []
    spear_score_avg = []
    f_name = []

    for name in fname:
        filenames = os.path.join(path, name)
        #full_path = os.path.join(path, name)
        #filenames = os.path.expanduser(full_path).split(',')
        pairs, scores = utils.readDataset(filename, no_skip=True)
        coefs_max = []
        coefs_avg = []
        missing = 0
        for pair in pairs:
            vecs0 = []
            trimed_p0 = trim(pair[0], wvssen)
            if trimed_p0 not in wvssen:
                vecs0.append(mean_vector)
                missing += 1
            else:
                for sense in s_list[trimed_p0]:
                    vecs0.append(wvs[sense])
            vecs1 = []
            trimed_p1 = trim(pair[1],wvssen)
            if trimed_p1 not in wvssen:
                vecs1.append(mean_vector)
                missing += 1
            else:
                for sense in s_list[trimed_p1]:
                    vecs1.append(wvs[sense])
            '''
                max_value and avg_value: see "Multi-Prototype Vector-Space Models of Word Meaning" section 3.2 Measuring Semantic Similarity
                http://www.cs.utexas.edu/~ml/papers/reisinger.naacl-2010.pdf
            ''' 
            max_value = max([1-cosine(a,b) for a in vecs0 for b in vecs1])
            avg_value = np.mean([1-cosine(a,b) for a in vecs0 for b in vecs1])
            coefs_max.append(max_value)
            coefs_avg.append(avg_value)
            
        spear_max = spearmanr(scores, coefs_max)
        pearson_max = pearsonr(scores, coefs_max)
        spear_avg = spearmanr(scores, coefs_avg)
        pearson_avg = pearsonr(scores, coefs_avg)
        spear_score_max.append(spear_max[0])
        spear_score_avg.append(spear_avg[0])
    print 'type     \t',
    for i in range(len(fname)):
        print fname[i],
    print '\nspear max\t',
    for i in range(len(fname)):
        print '%.06f   ' % (spear_score_max[i]),
    print '\nspear avg\t',
    for i in range(len(fname)):
        print '%.06f   ' % (spear_score_avg[i]),
    
if __name__ == "__main__":
    run('./eval_data', ['EN-MEN.txt', 'EN-TRUK.txt', 'EN-RW.txt', 'EN-WS353.txt'])

