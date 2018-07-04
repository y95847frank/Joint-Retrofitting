import argparse
import gzip
import math
import numpy
import re
import sys
from collections import defaultdict
from copy import deepcopy
import numpy as np

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')  
  for word, values in wordVectors.items():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')      
  outFile.close()
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, wordVecs):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[words[0]] = [word for word in words[1:]]
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters, w1, w2, w3):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = set()
  for w in lexicon.keys():
      if w.split('%')[0] in wvVocab:
          loopVocab.add(w)

  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    wvVocab = set(newWordVecs.keys())
    for word in loopVocab:
      wordNeighbours = []
      weightNeighbours = []
      numNeighbours = 0.0
      for w in lexicon[word]:
        weight = float(w.split('#')[1])
        if w.split('#')[0] in wvVocab:  #w = 'renew%0#0.6'
          numNeighbours += weight
          weightNeighbours.append(weight*w2)
          wordNeighbours.append(w.split('#')[0])
        elif w.split('%')[0] in wvVocab:  #
          numNeighbours += weight
          weightNeighbours.append(weight*w3)
          wordNeighbours.append(w.split('%')[0])
      
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue

      # the weight of the data estimate if the number of neighbours
      newVec = w1 * numNeighbours * wordVecs[word.split('%')[0]]
      numNeighbours *= w1

      for (ppWord, weight) in zip(wordNeighbours, weightNeighbours):
        newVec += newWordVecs[ppWord] * weight
        numNeighbours += weight
      tmp_vec = newVec/(numNeighbours)
      if word.split('#')[0] not in newWordVecs:
        newWordVecs[word.split('#')[0]] = tmp_vec
      elif np.linalg.norm(tmp_vec-newWordVecs[word.split('#')[0]]) > 0.1:
        newWordVecs[word.split('#')[0]] = tmp_vec
  return newWordVecs
  
if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  args = parser.parse_args()

  wordVecs = read_word_vecs(args.input)
  lexicon = read_lexicon(args.lexicon, wordVecs)
  numIter = int(args.numiter)
  outFileName = args.output
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit(wordVecs, lexicon, numIter, 1.0, 1.0, 1.0), outFileName) 
