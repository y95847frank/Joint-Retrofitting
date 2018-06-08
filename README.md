# Joint-Retrofitting
[Natural Language Processing Laboratory](http://nlg.csie.ntu.edu.tw) at National Taiwan University

## Overview
The joint sense retrofitting model utilizes the contextual and ontological information to derive sense vectors. The sense embedding is learned iteratively via constraining the distance between the sense vector and its word form vector, its sense neighbors and its contextual neighbors. You can use this tool to create sense embedding vector from any trained word vector quickly. Moreover, I provide the evaluation program and four benchmark datasets that can easily test your new sense vector.

## Requirements
1. Python3
2. Numpy

## Data
1. Word vector file

    A file containing a pre-trained word vector model. In word vector model, each line has a word vector as follows :
        `the -1.0 0.1 0.2`

    p.s. You can download pre-trained word vector in [Word2Vec](https://code.google.com/archive/p/word2vec/) or [GloVe](https://nlp.stanford.edu/projects/glove/).

2. Lexicon file (provided in `thesaurus_ontology/`)

    It's an ontology file that contains words and its' synonyms. Each line represents a word and all it's synonyms. The format is :
        `<wordsense><weight> <neighbor-1><weight> <neighbor-2><weight> ...`

    ps. I used [Thesaurus-API](https://github.com/Manwholikespie/thesaurus-api) to parse the ontology.

3. Word similarity evaluation dataset (provided in `eval_data/`)

## Program Execution

```
$ python joint_retrofit.py -i word_vec_file -l lexicon_file -n num_iter -o out_vec_file
-i : path of word vectors input file
-l : path of ontology file
-n : number of iterations (default : n=10)
-o : path of output file
```

Example : 
```
python joint_retrofit.py -i word_vec_file -l ontology_file -n num_iter -o out_vec_file
```

## Evaluation

```
$ python we_sensesim.py word_vec_file
```
This program will show the cosine similarity score of the word vector on each dataset.
In `eval_data/` directory, there are MEN, MTurk, RW, WS353 datasets. You can add more evaluation dataset to test your word vector on your own.


## Reference
- Pennington, J. et al. 2014. Glove: Global vectors for word representation.
- Jauhar, S.K. et al. 2015. Ontologically grounded multi-sense representation learning for semantic vector space models.
- M. Faruqui, J. Dodge, S.K. Jauhar, C. Dyer, E. Hovy and N.A. Smith et al. 2015. Retrofitting word vectors to semantic lexicons.

## How to cite this resource
Please cite the following paper when referring to Joint in academic publications and papers.

`
Ting-Yu Yen, Yang-Yin Lee, Hen-Hsen Huang and Hsin-Hsi Chen (2018). “That Makes Sense: Joint Sense Retrofitting from Contextual and Ontological Information.” In Proceedings of the Web Conference 2018, poster, 23-27 April 2018, Lyon, France.
`

## Contact
Feel free to [contact me](mailto:tyyen@nlg.csie.ntu.edu.tw) if there's any problems.

