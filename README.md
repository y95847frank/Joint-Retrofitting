# Joint-Retrofitting

## Requirements
1. Python3
2. Numpy

## Data
1. Word vector file

    A file containing a pre-trained word vector model. In word vector model, each line has a word vector as follows :
    `the -1.0 0.1 0.2`
2. Lexicon file (provided in thesaurus_ontology/)
3. Word similarity evaluation dataset (provided in eval_data/)

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

