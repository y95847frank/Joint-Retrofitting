# Joint-Retrofitting

## Requirements
1. Python3
2. Numpy

## Data
1. Word vector file
2. Lexicon file (provided in eval_data/)

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
python retrofit.py -i word_vec_file -l ontology_file -n num_iter -o out_vec_file
```

## Reference

