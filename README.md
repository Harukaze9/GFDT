# GFDT
**GFDT** is a fast algorithm for graph classification without preprocessing for feature representation. 


## About GFDT
GFDT constructs a _Graph Fragment Decision Tree_, a decision tree that has graph-extend operation on its inner nodes.
Then, each path of decision tree represents a graph pattern.
Please see the original paper for details.

## Reference
This implementation of GFDT is created by Haruki Sakagami and Hiroki Arimura, based on following paper.

 `坂上 陽規，栗田 和宏 ，瀧川 一学 ，有村 博紀 ，”決定化されたグラフパターントライの学習アルゴリズム”，人工知能学会 第105回人工知能基本問題研究会 (SIG-FPAI) 予稿集, SIG-FPAI-B508, pp. 63-68, 2018年01月`

## How to run
### Example

`main.py` execute a cross validation on given dataset.  

`python3 main.py -data ./dataset/example.txt`

Output
```
input file:  ./dataset/example.txt
5CV , random_seed_cv: None
average test acc:       0.8666666666666666
average train acc:      1.0
execution time of main.py: 0:00:00.005572
```

`main_rf.py` is a random forest(ensemble learning) version of `main.py`.

`python3 main_rf.py -data ./dataset/example.txt`
```
input file:  ./dataset/example.txt
5CV , random_seed_cv: None
average test acc:       0.9333333333333332
average train acc:      0.9777777777777779
execution time of main.py: 0:00:00.025505
```
## Parameters

### CV parameters

- -data: file name
- -ncv: number of folds(default=5)


### GFDT parameters

- -r: saturation ratio of saturation split method (default=1.0)
- -d: max depth (default=100)
- -mss: minimum number of samples required to split an internal node(default=2)
- -s: minimum number of samples in a leaf(default=1)

Following parameters are for random forest(ensemble learning).

- -tree: number of trees (default=10)
- -res: ratio of randomly chosen feature subset to original feature set(default=0.5)



## Lisence
MIT