This runs in Python 3

To install dependencies run 
pip install -r requirements.txt

Command line takes usage:
main.py [-h] -alg {rf,svm,nn,all} -n N [-estimators ESTIMATORS]
               [-criterion {gini,entropy}] [-depth DEPTH]
               [-kernel {poly,rbf,sigmoid}] [-gamma GAMMA]
               [-solver {lbfgs,sgd,adam}]
               [-activation {identity,logistic,tanh,relu}] [-alpha ALPHA]
               [-iterations ITERATIONS] [-seed SEED]