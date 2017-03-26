import xgboost as xgb
# read in data
demo_path = '/home/multiangle/software/xgboost/xgboost/'
dtrain = xgb.DMatrix(demo_path+'demo/data/agaricus.txt.train')
dtest = xgb.DMatrix(demo_path+'demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
