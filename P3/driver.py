from RL import finiteMDP
from sklearn.externals import joblib


fns = ('fmdp.pkl', 'fmdp1.pkl', 'fmdp2.pkl', 'traj.pkl', 'traj1.pkl', 'traj2.pkl')

fmdp, fmdp1, fmdp2, traj, traj1, traj2 = map(joblib.load, fns)

