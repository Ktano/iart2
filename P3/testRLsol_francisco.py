import numpy as np
from sklearn.externals import joblib
from RL_francisco import *
import RLsol_francisco


def test_Q(fmdp, Q, err_threshold=.3):
 
    print("Valores Q aprendidos")

    print(Q)

    err = np.linalg.norm(Q-fmdp.Q)

    print("err (Q - VI)", err)
    if err<err_threshold:
        print("Erro nos Q dentro dos limites de tolerância. OK\n")
    else:
        print("Erro nos Q acima dos limites de tolerância. FAILED\n")

def test_policy(runPolicy, policy, iterations=4, state_initial=5, j_threshold=.7):
    
    J,trajlearn = runPolicy(iterations, state_initial, policy)
    
    print("Trajectoria gerada com a politica aprendida")
    print(trajlearn)

    print("J", J)

    if J>j_threshold:
        print("Recompensa obtida dentro do previsto. OK\n")
    else:
        print("Recompensa obtida abaixo do previsto. FAILED\n")


def test(fmdp_fn, traj_fn, nS=7, nA=2, gamma=0.9):
    fmdp = joblib.load(fmdp_fn) 
    traj = joblib.load(traj_fn) 

    qlearn = RLsol_francisco.myRL(nS, nA, gamma)

    Q = qlearn.traces2Q(traj)

    test_Q(fmdp, Q)

    test_policy(fmdp.runPolicy, RLsol_francisco.Q2pol(Q))
   
    
def main():

    for fns in [('fmdp1.pkl','traj1.pkl'), ('fmdp2.pkl','traj2.pkl')]:
        test(*fns)

if __name__ == '__main__':
    main()
 
