import numpy as np
import pandas as pd
from scipy.stats import beta


def get_opponent_specs(data):
    t0 = ['T0P0_Class', 'T0P0_Spec', 'T0P1_Class', 'T0P1_Spec']
    if 'T0P2_Class' in data.columns:
        t0 += ['T0P2_Class', 'T0P2_Spec']
        
    t1 = [t.replace('T0', 'T1') for t in t0]
        
    opponentSpecs = pd.DataFrame(index=data.index, columns=['Player0', 'Player1'])
    opponentSide = 1 - data.loc[:, 'PlayerSide']
    for p, t in enumerate([t0,t1]):
        opponent = data.loc[opponentSide == p, t]
        opponentSpecs.loc[opponent.index, 'Player0'] = opponent.loc[:, t[1]] + ' '\
                                                     + opponent.loc[:, t[0]]
        opponentSpecs.loc[opponent.index, 'Player1'] = opponent.loc[:, t[3]] + ' '\
                                                     + opponent.loc[:, t[2]]
    return opponentSpecs


def get_outcome(data):
    return np.double(data.loc[:, 'PlayerSide'] == data.loc[:, 'Winner'])


def build_win_matrix(data, alpha=0.1):

    opponentSpecs = get_opponent_specs(data)
    opponentSpecs['ID'] = opponentSpecs.index
    opponentSpecs.loc[:, 'Win'] = get_outcome(data)
    allSpecs = pd.Series(index=list(set(opponentSpecs.Player0)
                                    & set(opponentSpecs.Player1)),
                         dtype=np.float)
    
    outcomes = pd.concat([pd.DataFrame(allSpecs)\
                          .join(opponentSpecs.set_index(player, drop=False))
                          .drop(0, axis=1) for player in ['Player0',
                                                          'Player1']],
                         axis=0)
    
    outcomes.loc[:, 'CLASS/SPEC'] = outcomes.index
    
    outcomes = outcomes.drop_duplicates()

    outcomes.index.name = 'Spec'
    G = outcomes.loc[:, 'Win'].groupby('Spec')

    winCount = G.sum()
    fightCount = G.count()

    lossCount = fightCount - winCount

    winRate = G.mean()
    d = beta(winCount.values+1, lossCount.values+1)
    winLower = pd.Series(d.ppf(alpha/2), index=winCount.index)
    winUpper = pd.Series(d.ppf(1-alpha/2), index=winCount.index)

    return pd.DataFrame([winRate, winLower, winUpper, fightCount],
                        index=['win rate', 'lower', 'upper', 'N']).T
