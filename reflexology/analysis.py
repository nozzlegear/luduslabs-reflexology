import numpy as np
import pandas as pd
from scipy.stats import beta

classColors = {
    'Death Knight': (0.77,0.12,0.23),
    'Demon Hunter': (0.64,0.19,0.79),
    'Druid': (1, 0.49, 0.04),
    'Hunter': (0.67, 0.83,0.45),
    'Mage': (0.25,0.78,0.92),
    'Monk': (0,1,0.59),
    'Paladin': (0.96,0.55,0.73),
    'Priest': (1,1,1),
    'Rogue': (1,0.96,0.41),
    'Shaman': (0,0.44,0.87),
    'Warlock': (0.53,0.53,0.93),
    'Warrior': (0.78,0.61,0.43)
}


def build_win_matrix(data, alpha=0.1):

    opponentSpecs = get_opponent_specs(data)
    opponentSpecs['ID'] = opponentSpecs.index
    opponentSpecs.loc[:, 'Win'] = get_outcome(data)
    allSpecs = pd.Series(index=list(set(opponentSpecs.Player0)\
                                    .union(set(opponentSpecs.Player1))),
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


def get_player_field(data, name, field):
    nameCols = pd.Series([k for k in data.columns if '_Name' in k])
    def _get_player_field(x, name):
        col = [c for c in nameCols if x.loc[c] == name][0]
        ratingCol = col.replace('_Name','_Rating')
        return x.loc[col.replace('_Name', '_'+field)]
    
    return data.apply(lambda x:_get_player_field(x, name), axis=1)


def get_player_rating(data, name):
    nameCols = pd.Series([k for k in data.columns if '_Name' in k])
    def _get_player_rating(x, name):
        col = [c for c in nameCols if x.loc[c] == name][0]
        ratingCol = col.replace('_Name','_Rating')
        return x.loc[ratingCol] + x.loc[ratingCol+' change']
    
    return data.apply(lambda x:_get_player_rating(x, name), axis=1)


def get_player_and_team_mates(data):
    allTeams = get_player_teams(data)
    players = [a for k in allTeams for a in k.split(', ')]

    # whoever features the most often is the player
    # note that if a player only plays with the same partners
    # then this will not be reliable
    gameCount = pd.Series(players).value_counts()
    maxPlayers = gameCount.index[gameCount == gameCount.max()]
    player = [k for k in maxPlayers if '-' not in k][0]
    
    teamMates = [k for k in set(players) if k != player]
    return player, teamMates


def get_player_teams(data):
    K = int(len([k for k in data.columns if '_Name' in k])/2)

    def _get_player_teams(x):
        team = x.loc['PlayerSide']
        nameCols = ['T%iP%i_Name'%(team, i) for i in range(K)]
        return ', '.join(sorted(x.loc[nameCols].values))
    
    return data.apply(_get_player_teams, axis=1)


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
    return pd.Series(np.double(data.loc[:, 'PlayerSide'] == data.loc[:, 'Winner']),
                     index=data.index)
