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


def build_win_matrix(data, alpha=0.1, player_name=None, group='specs'):

    opponentSpecs = get_opponent_specs(data)

    def _get_class(x):
        return ' '.join(x.split(' ')[1:])
    if group == 'class':
        opponentSpecs = opponentSpecs.apply(lambda x: x.apply(_get_class))
        
    opponentSpecs.columns = ['Player%i'%k for k in opponentSpecs.columns]
    allSpecs = pd.Series(index=list(set(opponentSpecs.values.ravel())))
    
    opponentSpecs['ID'] = opponentSpecs.index
    opponentSpecs.loc[:, 'Win'] = get_outcome(data)
    if player_name is None:
        opponentSpecs.loc[:, 'rating change'] = np.zeros(data.shape[0]) + np.nan
    else:
        opponentSpecs.loc[:, 'rating change'] = get_player_field(data, player_name,
                                                                 'Rating change')

    outcomes = pd.concat([pd.DataFrame(allSpecs)\
                          .join(opponentSpecs.set_index(player, drop=False))
                          .drop(0, axis=1) for player in opponentSpecs.columns],
                         axis=0)
    
    outcomes.loc[:, 'CLASS/SPEC'] = outcomes.index
    
    outcomes = outcomes.drop_duplicates()

    outcomes.index.name = 'Spec'
    

    G = outcomes.groupby('Spec')

    winCount = G['Win'].sum()
    fightCount = G['Win'].count()

    lossCount = fightCount - winCount

    winRate = G['Win'].mean()
    d = beta(winCount.values+1, lossCount.values+1)
    winLower = pd.Series(d.ppf(alpha/2), index=winCount.index)
    winUpper = pd.Series(d.ppf(1-alpha/2), index=winCount.index)

    avgRatingChange = G['rating change'].mean()

    return pd.DataFrame([winRate, winLower, winUpper, fightCount,
                         avgRatingChange],
                        index=['win rate', 'lower', 'upper', 'N',
                               'avg rating change']).T


def get_player_field(data, name, field):
    nameCols = pd.Series([k for k in data.columns if '_Name' in k])
    def _get_player_field(x, name):
        col = [c for c in nameCols if x.loc[c] == name][0]
        return x.loc[col.replace('_Name', '_'+field)]
    
    return data.apply(lambda x:_get_player_field(x, name), axis=1)


def get_player_rating(data, name):
    nameCols = pd.Series([k for k in data.columns if '_Name' in k])
    def _get_player_rating(x, name):
        col = [c for c in nameCols if x.loc[c] == name][0]
        ratingCol = col.replace('_Name','_Rating')
        return x.loc[ratingCol] + x.loc[ratingCol+' change']


    # This first bit gets the end-of-fight rating
    postfightRating = data.apply(lambda x:_get_player_rating(x, name), axis=1).values

    # we prepend the first pre-fight rating (which corresponds to 0 games played)
    prefightRating = np.array([get_player_field(data[:2], name,
                                                'Rating').values[0]])

    playerRating = np.concatenate([prefightRating, postfightRating])
    return playerRating


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
    is3v3 = np.any(pd.Series(data.columns).str.match('T0P2'))

    nPlayers = 2 + is3v3
    t0 = ['T0P0_Class', 'T0P0_Spec', 'T0P1_Class', 'T0P1_Spec']

    if is3v3:
        t0 += ['T0P2_Class', 'T0P2_Spec']

    t1 = [t.replace('T0', 'T1') for t in t0]

    opponentSpecs = pd.DataFrame(index=data.index, columns=np.arange(nPlayers))
    opponentSide = 1 - data.loc[:, 'PlayerSide']
    for p, t in enumerate([t0,t1]):
        opponent = data.loc[opponentSide == p, t]
        for k in range(nPlayers):
            opponentSpecs.loc[opponent.index, k] = opponent.loc[:, t[k*2+1]] +' '\
                                                   + opponent.loc[:, t[k*2]]
    return opponentSpecs


def get_outcome(data):
    return pd.Series(np.double(data.loc[:, 'PlayerSide'] == data.loc[:, 'Winner']),
                     index=data.index)


def get_team_mmr(data, opponent=False):
    def _get_team_mmr(x):
        if opponent:
            mmrCol = 'T%i_MMR'%(1-x.PlayerSide)
        else:
            mmrCol = 'T%i_MMR'%x.PlayerSide

        return x.loc[mmrCol]

    return data.apply(_get_team_mmr, axis=1)
