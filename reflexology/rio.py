import luadata
import pandas as pd

cols = ['Name', '', '', '', '', 'Team', 'Race', 'Class', '', 'Damage',
        'Healing', 'Rating', 'Rating change', '', '', 'Spec', '']


def parse_player(player):
    return [(cols[i], player[k]) for i, k in enumerate(range(0, len(player), 3))]


def get_arena(data, bracket, rated=True):
    filteredData = [x for x in data if type(x) == dict]
    playersNum = {'2v2': 4, '3v3': 6}[bracket.lower()]
    return [x for x in filteredData if x['PlayersNum'] == playersNum
            and x['isArena'] and x['isRated'] == rated]


def parse_match_data(match):
    players = [dict(parse_player(k)) for k in match['Players'] if len(k) > 10]

    team1 = [k for k in players if k['Team'] == 0]
    team2 = [k for k in players if k['Team'] == 1]

    def parse_team(team):
        return {'T%iP%i_%s'%(player['Team'], k, col) : player[col] \
                for k,player in enumerate(team)\
                for col in player}

    matchData = {k: match[k] for k in ['Map', 'Season', 'Duration',
                                       'Version', 'Time', 'PlayerSide',
                                       'Winner']}

    mmrData = {'T0_MMR': match['TeamData'][0][9],
               'T1_MMR': match['TeamData'][3][9]}

    data = dict(**matchData, **mmrData,
                **parse_team(team1),
                **parse_team(team2))

    return data


def parse_lua_file(file_name):
    data = luadata.read(file_name, encoding='utf-8')

    raw2v2 = get_arena(data['REFlexDatabase'], '2v2')
    raw3v3 = get_arena(data['REFlexDatabase'], '3v3')

    match2v2 = pd.DataFrame([parse_match_data(k) for k in raw2v2])
    match3v3 = pd.DataFrame([parse_match_data(k) for k in raw3v3])

    return match2v2, match3v3
