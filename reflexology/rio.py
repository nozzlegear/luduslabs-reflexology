import luadata
import pandas as pd

translation = {'Protección': 'Protection',
               'Guardián': 'Guardian',
               'Caos': 'Havoc',
               'Treffsicherheit': 'Marksmanship',
               'Gardien': 'Guardian',
               'Sangre': 'Blood',
               'Elementar': 'Elemental',
               'Feuer': 'Fire',
               'Disziplin': 'Discipline',
               'Wiederherstellung': 'Restoration',
               'Waffen': 'Arms',
               'Rachsucht': 'Vengeance',
               'Schatten': 'Shadow',
               'Gleichgewicht': 'Balance',
               'Täuschung': 'Subtlety',
               'Braumeister': 'Brewmaster',
               'Verwüstung': 'Havoc',
               'Wildheit': 'Feral',
               'Nebelwirker': 'Mistweaver',
               'Unheilig': 'Unholy',
               'Heilig': 'Holy',
               'Vergeltung': 'Retribution',
               'Windläufer': 'Windwalker',
               'Gebrechen': 'Affliction',
               'Furor': 'Fury',
               'Schutz': 'Protection',
               'Blut': 'Blood',
               'Zerstörung': 'Destruction',
               'Meucheln': 'Assassination',
               'Tierherrschaft': 'Beast Mastery',
               'Überleben': 'Survival',
               'Arkan': 'Arcane',
               'Wächter': 'Guardian',
               'Maître brasseur': 'Brewmaster',
               'Forajido': 'Outlaw',
               'Forajida': 'Outlaw',
               'Verstärkung': 'Enhancement',
               'Gesetzlosigkeit': 'Outlaw',
               'Arcanes': 'Arcane',
               'Dämonologie': 'Demonology',
               'Guardiana': 'Guardian',
               'Guardiano': 'Guardian',
               'Venganza': 'Vengeance',
               'Bestias': 'Beast Mastery',
               'Maestro cervecero': 'Brewmaster',
               'Maestra cervecera': 'Brewmaster',
               'Sagrada': 'Holy',
               'Sagrado': 'Holy',
               'Reprensión': 'Retribution',
               'Disciplina': 'Discipline',
               'Sutileza': 'Subtlety',
               'Assassinat': 'Assassination',
               'Finesse': 'Subtlety',
               'Sacré': 'Holy',
               'Restauration': 'Restoration',
               'Feu': 'Fire',
               'Restauración': 'Restoration',
               'Armas': 'Arms',
               'Vindicte': 'Retribution',
               'Farouche': 'Feral',
               'Survie': 'Survival',
               'Givre': 'Frost',
               'Ombre': 'Shadow',
               'Viajero del viento': 'Windwalker',
               'Viajera del viento': 'Windwalker',
               'Escarcha': 'Frost',
               'Equilibrio': 'Balance',
               'Supervivencia': 'Survival',
               'Aflicción': 'Affliction',
               'Profano': 'Unholy',
               'Profana': 'Unholy',
               'Tejedor de niebla': 'Mistweaver',
               'Tejedora de niebla': 'Mistweaver',
               'Puntería': 'Marksmanship',
               'Punterío': 'Marksmanship',
               'Fuega': 'Fire',
               'Fuego': 'Fire',
               'Arcano': 'Arcane',
               'Arcana': 'Arcane',
               'Destrucción': 'Destruction',
               'Marche-vent': 'Windwalker',
               'Devastación': 'Havoc',
               'Tisse-brume': 'Mistweaver',
               'Armes': 'Arms',
               'Amélioration': 'Enhancement',
               'Équilibre': 'Balance',
               'Fureur': 'Fury',
               'Mejora': 'Enhancement',
               'Sombra': 'Shadow',
               'Précision': 'Marksmanship',
               'Élémentaire': 'Elemental',
               'Maîtrise des bêtes': 'Beast Mastery',
               'Dévastation': 'Havoc',
               'Démonologie': 'Demonology',
               'Impie': 'Unholy',
               'Hors-la-loi': 'Outlaw',
               "Стихии": "Elemental",
               "Совершенствование": "Enhancement",
               "Исцеление": "Restoration",
               "Оружие": "Arms",
               "Неистовство": "Fury",
               "Защита": "Protection",
               "Колдовство": "Affliction",
               "Демонология": "Demonology",
               "Разрушение": "Destruction",
               "Послушание": "Discipline",
               "Свет": "Holy",
               "Тьма": "Shadow",
               "Повелитель зверей": "Beast Mastery",
               "Стрельба": "Marksmanship",
               "Стрел": "Marksmanship",
               "Выживание": "Survival",
               "Ликвидация": "Assassination",
               "Головорез": "Outlaw",
               "Скрытность": "Subtlety",
               "Тайная магия": "Arcane",
               "Огонь": "Fire",
               "Лед": "Frost",
               "Хмелевар": "Brewmaster",
               "Ткач туманов": "Mistweaver",
               "Танцующий с ветром": "Windwalker",
               "Воздаяние": "Retribution",
               "Кровь": "Blood",
               "Нечестивость": "Unholy",
               "Баланс": "Balance",
               "Сила зверя": "Feral",
               "Страж": "Guardian",
               "Истребление": "Havoc",
               "Месть": "Vengeance",
               "Повелительница зверей": "Beast Mastery",
               "Танцующая с ветром": "Windwalker"
}

cols = ['Name', '', '', '', '', 'Team', 'Race', '', 'Class', 'Damage',
        'Healing', 'Rating', 'Rating change', '', '', 'Spec', '']

def translate(data):
    specCols = [k for k in data.columns if '_Spec' in k]
    for col in specCols:
        data.loc[:, col] = data.loc[:, col].replace(translation)

    return data

def fix_class(data):
    classCols = [k for k in data.columns if '_Class' in k]
    for col in classCols:
        data.loc[:, col] = data.loc[:, col].replace({'Demonhunter': 'Demon Hunter',
                                                    'Deathknight': 'Death Knight'})

    return data

def parse_player(player):
    return [(cols[i], player[k]) for i, k in enumerate(range(0, len(player), 3))]


def get_arena(data, bracket, rated=True):
    filteredData = [x for x in data if type(x) == dict]
    playersNum = {'2v2': 4, '3v3': 6}[bracket.lower()]
    return [x for x in filteredData if x['PlayersNum'] == playersNum
            and x['isArena'] and x['isRated'] == rated]


def get_rbg(data, rated=True):
    filteredData = [x for x in data if type(x) == dict]
    playersNum = 20
    return [x for x in filteredData if x['PlayersNum'] == playersNum
            and not x['isArena'] and x['isRated'] == rated]


def parse_match_data(match):
    players = [dict(parse_player(k)) for k in match['Players'] if len(k) > 10]

    team1 = [k for k in players if k['Team'] == 0]
    team2 = [k for k in players if k['Team'] == 1]

    def parse_team(team):
        return {'T%iP%i_%s'%(player['Team'], k, col) : player[col] \
                for k, player in enumerate(team)\
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

from time import time
def timeit(func):
    def inner(*args, **kwargs):
        tic = time()
        x = func(*args, **kwargs)
        toc = time()
        print('%s took %.2f s'%(func.__name__, toc-tic))
        return x

    return inner

def capitalise_class(df):
    classCols = [k for k in df.columns if '_Class' in k]
    for c in classCols:
        df.loc[:, c] = df[c].str.capitalize()

    return df

def parse_lua_file(file_name):
    if len(file_name) < 100:
        data = luadata.read(file_name, encoding='utf-8')
    else:
        data = luadata.unserialize(file_name)

    raw2v2 = get_arena(data['REFlexDatabase'], '2v2')
    raw3v3 = get_arena(data['REFlexDatabase'], '3v3')

    match2v2 = fix_class(translate(capitalise_class(pd.DataFrame([parse_match_data(k) for k in raw2v2]))))
    match3v3 = fix_class(translate(capitalise_class(pd.DataFrame([parse_match_data(k) for k in raw3v3]))))

    return match2v2, match3v3

def parse_lua_file_rbg(file_name):
    if len(file_name) < 100:
        data = luadata.read(file_name, encoding='utf-8')
    else:
        data = luadata.unserialize(file_name)

    rawRbg = get_rbg(data['REFlexDatabase'])

    matchRbg = fix_class(translate(capitalise_class(pd.DataFrame([parse_match_data(k)
                                                                  for k in rawRbg]))))

    return matchRbg
