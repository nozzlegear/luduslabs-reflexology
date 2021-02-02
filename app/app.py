import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import os
import sys
import logging

import base64
import io
import hashlib

import re

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json
from time import time

from layout import (generate_layout, BGCOLOR, FONT_COLOR,
                    generate_spec_graph, generate_rating_graph,
                    generate_metric_menu, layout_comp_table)

baseDir = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(baseDir)

from reflexology import rio, analysis

def timeit(func):
    def inner(*args, **kwargs):
        tic = time()
        x = func(*args, **kwargs)
        toc = time()
        print('%s took %.2f s'%(func.__name__, toc-tic))
        return x

    return inner
    
    
def get_hash(df):
    return hashlib.sha256(str(df.values.ravel()).encode()).hexdigest()

logging.basicConfig(filename='reflex-app.log', level=logging.DEBUG)

dataFile = '../data/REFlex.lua'

# size given as (width, height)
SPEC_GRAPH_SIZE = (550, 500)
RATING_GRAPH_SIZE = (550, 450)
FONT_SIZE1 = 14

# For including a partner; need at least this many matches
MATCH_THRESHOLD = 5

CURRENT_SEASON = 30

app = dash.Dash('REFlex')

app.layout = generate_layout()

app.config['suppress_callback_exceptions'] = True

server = app.server

def get_spec(x):
    if 'Beast Mastery' in x:
        return 'Beast Mastery'
    else:
        return x.split(' ')[0]


def get_class(x):
    if 'Beast Mastery' in x:
        return 'Hunter'
    else:
        return ' '.join(x.split(' ')[1:])
    

@timeit
def filter_data(data, partners):
    if len(partners) > 0:
        playerTeams = analysis.get_player_teams(data)

        if len(partners) == 1:
            keepIndex = playerTeams.apply(lambda x: any([name in x for name in partners]))
        else:
            keepIndex = playerTeams.apply(lambda x: all([name in x for name in partners]))

        return data.loc[keepIndex, :]
    else:
        return data

@timeit
def make_spec_plot(data, col, player_name=None, group='spec'):

    winMatrix = timeit(analysis.build_win_matrix)(data, player_name=player_name,
                                          group=group)

    winMatrix['Class'] = winMatrix.index
    winMatrix = winMatrix.sort_values(by=col, ascending=True)

    if group == 'spec':
        cols = [analysis.classColors[get_class(k)]
                for k in winMatrix.index]
    else:
        cols = [analysis.classColors[k] for k in winMatrix.index]

    cols8bit = [tuple([k*255 for k in c]) for c in cols]
    plotlyCols = ['rgb(%i,%i,%i)'%col for col in cols8bit]

    xlabel = {'win rate': 'Win rate (%)',
              'avg rating change': 'Average rating change',
              'N': 'Number of games played',
              'total rating change': 'Total rating change'}[col]


    fig = go.Figure(
        data=[go.Bar(
        x=winMatrix[col],
        y=[k+'  ' for k in winMatrix['Class']],
        orientation='h',
        marker_color=plotlyCols
    )], layout={'margin': {'t': 0, 'r': 0},
                'paper_bgcolor': BGCOLOR,
                'plot_bgcolor': BGCOLOR,
                'font': {'color': FONT_COLOR, 'size': FONT_SIZE1},
                'xaxis_title': xlabel})

    fig.update_yaxes(showgrid=False, dtick=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    
    return fig

@timeit
def make_rating_plot(data, name='', partner1=None, partner2=None):

    SELF_COL = 'rgba(210, 40, 40, 0.9)'    
    partnerCol = 'rgb(210, 210, 210)'

    index = [0] + list(data.index)
    gamesPlayed = pd.Series(np.arange(0, data.shape[0]+1), index=index)

    # MMR trace
    teamMMR = analysis.get_team_mmr(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gamesPlayed,
        y=teamMMR,
        name='Team MMR',
        line={'width': 1, 'color': 'rgba(200,200,200,0.35)'}

    ))

    # Main rating trace
    playerRating = pd.Series(analysis.get_player_rating(data, name),
                             index=index)
    
    fig.add_trace(go.Scatter(
        x=gamesPlayed,
        y=playerRating,
        name=name,
        line={'width': 3}
    ))

    first = True
    if partner1 is not None:
        teamName = '<br>'.join([partner1, partner2]) if partner2 is not None \
                   else partner1

        partnerData = analysis.filter_by_partner(data, partner1, partner2)
        G = partnerData.groupby('session')
    
        for g in G.groups:
            idx = G.groups[g]
            fig.add_trace(go.Scatter(
                x=gamesPlayed[idx],
                y=playerRating[idx],
                mode='lines',
                line={'width': 3, 'color': partnerCol},
                showlegend=first,
                name='Playing with:<br>'+teamName
            ))
            first = False

            
    fig.update_layout(xaxis_title='Games played',
                      yaxis_title='Rating',
                      legend=dict(yanchor='top',
                                  y=1.05,
                                  orientation='h',
                                  xanchor='left',
                                  x=0.1),
                      margin={'t': 0, 'r': 0},
                      paper_bgcolor=BGCOLOR,
                      plot_bgcolor=BGCOLOR,
                      font={'color': FONT_COLOR, 'size': FONT_SIZE1}
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')

    return fig


def make_comp_table(data, playerName):
    opponentSpecs = analysis.get_opponent_specs(data)

    outcomes = analysis.get_outcome(data)
    ratingChange = analysis.get_player_field(data, playerName, 'Rating change')

    sortedOpponentSpecs = '(' + opponentSpecs.apply(lambda x: ', '.join(sorted(x)),
                                                    axis=1) + ')'

    compOutcome = pd.DataFrame([sortedOpponentSpecs,
                                outcomes]).T

    compOutcome.columns = ['Comp', 'Win']
    compOutcome.loc[:, 'Rating change'] = ratingChange
    winCount = compOutcome.groupby('Comp')['Win'].sum()
    fightCount = compOutcome.groupby('Comp')['Win'].count()
    winRate = np.round(winCount/fightCount * 100, 1)
    avgRatingChange = np.round(compOutcome.groupby('Comp')['Rating change'].mean(), 1)
    lossCount = fightCount-winCount

    tic = time()
    comps = pd.Series(fightCount.index).apply(lambda x:x.strip('(').strip(')')).values

    compTable = pd.DataFrame([comps, winCount, lossCount,
                              winRate, avgRatingChange, avgRatingChange*fightCount],
                             index=['Comp', 'Wins', 'Losses', 'Win rate (%)',
                                    'Avg rating change', 'Total rating change']).T

    compTable.loc[:, 'N'] = compTable.Wins + compTable.Losses

    sortedCompTable = compTable.sort_values(by='N', ascending=False)\
                        .drop('N', axis=1)
    toc = time()
    logging.info('C took %.3f s'%(toc-tic))
                        
    return sortedCompTable


def get_player_match_count(data2v2, data3v3):

    matchCountList = []
    for data in [data2v2, data3v3]:
        playerNameFields = [k for k in data.columns if '_Name' in k]
        matchCount = pd.Series(data.loc[:,playerNameFields].values.ravel())\
                     .value_counts()
        matchCountList.append(matchCount)

    combinedMatchCount = pd.DataFrame(matchCountList).fillna(0).sum(axis=0)
    return combinedMatchCount


@app.callback(
    [Output('partner1-selection', 'value'),
     Output('partner2-selection', 'value')],
    [Input('bracket-selection', 'value')]
    )
@timeit
def reset_partner_on_bracket_change(_):
    return None, None


@app.callback(
    [Output('partner1-selection', 'options'),
     Output('partner2-selection', 'options')],
    [Input('player-name', 'children'),
     Input('bracket-selection', 'value'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value')],
    [State('data-store', 'children')]
    )
@timeit
def update_partner_selection(player, bracket, partner1, partner2, json_data):
    tic = time()
    logging.info('UPDATE PARTNER SELECTION')
    if json_data is None:
        raise PreventUpdate
    logging.info('Updating partner selection')
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])
    if bracketData.shape[0] == 0:
        raise PreventUpdate
    allTeamMates = timeit(analysis.get_team_mates)(bracketData).values.ravel()
    teamMates = set(allTeamMates) - set([player])
    teamMateCounts = pd.Series(allTeamMates).value_counts()
    
    filteredData = timeit(analysis.filter_by_partner)(bracketData, partner1, partner2)

    conditionalTeamMates = set(analysis.get_team_mates(filteredData).values.ravel())\
                           - set([player])

    partner1Options = [{'value': mate, 'label': mate}
                       for mate in teamMates if mate != partner2
                       and teamMateCounts[mate] >= MATCH_THRESHOLD]


    partner2Options = [{'value': mate, 'label': mate}
                       for mate in conditionalTeamMates if mate != partner1
                       and teamMateCounts[mate] >= MATCH_THRESHOLD]

    toc = time()
    logging.info('update_partner_selection: %.3f s'%(toc-tic))

    return partner1Options, partner2Options

    
@app.callback(
    [Output('data-store', 'children'),
     Output('player-name', 'children')],
    [Input('upload', 'contents'),
     Input('button-demo', 'n_clicks')]
    )
@timeit
def load_data(content, n_clicks):
    logging.info('Loading data.')
    # So after upload:
    # 1) update the data-store div with hidden JSON data
    # 2) update the partner selection option

    if content is None and n_clicks is None:
        logging.info('Content is None')
        raise PreventUpdate

    if content is not None:
        contentType, contentString = content.split(',')
        
        #if contentType != 'data:text/x-lua;base64':
        #    raise NotImplementedError('Data type is not lua')

        decoded = base64.b64decode(contentString)
        inputData = decoded.decode('utf-8')
        outputFile = hashlib.sha256(inputData.encode()).hexdigest()
        if not os.path.isfile('/data/reflex/%s.lua'%outputFile):        
            with open('/data/reflex/%s.lua'%outputFile, 'w',
                      encoding='utf-8') as f:
                f.write(inputData)
            os.chmod('/data/reflex/%s.lua'%outputFile, 444)
            
    else:
        inputData = dataFile
                
    data2v2, data3v3 = rio.parse_lua_file(inputData)
    data2v2 = data2v2.loc[data2v2.Season==CURRENT_SEASON, :]
    data3v3 = data3v3.loc[data3v3.Season==CURRENT_SEASON, :]


    if data2v2.shape[0] > 0:
        # Filter out data from previous sesasons
        data2v2['session'] = analysis.get_sessions(data2v2)

    if data3v3.shape[0] > 0:
        data3v3['session'] = analysis.get_sessions(data3v3)

    matchCount = get_player_match_count(data2v2, data3v3)
    if np.sum(matchCount == matchCount.max()) == 1:
        playerName = matchCount.idxmax()
    else:
        # Should implement some verification step here if we can't
        # determine the player
        playerName = matchCount[matchCount==matchCount.max()].sample(1).index[0]
                                
    jsonData = '{"2v2":%s, "3v3":%s}'%(data2v2.to_json(),
                                       data3v3.to_json())

    size = sys.getsizeof(jsonData)/1e3
    logging.info('Data is %.2fKB in size.'%size)

    return jsonData, playerName


@app.callback(
    Output('hidden-comp-table', 'children'),
    [Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value'),
     Input('bracket-selection', 'value'),
     Input('player-name', 'children')],
    [State('data-store', 'children')]
    )
@timeit
def update_hidden_comp_table(partner1, partner2, bracket, 
                             player_name, json_data):
    logging.info('UPDATING HIDDEN COMP TABLE')

    if json_data is None:
        raise PreventUpdate
    logging.info('Updating hidden comp table.')
    partners = [a for a in [partner1, partner2] if a is not None]
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])

    if bracketData.shape[0] == 0:
        raise PreventUpdate

    logging.info('Filtering data')
    filteredData = filter_data(bracketData, partners)

    compTable = make_comp_table(filteredData, player_name).to_json()

    return compTable


@app.callback(
    [Output('spec-graph', 'figure'),
     Output('spec-graph', 'style'),
     Output('rating-graph', 'figure'),
     Output('rating-graph', 'style')],
    [Input('metric-selection', 'value'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value'),
     Input('bracket-selection', 'value'),
     Input('radio-class-spec', 'value'),
     Input('player-name', 'children')],
    [State('data-store', 'children')])
@timeit
def update_plots(metric, partner1, partner2, bracket, group_by,
                 player, json_data):

    logging.info('UPDATE PLOTS')
    if json_data is None:
        raise PreventUpdate
    
    logging.info('Updating plots.')
    partners = [a for a in [partner1, partner2] if a is not None]
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])

    if bracketData.shape[0] == 0:
        raise PreventUpdate

    logging.info('Filtering data')
    filteredData = filter_data(bracketData, partners)

    return (make_spec_plot(filteredData, metric, player, group_by),
            {'display': 'block'},
            make_rating_plot(bracketData, player, partner1, partner2),
            {'display': 'block'})


@app.callback(
    Output('spec-selection-1','value'),
    [Input('class-selection-1', 'value')]
    )
@timeit
def reset_spec_1_value(_):
    logging.info('RESET SPEC VALUE 1')
    return None


@app.callback(
    Output('spec-selection-2','value'),
    [Input('class-selection-2', 'value')]
    )
@timeit
def reset_spec_2_value(_):
    logging.info('RESET SPEC VALUE 2')    
    return None


@app.callback(
    Output('comp-table', 'data'),
    [Input('hidden-comp-table', 'children'),
     Input('class-selection-1', 'value'),
     Input('spec-selection-1', 'value'),
     Input('class-selection-2', 'value'),
     Input('spec-selection-2', 'value')
    ])
@timeit
def display_comp_table(comp_data, class1, spec1, class2, spec2):
    logging.info('DISPLAY COMP TABLE')    
    compTable = pd.DataFrame(json.loads(comp_data))

    def _get_spec_markdown(x):
        x = x.strip()
        if x == 'Beast Mastery Hunter':
            return 'Beast Mastery'
        else:
            spec = x.split(' ')[0].lower()
        wowclass = ''.join(x.split(' ')[1:]).lower()
        return '![classicon](/static/icons/%s_%s.png "%s")'%(wowclass, spec, x)

    
    def _get_comp_markdown(comp):
        return ' '.join([_get_spec_markdown(x) for x in comp.split(',')])

    
    def _get_class(x):
        '''
        input will be "Beast Mastery Hunter", "Frost Death Knight", "Arms Warrior"
        '''
        x = x.strip()
        if x == 'Beast Mastery Hunter':
            return 'Hunter'
        else:
            return ' '.join(x.split(' ')[1:])


    def get_filter(filterClass, filterSpec, opponentClassSpec1, opponentClassSpec2):
        filterClassSpec = None
        if filterClass is not None:
            if filterSpec is not None:
                filterClassSpec = filterSpec + ' ' + filterClass

        if filterClass is None:
            return np.ones(len(opponentClassSpec1), dtype=bool)
        
        elif filterClassSpec is None:
            opponentClass1 = opponentClassSpec1.apply(_get_class)
            opponentClass2 = opponentClassSpec2.apply(_get_class)
            return (opponentClass1 == filterClass) | (opponentClass2 == filterClass)

        else:
            return (opponentClassSpec1 == filterClassSpec) |\
                   (opponentClassSpec2 == filterClassSpec)

    opponentClassSpec1 = compTable.Comp.apply(lambda x:x.split(',')[0])
    opponentClassSpec2 = compTable.Comp.apply(lambda x:x.split(', ')[1])

    filter1 = get_filter(class1, spec1, opponentClassSpec1, opponentClassSpec2)
    filter2 = get_filter(class2, spec2, opponentClassSpec1, opponentClassSpec2)
    filterIndex = filter1 & filter2
    
    dispCompTable = compTable.loc[filterIndex, :]

    compsHTML = np.array([_get_comp_markdown(x) for x in dispCompTable.Comp])
    dispCompTable.loc[:, 'Comp'] = compsHTML

    return dispCompTable.to_dict('records')



@app.callback(
    Output('div-dropdown-partner2', 'style'),
    [Input('bracket-selection', 'value')]
    )
@timeit
def hide_show_partner2_selection(bracket):
    logging.info('SHOW/HIDE PARTNER 2 SELECTION')    
    if bracket == '2v2':
        logging.info('Hiding partner2 selection.')
        return {'display': 'none'}

    else:
        logging.info('Showing partner2 selection.')
        return {'display': 'inline-block'}


def get_class_from_markdown_string(comp, get_spec=False):

    specClass = comp.str.findall('"(.*?)"')\
                   .explode()\
                   .str.strip()

    if get_spec:
        return specClass.apply(lambda x: x.split(' ')[0] \
                               if 'Beast Mastery' not in x else 'Beast Mastery')

    
    return specClass.apply(lambda x: ' '.join(x.split(' ')[1:]) if 'Beast Mastery'\
                           not in x else 'Hunter')



@app.callback(
    Output('class-selection-1', 'options'),
    [Input('hidden-comp-table', 'children')]
    )
@timeit
def update_class1_selection(data):
    logging.info('UPDATE CLASS SELECTION 1')    
    df = pd.DataFrame(json.loads(data))
    specClasses = df.Comp.str.split(',').explode().str.strip()
    classes = specClasses.apply(lambda x: ' '.join(x.split(' ')[1:]))
    classSelection = [{'label': c, 'value': c} for c in set(classes)]
    
    return classSelection

@app.callback(
    Output('spec-selection-1', 'options'),
    [Input('class-selection-1', 'value')],
    [State('hidden-comp-table', 'children')]
)
@timeit
def update_spec1_selection(class1, data):
    logging.info('UPDATE SPEC SELECTION 1')    
    if class1 is None:
        return []
    
    df = pd.DataFrame(json.loads(data))
    specClasses = df.Comp.str.split(',').explode().str.strip()
    classes = specClasses.apply(lambda x: ' '.join(x.split(' ')[1:]))
    specs = specClasses.apply(lambda x: x.split(' ')[0])
    currentSpecs = specs[classes == class1]
    specSelection = [{'label': c, 'value': c} for c in set(currentSpecs)]

    return specSelection

@app.callback(
    [Output('class-selection-2', 'options'),
     Output('div-class-selection-2', 'style')],
    [Input('class-selection-1', 'value'),
     Input('spec-selection-1', 'value')],
    [State('hidden-comp-table', 'children'),
     State('class-selection-2', 'value')])
@timeit
def update_class2_selection(class1, spec1, data, class2):
    logging.info('UPDATE CLASS SELECTION 2')
    if data is None:
        return [], {}

    df = pd.DataFrame(json.loads(data))
    specClasses = df.Comp.str.split(',').explode().str.strip()
    classes = specClasses.apply(lambda x: ' '.join(x.split(' ')[1:]))
    specs = specClasses.apply(lambda x: x.split(' ')[0])

    if spec1 is None:
        idx = (classes == class1)
    else:
        idx = (classes == class1) & (specs == spec1)

    targetIndex = pd.unique(classes.index[idx])
    boolTargetIndex = np.in1d(classes.index, targetIndex)
    partnerIndices = ~idx & boolTargetIndex

    currentClasses = classes[partnerIndices]

    return ([{'label': c, 'value': c} for c in set(currentClasses)],
            {'display': 'inline-block'})

@app.callback(
    Output('spec-selection-2', 'options'),
    [Input('class-selection-2', 'value')],
    [State('class-selection-1', 'value'),
     State('spec-selection-1', 'value'),
     State('hidden-comp-table', 'children')])
@timeit
def update_spec2_selection(class2, class1, spec1, data):
    logging.info('UPDATE SPEC SELECTION 2')
    if class2 is None or data is None:
        return []

    df = pd.DataFrame(json.loads(data))

    specClasses = df.Comp.str.split(',').explode().str.strip()
    classes = specClasses.apply(lambda x: ' '.join(x.split(' ')[1:]))
    specs = specClasses.apply(lambda x: x.split(' ')[0])

    if spec1 is None:
        idx = (classes == class1)
    else:
        idx = (classes == class1) & (specs == spec1)

    targetIndex = pd.unique(classes.index[idx])
    boolTargetIndex = np.in1d(classes.index, targetIndex)
    partnerIndices = ~idx & boolTargetIndex

    currentSpecs = specs[partnerIndices & (classes == class2)]

    return [{'label': c, 'value': c} for c in set(currentSpecs)]

@app.callback(
    [Output('metric-rating-change', 'children'),
     Output('metric-games-played', 'children'),
     Output('metric-sessions-played', 'children'),
     Output('metric-rating-change', 'style')],
    [Input('bracket-selection', 'value'),
     Input('player-name', 'children'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value')],
    [State('data-store', 'children')]
    )
@timeit
def update_kpis(bracket, player, partner1, partner2, json_data):
    logging.info('UPDATE KPIs')
    if json_data is None:
        raise PreventUpdate
    green = '#090'
    red = '#900'
    allData = json.loads(json_data)
    partners = [p for p in [partner1, partner2] if p is not None]
    bracketData = pd.DataFrame(allData[bracket])
    filteredData = filter_data(bracketData, partners)
    if filteredData.shape[0] == 0:
        raise PreventUpdate

    ratingChange = analysis.get_player_field(filteredData, player, 'Rating change').sum()


    symbol = '+' if ratingChange >= 0 else ''

    ratingChangeString = '%s%i'%(symbol, ratingChange)
    gamesPlayed = '%i'%filteredData.shape[0]
    sessionsPlayed = '%i'%len(pd.unique(filteredData['session']))

    ratingChangeStyle = {'color' : green if ratingChange >= 0 else red}

    return ratingChangeString, gamesPlayed, sessionsPlayed, ratingChangeStyle


@app.callback(
    Output('div-tab-content', 'children'),
    [Input('main-tab', 'value')])
@timeit
def display_tab(tab):
    logging.info('DISPLAY TABS')    
    if tab == 'graph':
        return [generate_metric_menu(),
                html.Div(id='div-graphs',children=[
                    generate_spec_graph(),
                    generate_rating_graph()])
                ]
    
    elif tab == 'comp':
        return [layout_comp_table()]


@app.callback(
    [Output('div-main-content', 'style'),
     Output('div-main-greet', 'style')],
    [Input('button-demo', 'n_clicks'),
     Input('upload', 'filename')]
    )
@timeit
def show_main_content(n_clicks, filename):
    logging.info('SHOW MAIN CONTENT')
    if (n_clicks is not None) or (filename is not None):
        return [{'display': 'block'}, {'display': 'none'}]

    return [{'display': 'none'}, {'display': 'block'}]

    
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)
