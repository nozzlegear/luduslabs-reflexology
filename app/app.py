import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import os
import sys
import logging

import base64
import io

import re

from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json

from layout import (generate_layout, BGCOLOR, FONT_COLOR,
                    generate_spec_graph, generate_rating_graph,
                    generate_metric_menu, layout_comp_table)

baseDir = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(baseDir)

from reflexology import rio, analysis


logging.basicConfig(filename='reflex-app.log', level=logging.DEBUG)

dataFile = '../data/REFlex.lua'

# size given as (width, height)
SPEC_GRAPH_SIZE = (550, 500)
RATING_GRAPH_SIZE = (550, 450)

app = dash.Dash('REFlex')

app.layout = generate_layout()

app.config['suppress_callback_exceptions'] = True


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


def make_spec_plot(data, col, player_name=None, group='spec'):

    winMatrix = analysis.build_win_matrix(data, player_name=player_name,
                                          group=group)
    winMatrix['Class'] = winMatrix.index
    winMatrix = winMatrix.sort_values(by=col, ascending=True)

    if group == 'spec':
        cols = [analysis.classColors[' '.join(k.split(' ')[1:])]
                for k in winMatrix.index]
    else:
        cols = [analysis.classColors[k] for k in winMatrix.index]

    cols8bit = [tuple([k*255 for k in c]) for c in cols]
    plotlyCols = ['rgb(%i,%i,%i)'%col for col in cols8bit]

    xlabel = {'win rate': 'Win rate (%)',
              'avg rating change': 'Average rating change',
              'N': 'Number of games played'}[col]


    fig = go.Figure(
        data=[go.Bar(
        x=winMatrix[col],
        y=[k+'  ' for k in winMatrix['Class']],
        orientation='h',
        marker_color=plotlyCols
    )], layout={'margin': {'t': 0, 'r': 0},
                'paper_bgcolor': BGCOLOR,
                'plot_bgcolor': BGCOLOR,
                'font': {'color': FONT_COLOR, 'size': 14},
                'xaxis_title': xlabel})

    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    
    return fig


def make_rating_plot(data, names):
    teamMMR = analysis.get_team_mmr(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(0, len(teamMMR)+1)),
        y=teamMMR,
        name='Team MMR',
        line={'width': 2, 'color': 'rgba(200,200,200,0.5)'}
    ))

    for name in names:
        playerRating = analysis.get_player_rating(data, name)

        fig.add_trace(go.Scatter(
            x=list(range(0, len(playerRating)+1)),
            y=playerRating,
            name=name,
            line={'width': 3}
        ))

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
                      font={'color': FONT_COLOR}
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')

    return fig


def make_comp_table(data):
    opponentSpecs = analysis.get_opponent_specs(data)

    outcomes = analysis.get_outcome(data)
    ratingChange = analysis.get_player_field(data, 'Geebs', 'Rating change')

    compOutcome = pd.DataFrame([opponentSpecs
                                .apply(lambda x:'('+', '.join(sorted(x))+')',
                                       axis=1),
                                outcomes]).T

    compOutcome.columns = ['Comp', 'Win']
    compOutcome.loc[:, 'Rating change'] = ratingChange
    winCount = compOutcome.groupby('Comp')['Win'].sum()
    fightCount = compOutcome.groupby('Comp')['Win'].count()
    winRate = np.round(winCount/fightCount * 100, 1)
    avgRatingChange = np.round(compOutcome.groupby('Comp')['Rating change'].mean(), 1)
    lossCount = fightCount-winCount

    comps = pd.Series(fightCount.index).apply(lambda x:x.strip('(').strip(')')).values

    compTable = pd.DataFrame([comps, winCount, lossCount,
                              winRate, avgRatingChange],
                             index=['Comp', 'Wins', 'Losses', 'Win rate (%)',
                                    'Avg rating change']).T

    compTable.loc[:, 'N'] = compTable.Wins + compTable.Losses

    sortedCompTable = compTable.sort_values(by='N', ascending=False)\
                        .drop('N', axis=1)
                        
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
def reset_partner_on_bracket_change(_):
    return None, None


@app.callback(
    [Output('partner1-selection', 'options'),
     Output('partner2-selection', 'options')],
    [Input('data-store', 'children'),
     Input('bracket-selection', 'value'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value')]
    )
def update_partner_selection(json_data, bracket, partner1, partner2):
    logging.info('Updating partner selection')
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])
    player, teamMates = analysis.get_player_and_team_mates(bracketData)
    partner1Options = [{'value': mate, 'label': mate}
                       for mate in teamMates if mate != partner2]

    partner2Options = [{'value': mate, 'label': mate}
                       for mate in teamMates if mate != partner1]

    return partner1Options, partner2Options


@app.callback(
    [Output('data-store', 'children'),
     Output('player-name', 'children')],
    [Input('upload', 'contents')]
    )
def load_data(content):
    logging.info('Loading data.')
    # So after upload:
    # 1) update the data-store div with hidden JSON data
    # 2) update the partner selection option

    if content is not None:
        contentType, contentString = content.split(',')
        
        if contentType != 'data:text/x-lua;base64':
            raise NotImplementedError('Data type is not lua')

        decoded = base64.b64decode(contentString)
        inputData = io.StringIO(decoded.decode('utf-8'))
    else:
        inputData = dataFile
                
    data2v2, data3v3 = rio.parse_lua_file(dataFile)

    matchCount = get_player_match_count(data2v2, data3v3)
    if np.sum(matchCount == matchCount.max()) == 1:
        playerName = matchCount.idxmax()
    else:
        # Should implement some verification step here if we can't
        # determine the player
        playerName = 'Unknown'
                                
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
     Input('data-store', 'children')],
    [State('player-name', 'children')]
    )
def update_hidden_comp_table(partner1, partner2, bracket, json_data,
                             player_name):
    logging.info('Updating hidden comp table.')
    partners = [a for a in [partner1, partner2] if a is not None]
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])

    logging.info('Getting players')
    player, _ = analysis.get_player_and_team_mates(bracketData)
    logging.info('Filtering data')
    filteredData = filter_data(bracketData, partners)

    return make_comp_table(filteredData).to_json()

@app.callback(
    [Output('spec-graph', 'figure'),
     Output('rating-graph', 'figure')],
    [Input('metric-selection', 'value'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value'),
     Input('bracket-selection', 'value'),
     Input('radio-class-spec', 'value'),
     Input('data-store', 'children')],
    [State('player-name', 'children')])
def update_plots(metric, partner1, partner2, bracket, group_by,
                 json_data, player_name):

    logging.info('Updating plots.')
    partners = [a for a in [partner1, partner2] if a is not None]
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])

    logging.info('Getting players')
    player, _ = analysis.get_player_and_team_mates(bracketData)
    logging.info('Filtering data')
    filteredData = filter_data(bracketData, partners)

    return (make_spec_plot(filteredData, metric, player_name, group_by),
            make_rating_plot(filteredData, [player]+partners))


@app.callback(
    Output('spec-selection-1','value'),
    [Input('class-selection-1', 'value')]
    )
def reset_spec_1_value(_):
    return None


@app.callback(
    Output('spec-selection-2','value'),
    [Input('class-selection-2', 'value')]
    )
def reset_spec_2_value(_):
    return None


@app.callback(
    Output('comp-table', 'data'),
    [Input('hidden-comp-table', 'children'),
     Input('class-selection-1', 'value'),
     Input('spec-selection-1', 'value'),
     Input('class-selection-2', 'value'),
     Input('spec-selection-2', 'value')
    ])
def display_comp_table(comp_data, class1, spec1, class2, spec2):
    compTable = pd.DataFrame(json.loads(comp_data))

    def _get_spec_markdown(x):
        x = x.strip()
        spec = x.split(' ')[0].lower()
        wowclass = ''.join(x.split(' ')[1:]).lower()
        return '![classicon](/static/icons/%s_%s.png "%s")'%(wowclass, spec, x)

    def _get_comp_markdown(comp):
        return ' '.join([_get_spec_markdown(x) for x in comp.split(',')])

    if class1 is not None:
        filter1 = spec1 + ' ' + class1 if spec1 is not None else class1
        print(filter1)
    else:
        filter1 = ''

    if class2 is not None:
        filter2 = spec2 + ' ' + class2 if spec2 is not None else class2
    else:
        filter2 = ''

    compProcessed = compTable['Comp'].str.replace('Demon ','Demon')
    filterIndex = compProcessed.str.contains(filter1) &\
                  compProcessed.str.contains(filter2)

    dispCompTable = compTable.loc[filterIndex, :]

    compsHTML = np.array([_get_comp_markdown(x) for x in dispCompTable.Comp])
    dispCompTable.loc[:, 'Comp'] = compsHTML

    return dispCompTable.to_dict('records')


@app.callback(
    Output('sum-comp-table', 'data'),
    [Input('comp-table', 'data'),
     Input('comp-table', 'derived_virtual_data')]
    )
def update_sum_comp_table(compData, derivedData):


    '''
    data = pd.DataFrame(compData if derivedData is None else derivedData)

    wins, losses = data['Wins'].sum(), data['Losses'].sum()
    N = wins + losses
    ratingChange = data.loc[:, 'Avg rating change'] * \
                   (data.loc[:, 'Wins'] + data.loc[:, 'Losses'])

    winRate = np.round(wins/(wins + losses) * 100, 1)
    avgRatingChange = np.round(np.sum(ratingChange)/N, 2)

    return [{'Comp': 'Total',
             'Wins': wins,
             'Losses': losses,
             'Win rate (%)': winRate,
             'Avg rating change': avgRatingChange}]

    '''
    return []


@app.callback(
    Output('div-dropdown-partner2', 'style'),
    [Input('bracket-selection', 'value')]
    )
def hide_show_partner2_selection(bracket):
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
        return specClass.apply(lambda x: x.split(' ')[0])

    return specClass.apply(lambda x: ' '.join(x.split(' ')[1:]))



@app.callback(
    Output('class-selection-1', 'options'),
    [Input('hidden-comp-table', 'children')]
    )
def update_class1_selection(data):
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
def update_spec1_selection(class1, data):

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
def update_class2_selection(class1, spec1, data, class2):
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
def update_spec2_selection(class2, class1, spec1, data):
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
     Output('metric-rating-change', 'style')],
    [Input('bracket-selection', 'value'),
     Input('data-store', 'children'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value')]
    )
def update_kpis(bracket, json_data, partner1, partner2):

    green = '#090'
    red = '#900'
    allData = json.loads(json_data)
    partners = [p for p in [partner1, partner2] if p is not None]
    bracketData = pd.DataFrame(allData[bracket])
    filteredData = filter_data(bracketData, partners)
    player, _ = analysis.get_player_and_team_mates(filteredData)
    
    print(player)
    print(analysis.get_player_field(filteredData, player, 'Rating change'))
    ratingChange = analysis.get_player_field(filteredData, player, 'Rating change').sum()
    print(ratingChange)

    symbol = '+' if ratingChange >+ 0 else ''

    ratingChangeString = '%s%i'%(symbol, ratingChange)
    gamesPlayed = '%i'%filteredData.shape[0]

    ratingChangeStyle = {'color' : green if ratingChange >= 0 else red}

    return ratingChangeString, gamesPlayed, ratingChangeStyle


@app.callback(
    Output('div-tab-content', 'children'),
    [Input('main-tab', 'value')]
    )
def display_tab(tab):
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
def show_main_content(n_clicks, filename):
    if (n_clicks is not None) or (filename is not None):
        return [{'display': 'block'}, {'display': 'none'}]

    return [{'display': 'none'}, {'display': 'block'}]

    
if __name__ == "__main__":
    #app.run_server(host='0.0.0.0', port=5050)
    app.run_server(host='0.0.0.0', port=8050, debug=True)
