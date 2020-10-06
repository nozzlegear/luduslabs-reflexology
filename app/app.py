import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import os
import sys
import logging

from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json

from layout import generate_layout, BGCOLOR, FONT_COLOR

baseDir = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(baseDir)

from reflexology import rio, analysis


logging.basicConfig(filename='reflex-app.log', level=logging.DEBUG)

dataFile = '../data/REFlex.lua'

# size given as (width, height)
SPEC_GRAPH_SIZE = (500, 450)
RATING_GRAPH_SIZE = (500, 400)

app = dash.Dash('REFlex')

app.layout = generate_layout()


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


def make_spec_plot(data, col):
    winMatrix = analysis.build_win_matrix(data)
    winMatrix['Class'] = winMatrix.index
    winMatrix = winMatrix.sort_values(by=col, ascending=False)

    cols = [analysis.classColors[' '.join(k.split(' ')[1:])]
            for k in winMatrix.index]
    cols8bit = [tuple([k*255 for k in c]) for c in cols]
    plotlyCols = ['rgb(%i,%i,%i)'%col for col in cols8bit]


    return go.Figure(data=[go.Bar(
        x=winMatrix['Class'],
        y=winMatrix[col],
        marker_color=plotlyCols
    )], layout={'width': SPEC_GRAPH_SIZE[0],
                'height': SPEC_GRAPH_SIZE[1],
                'margin': {'t': 0},
                'paper_bgcolor': BGCOLOR,
                'plot_bgcolor': BGCOLOR,
                'font': {'color': FONT_COLOR}
    })


def make_rating_plot(data, names):
    fig = go.Figure()
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
                      width=RATING_GRAPH_SIZE[0],
                      height=RATING_GRAPH_SIZE[1],
                      legend=dict(yanchor='top',
                                  y=0.99,
                                  xanchor='left',
                                  x=0.01),
                      margin={'t': 0},
                      paper_bgcolor=BGCOLOR,
                      plot_bgcolor=BGCOLOR,
                      font={'color': FONT_COLOR}
    )                                  

    return fig


def make_comp_table(data):
    opponentSpecs = analysis.get_opponent_specs(data)
    outcomes = analysis.get_outcome(data)
    ratingChange = analysis.get_player_field(data, 'Geebs', 'Rating change')

    cat_comp = lambda x:'('+', '.join(sorted([x['Player0'], x['Player1']])) +')'
    compOutcome = pd.DataFrame([opponentSpecs.apply(cat_comp, axis=1),
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
    Output('data-store', 'children'),
    [Input('upload', 'n_clicks')]
    )
def load_data(_):
    logging.info('Loading data.')
    # So after upload:
    # 1) update the data-store div with hidden JSON data
    # 2) update the partner selection option

    data2v2, data3v3 = rio.parse_lua_file(dataFile)
                                
    jsonData = '{"2v2":%s, "3v3":%s}'%(data2v2.to_json(),
                                       data3v3.to_json())

    size = sys.getsizeof(jsonData)/1e3
    logging.info('Data is %.2fKB in size.'%size)

    return jsonData


@app.callback(
    [Output('spec-graph', 'figure'),
     Output('rating-graph', 'figure'),
     Output('hidden-comp-table', 'children')],
    [Input('metric-selection', 'value'),
     Input('partner1-selection', 'value'),
     Input('partner2-selection', 'value'),
     Input('bracket-selection', 'value'),
     Input('data-store', 'children')])
def update_plots(metric, partner1, partner2, bracket, json_data):
    logging.info('Updating plots.')
    partners = [a for a in [partner1, partner2] if a is not None]
    allData = json.loads(json_data)
    bracketData = pd.DataFrame(allData[bracket])

    logging.info('Getting players')
    player, _ = analysis.get_player_and_team_mates(bracketData)
    logging.info('Filtering data')
    filteredData = filter_data(bracketData, partners)

    return (make_spec_plot(filteredData, metric),
            make_rating_plot(filteredData, [player]+partners),
            make_comp_table(filteredData).to_json())


@app.callback(
    Output('comp-table', 'data'),
    [Input('hidden-comp-table', 'children'),
     Input('class-selection-1', 'value'),
     Input('spec-selection-1', 'value'),
     Input('class-selection-2', 'value'),
     Input('spec-selection-2', 'value')
    ])
def filter_comp_table(comp_data, class1, spec1, class2, spec2):
    
    compTable = pd.DataFrame(json.loads(comp_data))

    if class1 is not None:
        filter1 = spec1 + class1 if spec1 is not None else class1
    else:
        filter1 = ''

    if class2 is not None:
        filter2 = spec2 + class2 if spec2 is not None else class2
    else:
        filter2 = ''

    filterIndex = compTable['Comp'].str.contains(filter1) &\
                  compTable['Comp'].str.contains(filter2)

    return compTable.loc[filterIndex, :].to_dict('records')

    

@app.callback(
    Output('sum-comp-table', 'data'),
    [Input('comp-table', 'data'),
     Input('comp-table', 'derived_virtual_data')]
    )
def update_sum_comp_table(compData, derivedData):

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


@app.callback(
    Output('class-selection-1', 'options'),
    [Input('comp-table', 'data')]
    )
def update_class1_selection(data):
    df = pd.DataFrame(data)
    classesSpecs = df.Comp.str.split(',').explode().str.strip()
    classes = classesSpecs.apply(lambda x: ' '.join(x.split(' ')[1:]))
    return [{'label': c, 'value': c} for c in set(classes)]

@app.callback(
    [Output('spec-selection-1', 'options'),
     Output('div-spec-selection-1', 'style'),
     Output('class-selection-2', 'options'),
     Output('div-class-selection-2', 'style')],
    [Input('class-selection-1', 'value')],
    [State('comp-table', 'data')])
def update_spec1_selection(class1, data):

    if class1 is None:
        return [], {'display': 'none'}, [], {'display': 'none'}
    
    df = pd.DataFrame(data)
    classesSpecs = df.Comp.str.split(',').explode().str.strip()
    classes = classesSpecs.apply(lambda x: ' '.join(x.split(' ')[1:]))
    specs = classesSpecs.apply(lambda x:x.split(' ')[0])

    specsForCurrentClass = specs[classes==class1]

    def f(x):
        z = [k for k in x if class1 not in k]
        if len(z) > 0:
            return z[0]
        else:
            return np.nan
    
    partnerClassesForCurrentClass = set(df.Comp.str.split(',').apply(f).dropna())

    return ([{'label': c, 'value': c} for c in set(specsForCurrentClass)],
            {'display': 'inline-block'},
            [{'label': c, 'value': c} for c in set(partnerClassesForCurrentClass)],
            {'display' : 'inline-block'})


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
    
    rating = analysis.get_player_rating(filteredData, player)

    ratingChange = (rating.iloc[-1] - rating.iloc[0])
    symbol = '+' if ratingChange >+ 0 else ''

    ratingChangeString = '%s%i'%(symbol, ratingChange)
    gamesPlayed = '%i'%len(rating)

    ratingChangeStyle = {'color' : green if ratingChange >= 0 else red}

    return ratingChangeString, gamesPlayed, ratingChangeStyle


if __name__ == "__main__":
    app.run_server(debug=True)