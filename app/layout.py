import dash_html_components as html
import dash_core_components as dcc
import dash_table


FONT_COLOR = '#cccccc'
BGCOLOR = 'rgba(0,0,0,0)'

TABLE_HEIGHT = 450

metricOptions = [
    {'label': 'Games played', 'value': 'N'},
    {'label': 'Win rate', 'value': 'win rate'},
    {'label': 'Average rating change', 'value': 'avg rating change'}]

tableColumns = ['Comp', 'Wins', 'Losses', 'Win rate (%)', 'Avg rating change']

NO_MARGIN = {'margin': '0px'}


TAB_STYLE = {'backgroundColor': '#111',
             'borderBottom': '0',
             'color': '#fff'}

SELECTED_TAB_STYLE = {'backgroundColor': '#222',
                      'borderBottom': '0',
                      'color': '#fff'}


def generate_menu():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.P('Rating change', className='app__metric__title'),
                    html.Hr(className='kpi-divider'),
                    html.P('', id='metric-rating-change',
                           className='app__metric__value')
                ], className='div-metric-rating'),
                
                html.Div([
                    html.P('Games played', className='app__metric__title'),
                    html.Hr(className='kpi-divider'),
                    html.P('', id='metric-games-played',
                           className='app__metric__value')
                ], className='div-metric-rating')
                
            ], id='div-metric-kpi-container'),
            html.Div([
                html.H4('Bracket', className='app__menu__title'),
                dcc.RadioItems(options=[{'label': '2v2', 'value': '2v2'},
                                        {'label': '3v3', 'value': '3v3'}],
                               value='2v2', id='bracket-selection')
            ], id='div-radio-bracket'),
            
            html.Div([
                html.H4('Filter by partner', className='app__menu__title'),
                html.Div([
                    dcc.Dropdown(id='partner1-selection',
                                 options=[], multi=False)
                ], id='div-dropdown-partner1', className='dash-bootstrap'),
                
                html.Div([
                    dcc.Dropdown(id='partner2-selection',
                                 options=[], multi=False)
                    
                ], id='div-dropdown-partner2', style={'display': 'none'},
                         className='dash-bootstrap')
            ], id='div-dropdown-partner')
        ], id='div-radio-partner')
    ], id='div-main-menu')


def generate_main_content():
    return html.Div([
        html.Div([dcc.Tabs(id='main-tab', value='graph',
                           parent_className='tabs-parent',
                           children=[
                               dcc.Tab(label='Graphical', value='graph',
                                       className='menu-tab',
                                       style=TAB_STYLE,
                                       selected_style=SELECTED_TAB_STYLE),

                               dcc.Tab(label='Compositions', value='comp',
                                       className='menu-tab',
                                       style=TAB_STYLE,
                                       selected_style=SELECTED_TAB_STYLE)
                           ])], id='tab-container'),
        html.Div(id='div-tab-content')],
                    id='div-tab-main')


def generate_left_column():
    return html.Div([
        html.Div(id='div-metric-selection', children=[
            html.H3('Metric', style={'margin-bottom': '0px'}),
            dcc.Dropdown(id='metric-selection', options=metricOptions,
                         value='N', multi=False, clearable=False,
                         searchable=False)],
                 className='dash-bootstrap'),
        dcc.Graph(id='spec-graph', style={'width': '90%'})],
                    id='div-main-left')


def generate_central_column():
    return html.Div([
        dcc.Graph(id='rating-graph',
                  style={'width': '90%'})
    ], id='div-main-center')


def layout_comp_table():
    return html.Div([
        html.H2('Filter by class'),
        dcc.Dropdown(options=[{'value': 'Rogue',
                               'label': 'Rogue'}],
                     id='class-selection-1',
                     value='Rogue'),

        html.Div([
            html.H4('Class 1'),
            html.Div([
            ], className='div-class-selection dash-bootstrap'),
            html.Div([
                dcc.Dropdown(options=[], id='spec-selection-1')
            ], className='div-class-selection',
                     style={'display': 'none'}, id='div-spec-selection-1')
        ], className='div-class-spec'),
        html.Div([            
            html.Div([
                html.H4('Class 2'),
                dcc.Dropdown(options=[], id='class-selection-2')
            ], className='div-class-selection',
                     style={'display': 'none'}),
            html.Div([
                dcc.Dropdown(options=[], id='spec-selection-2')
            ], className='div-class-selection',
                     style={'display': 'none'},
                     id='div-class-selection-2')
        ], className='div-class-spec'),
        html.Div([
            dash_table.DataTable(id='comp-table',
                                 columns=[{'name': col, 'id': col,
                                           'presentation': 'markdown'}
                                          for col in tableColumns],
                                 sort_action='native',
                                 style_table={'height': TABLE_HEIGHT,
                                              'overflowY': 'scroll'},
                                 style_header={'height': 'auto',
                                               'font-size': '22pt',
                                               'font-family':
                                               'Helvetica, Arial, sans-serif',
                                               'backgroundColor': '#000',
                                               'text-align': 'center'},
                                 style_cell={'height': 'auto',
                                             'whiteSpace': 'normal'},
                                 css=[dict(
                                     selector='.dash-spreadsheet td div',
                                     rule='''
                                     min-width: 15vw;
                                     display: block;
                                     font-size: 18pt;
                                     text-align : center;
                                     padding : 0;
                                     '''),
                                      dict(selector='tr:hover',
                                           rule='background-color:#111')],
                                 style_data={'backgroundColor': BGCOLOR}),
            html.Br(),
            html.Br(),
            dash_table.DataTable(id='sum-comp-table',
                                 columns=[{'name': col, 'id': col}
                                          for col in tableColumns]
            )
        ])
    ], id='div-main-left')


def generate_upload_menu():
    return html.Div([
        html.Button('Upload', id='upload'),
    ], id='div-bottom')


def generate_layout():
    return html.Div([
        html.Div(id='data-store', style={'display': 'none'}),
        html.Div(id='player-name', style={'display': 'none'}),
        html.Div(id='hidden-comp-table', style={'display': 'none'}),
        html.Div(id='div-main-top', children=[
            html.H3('~ REFlexology ~', className='app__header__title'),
            html.H4('by Geebs', className='app__header__subtitle')
        ]),
        generate_menu(),
        html.Div([
            generate_main_content()
        ], id='div-main-mother'),
        generate_upload_menu()
    ], id='div-mother')
