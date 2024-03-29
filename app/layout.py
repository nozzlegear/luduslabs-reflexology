from dash import html, dcc, dash_table
import requests

FONT_COLOR = '#d0d0d0'
BGCOLOR = 'rgba(30, 48, 80, 0.0)'

TABLE_HEIGHT = 400


metricOptions = [
    {'label': 'Total rating change', 'value': 'total rating change'},    
    {'label': 'Games played', 'value': 'N'},
    {'label': 'Win rate', 'value': 'win rate'},
    {'label': 'Average rating change', 'value': 'avg rating change'}]


tableColumns = ['Comp', 'Games', 'Record', 'Win rate (%)',
                'Avg rating change', 'Total rating change']

NO_MARGIN = {'margin': '0px'}


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
                ], className='div-metric-rating'),
                html.Div([
                    html.P('Sessions played', className='app__metric__title'),
                    html.Hr(className='kpi-divider'),
                    html.P('', id='metric-sessions-played',
                           className='app__metric__value')
                ], className='div-metric-rating')
                
                
            ], id='div-metric-kpi-container'),
            html.Div(id='div-season-selection', children=[
                html.H4('Season', className='app__menu_title'),
                html.Div(className='dash-bootstrap', 
                         children=[
                             dcc.Dropdown(id='season-selection', options=[],
                                          multi=False, clearable=False)
                         ])
                ]),
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
            ], id='div-dropdown-partner'),
            generate_link_share()
        ], id='div-radio-partner')

    ], id='div-main-menu')


def generate_main_content():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div([dcc.Tabs(id='main-tab', value='graph',
                           parent_className='tabs-parent',
                           children=[
                               dcc.Tab(label='Graphical overview', value='graph',
                                       className='menu-tab'),
                               dcc.Tab(label='Compositions', value='comp',
                                       className='menu-tab')
                               # dcc.Tab(label='Analysis', value='analysis',
                               #         className='menu-tab',
                               #         style=TAB_STYLE,
                               #         selected_style=SELECTED_TAB_STYLE)
                           ])
        ], id='tab-container'),
        html.Div(id='div-tab-1', className='div-tab-content',
                 children=[generate_metric_menu(),
                html.Div(id='div-graphs',children=[
                    generate_spec_graph(),
                    generate_rating_graph()])
                 ]),
        html.Div(id='div-tab-2', className='div-tab-content',
                 children=[layout_comp_table()]),
        html.Div(id='div-tab-3', className='div-tab-content',
                 children=[generate_analysis_page()])
    ],
                    id='div-tab-main')


def generate_spec_graph():
    return html.Div(children=[
        dcc.Graph(id='spec-graph',
                  style={'width': '100%',
                         'display': 'none'},
                  config={'displayModeBar': False})],
                    id='div-spec-graph')


def generate_metric_menu():
    return html.Div([
        html.Div(id='div-metric-selection-1',
                 children=[
                     html.H3('Metric', className='metric-title'),
                     dcc.Dropdown(id='metric-selection', options=metricOptions,
                                  value='total rating change', multi=False, clearable=False,
                                  searchable=False)
                 ], className='dash-bootstrap div-metric-selection'),
        
        html.Div(id='div-metric-selection-2', children=[
            html.H3('Group by', className='metric-title'),
            dcc.RadioItems(options=[{'value': 'class', 'label': 'Class'},
                                    {'value': 'spec', 'label': 'Spec'}],
                           value='class', id='radio-class-spec')],
                 className='dash-bootstrap div-metric-selection'),
        html.P(children=["Tip: Select a time range you're interested in. ",
                         html.Br(),
                         "Double click on the plot to reset the filter."],
               id='tip-time-range')
    ], 
                    id='div-metric-menu')


def generate_rating_graph():
    return html.Div([
        dcc.Graph(id='rating-graph',
                  style={'width': '90%', 'display': 'none'},
                  config={'displayModeBar': False})
    ], id='div-rating-graph')


def generate_comp_menu():
    return html.Div([
        html.Div([
            html.H3('Opponent 1'),
            dcc.Dropdown(options=[],
                         id='class-selection-1',
                         placeholder='Filter by class'),
            dcc.Dropdown(options=[], id='spec-selection-1',
                         placeholder='Filter by spec')
            ], id='div-class-selection-1'),
        html.Div([
            html.H3('Opponent 2'),            
            dcc.Dropdown(options=[], id='class-selection-2',
                         placeholder='Filter by class'),
            dcc.Dropdown(options=[], id='spec-selection-2',
                         placeholder='Filter by spec')
            ], id='div-class-selection-2', style={'display': 'none'}),

        html.Div([
            dcc.Checklist(options=[
                {'value': 'grouped', 'label': 'Group healers'}
                ], id='group-healers')],
                 id='div-group-healers'),
        
        html.Div([
            html.Button('Export to CSV', 'download-button',
                        className="reflexology-button"),
            dcc.Download(id='download-csv')
        ], id='div-download')

    ], className='div-class-selection-outer dash-bootstrap')


def layout_comp_table():
    return html.Div([
        html.Div([
            generate_comp_menu()
        ], id='div-comp-menu'),
        html.Div([
            dash_table.DataTable(id='comp-table',
                                 columns=[{'name': col, 'id': col,
                                           'presentation': 'markdown'}
                                          for col in tableColumns],
                                 sort_action='native',
                                 style_table={'height': TABLE_HEIGHT,
                                              'overflowY': 'scroll'},
                                 style_header={'height': 'auto',
                                               'fontSize': '15pt',
                                               'fontFamily':
                                               'Helvetica, Arial, sans-serif',
                                               'backgroundColor': 'rgba(30, 48, 80, 1.0)',
                                               'textAlign': 'center'},
                                 style_cell={'height': 'auto',
                                             'whiteSpace': 'normal'},
                                 css=[dict(
                                     selector='.dash-spreadsheet td div',
                                     rule='''
                                     min-width: 10vw;
                                     display: block;
                                     font-size: 12pt;
                                     text-align: center;
                                     padding: 0;
                                     '''),
                                      dict(selector='tr:hover',
                                           rule='background-color: rgba(30, 48, 80, 0.5)')],
                                 style_data={'backgroundColor': 'rgba(30, 48, 80, 0.8)' })
        ], id='div-comp-content')], id='div-main-right')


def generate_analysis_page():
    return html.Div(id='analysis', children=[])


def generate_greeting_page():
    return html.Div([
    html.P('''
REFlexology lets you upload your own REFlex data and gives you a breakdown
of your arena performance. If you just want to take REFlexology for
a spin, hit the demo button.''', className='greet-string'),
        
    html.Div(children=[
        html.Button(id='button-demo', children="Let's do a demo",
                    className="reflexology-button")],
            id='div-demo'),
    html.P('''
    If you want to use REFlexology on your own REFlex data, the file you are after
    is called REFlex.lua and is buried deep in your WoW folder:
    ''', className='greet-string'),
        html.P('''
        World of Warcraft\_retail_\WTF\Account\<ACCOUNTID>\<Server>\<Character>\SavedVariables\REFlex.lua
        ''', className='greet-string'),
        
    html.P([html.B('The fine print: '), '''
    We need to store and process your REFlex data in order to provide this service.
    This data contains information about your arena games.
    It should go without saying that uploading your data means you consent to us 
    handling it, but data protection legislation (GDPR) means that we have to 
    be explicit about getting consent. This is a good thing, though. If you're happy 
    with us storing and processing your REFlex data, do carry on using the tool. If you 
    don't want us handling your data, you can still check out the Demo feature.'''])
    ]
)

def generate_logo():
    return html.Div([html.Img(src='/static/logo.png',
                              style={'width': '180px',
                                     'align': 'center'})],
                    style={'textAlign': 'center', 'width':'100%'})


def generate_upload_menu():
      return html.Div([
          dcc.Upload(id='upload',
                     children=html.Div([
                         'Drag and Drop or ',
                         html.A('Click to Select Files')
                     ]),
                     style={
                         'width': '100%',
                         'height': '80px',
                         'lineHeight': '60px',
                         'borderWidth': '1px',
                         'borderStyle': 'dashed',
                         'borderRadius': '5px',
                         'textAlign': 'center',
                         'margin': '10px'
                     },
                     multiple=False,
                     max_size=5*10**7)
          ], id='div-upload')


def generate_character_confirmation():

    return html.Div([
        dcc.RadioItems(options=[], id='player-name-toggle'),
        html.Button(id='player-name-confirm', value='Confirm')
    ], id='player-name-toggle-div')


def generate_link_share():
    return html.Div([
        html.P(
            id="linkshare",
            children=["https://luduslabs.org/reflexology"]
        ),
        html.P('Shareable link', id='p-linkshare'),
        dcc.Clipboard(
            id='clipboard',
            target_id="linkshare",
            title="Copy link to clipboard",
        )], id='div-linkshare')


def generate_layout():
    return html.Div(children=[html.Div([
        html.Div(id='opponent-data', style={'display': 'none'}),
        html.Div(id='player-name', style={'display': 'none'}),
        html.Div(id='hidden-comp-table', style={'display': 'none'}),
        html.Div(id='player-name-validation', style={'display': 'none'}),
        generate_menu(),
        html.Div([
            generate_main_content()
        ], id='div-main-mother')
    ], id='div-main-content', style={'display': 'none'}),
                              html.Div(children=[
                                  generate_greeting_page(),
                                  dcc.Loading(id='loading-data-store',
                                              type='circle',
                                              children=html.Div(id='data-store')),
                                  generate_upload_menu(),
                                  generate_logo()], id='div-main-greet',
                                       style={'display': 'block'})
    ], id='div-mother')
