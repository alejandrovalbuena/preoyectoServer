import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.express as px

app = dash.Dash(__name__)
server = app.server

top_50_sp500 = [
    {'label': 'Apple Inc', 'value': 'AAPL'},
    {'label': 'Microsoft Corporation', 'value': 'MSFT'},
    {'label': 'Amazon.com Inc', 'value': 'AMZN'},
    {'label': 'Trimble Inc.', 'value': 'TRMB'},
    {'label': 'Facebook, Inc. (Meta Platforms)', 'value': 'META'},
    {'label': 'Alphabet Inc. (Class A)', 'value': 'GOOGL'},
    {'label': 'Tesla, Inc.', 'value': 'TSLA'},
    {'label': 'NVIDIA Corporation', 'value': 'NVDA'},
    {'label': 'JPMorgan Chase & Co.', 'value': 'JPM'},
    {'label': 'Johnson & Johnson', 'value': 'JNJ'},
    {'label': 'Visa Inc.', 'value': 'V'},
    {'label': 'UnitedHealth Group Incorporated', 'value': 'UNH'},
    {'label': 'Home Depot, Inc.', 'value': 'HD'},
    {'label': 'Procter & Gamble', 'value': 'PG'},
    {'label': 'Mastercard Incorporated', 'value': 'MA'},
    {'label': 'Bank of America Corp', 'value': 'BAC'},
    {'label': 'Walt Disney Company', 'value': 'DIS'},
    {'label': 'Adobe Inc.', 'value': 'ADBE'},
    {'label': 'Salesforce.com', 'value': 'CRM'},
    {'label': 'Comcast Corporation', 'value': 'CMCSA'},
    {'label': 'Netflix, Inc.', 'value': 'NFLX'},
    {'label': 'Cisco Systems, Inc.', 'value': 'CSCO'},
    {'label': 'Pfizer Inc.', 'value': 'PFE'},
    {'label': 'Intel Corporation', 'value': 'INTC'},
    {'label': 'Verizon Communications Inc.', 'value': 'VZ'},
    {'label': 'Coca-Cola Company', 'value': 'KO'},
    {'label': 'AT&T Inc.', 'value': 'T'},
    {'label': 'Merck & Co., Inc.', 'value': 'MRK'},
    {'label': 'PepsiCo, Inc.', 'value': 'PEP'}
]


def fetch_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def calculate_rsi(dataframe, window=14):
    delta = dataframe['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    rsi = rsi.rename("RSI")
    return rsi

def calculate_bollinger_bands(dataframe, window=20, num_of_std=2):
    rolling_mean = dataframe['Close'].rolling(window=window).mean()
    rolling_std = dataframe['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    
    return rolling_mean, upper_band, lower_band

def fetch_hourly_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(interval="1h", start=start_date, end=end_date)
    return df

def perform_hourly_linear_regression(dataframe, hours_ahead=24):
    dataframe = dataframe.sort_index()
    hourly_data = dataframe
    
    hourly_data['NumericIndex'] = range(len(hourly_data))
    X = hourly_data[['NumericIndex']] 
    y = hourly_data['Close']


    model = LinearRegression()
    model.fit(X, y)

    last_numeric_index = hourly_data['NumericIndex'].iloc[-1]
    future_indices = np.array(range(last_numeric_index + 1, last_numeric_index + hours_ahead + 1)).reshape(-1, 1)
    future_preds = model.predict(future_indices)

    y_pred = model.predict(X)
    rmse = sqrt(mean_squared_error(y, y_pred))
    
    lower_bound = future_preds - 1.96 * rmse
    upper_bound = future_preds + 1.96 * rmse
    
    return hourly_data.index, hourly_data['Close'], future_preds, lower_bound, upper_bound, model

def find_anomalies(data, std_multiplier=3):
    anomalies = []
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * std_multiplier
    
    lower_limit = data_mean - anomaly_cut_off 
    upper_limit = data_mean + anomaly_cut_off

    for index, value in enumerate(data):
        if value > upper_limit or value < lower_limit:
            anomalies.append((index, value))
    return anomalies


app.layout = html.Div([
    html.Div([
        html.H1('Trader Assistant', style={'text-align': 'center', 'margin-bottom': '50px', 'color': '#007BFF', 'font-family': 'Arial', 'font-size': '40px'})
    ], style={'background': '#333', 'padding': '20px', 'font-family': 'Arial'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='stock-selector',
                options=top_50_sp500,
                value='TRMB',
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            dcc.Dropdown(
                id='time-range-selector',
                options=[
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'}
                ],
                value='1y', 
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            html.Button('Select All Indicators', id='select-all-button', style={'width': '100%', 'margin-bottom': '20px'}),
            dcc.Checklist(
                id='ma-options-selector',
                options=[
                    {'label': 'Show 200-session Moving Average', 'value': 'MA200'},
                    {'label': 'Show 100-session Moving Average', 'value': 'MA100'},
                    {'label': 'Show 50-session Moving Average', 'value': 'MA50'}
                ],
                value=[], 
                style={'width': '100%', 'margin-bottom': '20px'}
            ),

            html.H4("Select main statistics to compare", 
                    style={'width': '100%', 'margin-top': '250px', 'margin-bottom': '20px'}),


            dcc.Checklist(
                id='indicator-selector',
                options=[
                    {'label': 'Show RSI', 'value': 'RSI'},
                    {'label': 'Show Bollinger Bands', 'value': 'BOLL'}
                ],
                value=[],
                style={'width': '100%', 'margin-bottom': '50px'}
),
            html.H4("Compare up to 4 stocks using a confusion matrix", 
                    style={'width': '100%', 'margin-top': '900px', 'margin-bottom': '20px'}),

            dcc.Dropdown(
                id='stock-selector-1',
                options=top_50_sp500,
                value='AAPL', 
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            dcc.Dropdown(
                id='stock-selector-2',
                options=top_50_sp500,
                value='MSFT',
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            dcc.Dropdown(
                id='stock-selector-3',
                options=top_50_sp500,
                value=top_50_sp500[2]['value'], 
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            dcc.Dropdown(
                id='stock-selector-4',
                options=top_50_sp500,
                value=top_50_sp500[3]['value'],
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
        ], style={'flex': '1', 'margin-right': '20px', 'font-family': 'Arial'}),
        html.Div([
            dcc.Graph(id='stock-price-graph', style={'margin-bottom': '20px'}),
            dcc.Graph(id='indicator-graph', style={'margin-bottom': '20px'}),
            dcc.Graph(id='linear-regression-prediction-graph', style={'margin-bottom': '20px'}),
            dcc.Graph(id='correlation-graph', style={'margin-bottom': '20px'}),
            dcc.Graph(id='anomaly-graph')
        ], style={'flex': '3', 'font-family': 'Arial'}),
        
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between', 'font-family': 'Arial'}),

], style={'padding': '20px', 'font-family': 'Arial'})



app.css.append_css({"external_url": "assets/style.css"})

@app.callback(
    Output('ma-options-selector', 'value'),
    Output('indicator-selector', 'value'),
    Input('select-all-button', 'n_clicks'),
    State('ma-options-selector', 'options'),
    State('indicator-selector', 'options'),
    prevent_initial_call=True
)
def select_all(n_clicks, ma_options, indicator_options):
    if n_clicks is None:
        raise PreventUpdate
    ma_values = [option['value'] for option in ma_options]
    indicator_values = [option['value'] for option in indicator_options]

    return ma_values, indicator_values

@app.callback(
    Output('stock-price-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('ma-options-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_graph(selected_stock, ma_options, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name=f'{selected_stock} Close Price'))
    if 'MA200' in ma_options:
        df_stock['200_MA'] = df_stock['Close'].rolling(window=200, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['200_MA'], mode='lines', name='200 Session MA', line=dict(color='red')))

    if 'MA100' in ma_options:
        df_stock['100_MA'] = df_stock['Close'].rolling(window=100, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['100_MA'], mode='lines', name='100 Session MA', line=dict(color='green')))

    if 'MA50' in ma_options:
        df_stock['50_MA'] = df_stock['Close'].rolling(window=50, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['50_MA'], mode='lines', name='50 Session MA', line=dict(color='orange')))

    fig.update_layout(title='Stock Price and Moving Averages')


    return fig


@app.callback(
    Output('indicator-graph', 'figure'),
    [Input('stock-selector', 'value'),
     Input('indicator-selector', 'value'),
     Input('time-range-selector', 'value')]
)
def update_indicator_graph(selected_stock, selected_indicators, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    
    if df_stock.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if 'RSI' in selected_indicators:
        rsi = calculate_rsi(df_stock)
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=rsi, mode='lines', name='RSI'),
            secondary_y=False,
        )

    if 'BOLL' in selected_indicators:
        rolling_mean, upper_band, lower_band = calculate_bollinger_bands(df_stock)
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='green')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=rolling_mean, mode='lines', name='Middle Bollinger Band', line=dict(color='blue')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='red')),
            secondary_y=True,
        )

    fig.update_layout(title=f'{selected_stock} Indicators')
    fig.update_yaxes(title_text="RSI", secondary_y=False)
    fig.update_yaxes(title_text="Bollinger Bands", secondary_y=True)

    return fig

@app.callback(
    Output('linear-regression-prediction-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_hourly_linear_regression_graph(selected_stock, selected_time_range):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(weeks=4) 

    
    df_stock = fetch_hourly_stock_data(selected_stock, start_date, end_date)
    if df_stock.empty:
        return go.Figure()

    historical_dates, historical_prices, future_preds, lower_bound, upper_bound, _ = perform_hourly_linear_regression(df_stock)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices'))
    future_dates = pd.date_range(start=historical_dates[-1], periods=len(future_preds)+1, freq='H', closed='right')
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Linear Regression Predictions', line=dict(color='orange')))
    
    fig.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='lines', name='Lower Confidence Bound', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', name='Upper Confidence Bound', line=dict(color='green')))
    
    fig.add_traces([go.Scatter(x=list(future_dates)+list(future_dates)[::-1],
                               y=list(upper_bound)+list(lower_bound)[::-1],
                               fill='toself',
                               fillcolor='rgba(231,107,243,0.2)',
                               line=dict(color='rgba(255,255,255,0)'),
                               name='Confidence Interval')])

    fig.update_layout(title='Hourly Stock Price Prediction with Linear Regression and Confidence Interval')
    
    return fig

@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('stock-selector-1', 'value'),
     Input('stock-selector-2', 'value'),
     Input('stock-selector-3', 'value'),
     Input('stock-selector-4', 'value')]
)
def update_correlation_graph(stock1, stock2, stock3, stock4):
    dfs = []
    for stock in [stock1, stock2, stock3, stock4]:
        df = fetch_stock_data(stock, '2y')[['Close']].rename(columns={'Close': stock})
        dfs.append(df)

    df_combined = pd.concat(dfs, axis=1)
    correlation = df_combined.corr()
    fig = px.imshow(correlation, text_auto=True, aspect='auto')

    fig.update_layout(
        title='4x4 Stock Correlation Matrix',
        xaxis_nticks=4,
        yaxis_nticks=4
    )

    return fig

@app.callback(
    Output('anomaly-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_anomaly_graph(selected_stock, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    closing_prices = df_stock['Close'].tolist()
    
    anomalies = find_anomalies(closing_prices, std_multiplier=2.5) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name='Close Price'))
    
    for anomaly in anomalies:
        fig.add_trace(go.Scatter(x=[df_stock.index[anomaly[0]]], 
                                 y=[anomaly[1]], 
                                 mode='markers', 
                                 marker=dict(color='red', size=10),
                                 name='Anomaly'))
    
    fig.update_layout(title='Stock Price Anomaly Detection')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)