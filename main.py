import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.dash import no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import sqrt
from typing import *

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = 'Automatyka'

def round_num(x):
    if x > 1 or x < -1:
        x = "%.2F " % x
    else:
        x = "%.2e" % x
    return x

app.layout = html.Div([
    html.H1(children=['AUTOMATYKA', html.Br(), html.P(id="subtitle", children="Model matematyczny układu automatycznej regulacji w podgrzewaczu przepływowym")]),
    html.P(id="description", children="Przedstawiony model matematyczny regulatora typu PID pozwala na ustawienie zadanej temperatury wody oraz parametrów podgrzewacza. "
                                      "Wielkością sterującą układem jest napięcie grzałki podgrzewającej ciecz, którą jest woda. Wyświetlane są wykresy zależności temperatury, "
                                      "sygnału sterującego,napięcia i mocy od czasu. "
                                      "Dwa niezależne pola do wprowadzenia parametrów umożliwiają wyświetlanie i porównanie wykresów dla różnych danych wejściowych. "
                                      "Dodatkowo obliczane są wskazniki jakości regulacji."),
    #podgrzewacz 1
    html.Div(id="block", children=[
        html.Div(className='sliders', children=["Wybierz zadaną temperaturę i parametry podgrzewacza:", html.Br(),
            html.Div(className="temperature", children=["T [30 - 70 °C]: ", dcc.Input(className='slider', id='T', min=30, max=70, value=50, step=1, type='number'), html.Br(),]),
                html.Div(className="parameters", children=[
                    html.Div(className="left", children=[
                        "P [3.5 - 7.0 kW]: ", dcc.Input(className='slider', id='P', min=3.5, max=7, value=5, step=0.1, type='number'), html.Br(),
                        "V [1 - 4 l]: ", dcc.Input(className='slider', id='V', min=1, max=4, value=2, step=0.1, type='number'), html.Br(),
                        "Tw [18-20 °C]: ", dcc.Input(className='slider', id='Tw', min=18, max=20, value=20, step=0.1, type='number'), html.Br(),
                        "dw [1.85 - 2.15 l/min]: ", dcc.Input(className='slider', id='dw', min=1.85, max=2.15, value=2, step=0.01, type='number'), html.Br(),
                        "Umax [220 - 240 V]: ", dcc.Input(className='slider', id='Umax', min=220, max=240, value=230, step=0.1, type='number'), html.Br(),
                    ]),
                    html.Div(className="right", children=[
                        "t [10 - 60 min]: ", dcc.Input(className='slider', id='t', min=10, max=60, value=30, step=1, type='number'), html.Br(),
                        "Tp [0.01 - 1 s]: ", dcc.Input(className='slider', id='Tp', min=0.01, max=1, value=0.1, step=0.01, type='number'), html.Br(),
                        "kp [0 - 100]: ", dcc.Input(className='slider', id='kp', min=0, max=100, value=0.0011, step=0.00001, type='number'), html.Br(),
                        "Ti [0 - 10]: ", dcc.Input(className='slider', id='Ti', min=0, max=10, value=0.6, step=0.01, type='number'), html.Br(),
                        "Td [0 - 10]: ", dcc.Input(className='slider', id='Td', min=0, max=10, value=0.12, step=0.01, type='number'), html.Br(),
                    ]),
                ]),
            html.Div(id='loading', children=[
                dbc.Button("Pokaż wykres", id='submit-button-state', n_clicks=0),
                dbc.Alert("Wprowadzono wartości spoza zakresu! Należy poprawić pola oznaczone czerwoną ramką.", color="danger", id="alert-auto", is_open=False, duration=10000)
            ])
        ]),

    #legenda
    html.Div(className="legend", children=[html.P(id="legend_title", children=["Legenda"]),
        "T - temperatura zadana", html.Br(), "P - moc grzałki", html.Br(), "V - objętość boilera", html.Br(), "Tw - temperatura początkowa wody", html.Br(), "dw - dopływ wody", html.Br(), "Umax - max. napięcie na grzałce", html.Br(),
        "t - czas symulacji", html.Br(), "Tp - okres próbkowania", html.Br(), "kp - wzmocnienie regulatora", html.Br(), "Ti - czas zdwojenia", html.Br(), "Td - czas wyprzedzania"
    ]),

    #podgrzewacz 2
        html.Div(className='sliders', children=["Wybierz zadaną temperaturę i parametry podgrzewacza:", html.Br(),
            html.Div(className="temperature", children=["T [30 - 70 °C]: ", dcc.Input(className='slider', id='T1', min=30, max=70, value=50, step=1, type='number'), html.Br(),]),
                html.Div(className="parameters", children=[
                    html.Div(className="left", children=[
                        "P [3.5 - 7.0 kW]: ", dcc.Input(className='slider', id='P1', min=3.5, max=7, value=5, step=0.1, type='number'), html.Br(),
                        "V [1 - 4 l]: ", dcc.Input(className='slider', id='V1', min=1, max=4, value=2, step=0.1, type='number'), html.Br(),
                        "Tw [18-20 °C]: ", dcc.Input(className='slider', id='Tw1', min=18, max=20, value=20, step=0.1, type='number'), html.Br(),
                        "dw [1.85 - 2.15 l/min]: ", dcc.Input(className='slider', id='dw1', min=1.85, max=2.15, value=2, step=0.01, type='number'), html.Br(),
                        "Umax [220 - 240 V]: ", dcc.Input(className='slider', id='Umax1', min=220, max=240, value=230, step=0.1, type='number'), html.Br(),
                    ]),
                    html.Div(className="right", children=[
                        "t [10 - 60 min]: ", dcc.Input(className='slider', id='t1', min=10, max=60, value=30, step=1, type='number'), html.Br(),
                        "Tp [0.01 - 1 s]: ", dcc.Input(className='slider', id='Tp1', min=0.01, max=1, value=0.1, step=0.01, type='number'), html.Br(),
                        "kp [0 - 100]: ", dcc.Input(className='slider', id='kp1', min=0, max=100, value=0.0011, step=0.00001, type='number'), html.Br(),
                        "Ti [0 - 10]: ", dcc.Input(className='slider', id='Ti1', min=0, max=10, value=0.6, step=0.01, type='number'), html.Br(),
                        "Td [0 - 10]: ", dcc.Input(className='slider', id='Td1', min=0, max=10, value=0.12, step=0.01, type='number'), html.Br(),
                    ]),
                ]),
            html.Div(id='loading1', children=[
                dbc.Button("Pokaż wykres", id='submit-button-state1', n_clicks=0),
                dbc.Alert("Wprowadzono wartości spoza zakresu! Należy poprawić pola oznaczone czerwoną ramką.", color="danger", id="alert-auto1", is_open=False, duration=10000)
            ])
        ]),
    ]),

    html.Div(className="regulation_block", children=[
        html.Div(className="regulation", children=[
            "Czas regulacji: ", html.Output(id="czas"), html.Br(),
            "Przeregulowanie: ", html.Output(id="przeregulowanie"), html.Br(),
            "Uchyb ustalony: ", html.Output(id="uchyb"), html.Br(),
            "Dokładność regulacji: ", html.Output(id="dokladnosc"), html.Br(),
            "Koszty regulacji: ", html.Output(id="koszty")
        ]),
    ]),

    html.Div(className="regulation_block", children=[
        html.Div(className="regulation", children=[
            "Czas regulacji: ", html.Output(id="czas1"), html.Br(),
            "Przeregulowanie: ", html.Output(id="przeregulowanie1"), html.Br(),
            "Uchyb ustalony: ", html.Output(id="uchyb1"), html.Br(),
            "Dokładność regulacji: ", html.Output(id="dokladnosc1"), html.Br(),
            "Koszty regulacji: ", html.Output(id="koszty1")
        ]),
    ]),

    html.Div(className="graph_display", children=[
        dcc.Loading(id='loading_icon', type='circle', children=[dcc.Graph(id='the_graph')])
    ]),
    html.Div(className="graph_display", children=[
        dcc.Loading(id='loading_icon1', type='circle', children=[dcc.Graph(id='the_graph1')])
    ]),
    html.P(id="footer", children=["2021 © Wiktor Jankowski, Patryk Jędrzejewski, Filip Kaczmarek"])
])

@app.callback(
    Output('the_graph', 'figure'),
    Output('alert-auto', 'is_open'),
    Output('czas', 'children'),
    Output('przeregulowanie', 'children'),
    Output('uchyb', 'children'),
    Output('dokladnosc', 'children'),
    Output('koszty', 'children'),
    [Input('submit-button-state', 'n_clicks')],
    [State('alert-auto', 'is_open'),
    State('T', 'value'),
    State('P', 'value'),
    State('V', 'value'),
    State('Tw', 'value'),
    State('dw', 'value'),
    State('kp', 'value'),
    State('Ti', 'value'),
    State('Td', 'value'),
    State('Umax', 'value'),
    State('t', 'value'),
    State('Tp', 'value')])

def update_output(n_clicks, is_open, selected_T, selected_P, selected_V, selected_Tw, selected_dw, selected_kp, selected_Ti, selected_Td, selected_Umax, selected_t, selected_Tp):

    if n_clicks == 0:
        return no_update, is_open, no_update, no_update, no_update, no_update, no_update
    if (selected_T is None) or (selected_P is None) or (selected_V is None) or (selected_Tw is None) or (selected_dw is None) or (selected_kp is None) or (selected_Ti is None) or (selected_Td is None) or (selected_Umax is None) or (selected_t is None) or (selected_Tp is None):
        return no_update, not is_open, no_update, no_update, no_update, no_update, no_update

    T_wody_wplywajacej_do_boilera = selected_Tw
    doplyw_wody = selected_dw/60000
    gestosc_cieczy = 1000
    cieplo_wlasciwe = 4190
    U_minimalne = 0
    U_maksymalne = selected_Umax

    V_boilera = selected_V / 1000  # [m3]  0.005<V<0.02
    opor = U_maksymalne**2/(1000*selected_P)  # 0 < R < 50
    T_zadane = selected_T

    T_maksymalne = 80

    kp = selected_kp
    Ti = selected_Ti
    Td = selected_Td
    Tp = selected_Tp
    czas_symulacji = selected_t*60
    n = int(czas_symulacji / Tp)
    rozmiar = n + 1

    T: List[float] = [T_wody_wplywajacej_do_boilera]
    U: List[float] = [U_minimalne]
    P: List[float] = [U_minimalne * U_minimalne / opor]

    # regulator PID
    e: List[float] = [0]
    u_min: float = 0
    u_max: float = 10
    u: List[float] = [0]
    u_unlimited: List[float] = [0]

    # model regulacyjny
    a = (U_maksymalne - U_minimalne) / (u_max - u_min)
    b = U_minimalne - u_min * a

    for x in range(n):
        e.append(T_zadane - T[-1])
        u_unlimited.append((kp * (e[-1] + (Tp / Ti) * sum(e) + (Td / Tp) * (e[-1] - e[-2]))))
        u.append(max(u_min, min(u_max, u_unlimited[-1])))
        U.append(max(U_minimalne, min(U_maksymalne, a * u[-1] + b)))
        P.append(U[-1] * U[-1] / opor)
        T.append(max(T_wody_wplywajacej_do_boilera, min(T_maksymalne, (Tp / (V_boilera * gestosc_cieczy * cieplo_wlasciwe)) * (doplyw_wody * gestosc_cieczy * cieplo_wlasciwe * (T_wody_wplywajacej_do_boilera - T[-1]) + (U[-1] * U[-1] / opor)) + T[-1])))


    for x in range(n, 1, -1):
        if (T[x] <= T_zadane * 0.95 or T[x] >= T_zadane * 1.05):
            czas_regulacji = x
            break

    czas_regulacji = czas_regulacji * Tp / 60
    czas_regulacji = round_num(czas_regulacji)
    czas_regulacji = str(czas_regulacji + " min")

    przeregulowanie = (max(T) - T_zadane) / T_zadane * 100
    przeregulowanie = round_num(przeregulowanie)
    przeregulowanie = str(przeregulowanie + " %")

    uchyb_ustalony = e[-1]
    uchyb_ustalony = round_num(uchyb_ustalony)

    dokladnosc_regulacji = Tp * sum(map(abs, e))
    dokladnosc_regulacji = round_num(dokladnosc_regulacji)

    koszty_regulacji = Tp * sum(map(abs, u))
    koszty_regulacji = round_num(koszty_regulacji)

    # skalowanie
    n = [float(x * Tp / 60) for x in range(rozmiar)]
    Tmax_list = [float(T_zadane) for _ in range(rozmiar)]
    U_koncowe = sqrt(
        (T_zadane - T_wody_wplywajacej_do_boilera) * (doplyw_wody * gestosc_cieczy * cieplo_wlasciwe * opor))
    Umax_list = [float(U_koncowe) for _ in range(rozmiar)]

    plot = make_subplots(rows=4, cols=1, subplot_titles=(
        "Zależność temperatury od czasu - T(t)", "Zależność sygnału sterującego (wielkości sterującej) od czasu - u(t)",
        "Zależność napięcia na grzałce (wielkości sterowanej) od czasu - U(t)", "Zależność mocy grzałki od czasu - P(t)"))

    plot.add_trace(go.Scatter(x=n, y=T, name="T"), row=1, col=1)
    plot.add_trace(go.Scatter(x=n, y=Tmax_list, name="T zadane", line=dict(dash='dash')), row=1, col=1)
    plot.update_xaxes(title_text="t [min]", row=1, col=1)
    plot.update_yaxes(title_text="T [°C]", range=[0, 85], row=1, col=1)

    plot.add_trace(go.Scatter(x=n, y=u, name="u"), row=2, col=1)
    plot.update_xaxes(title_text="t [min]", row=2, col=1)
    plot.update_yaxes(title_text="u [V]", range=[0, 12], row=2, col=1)

    plot.add_trace(go.Scatter(x=n, y=U, name="U"), row=3, col=1)
    plot.add_trace(go.Scatter(x=n, y=Umax_list, name="Umax"), row=3, col=1)
    plot.update_xaxes(title_text="t [min]", row=3, col=1)
    plot.update_yaxes(title_text="U [V]", range=[0, 250], row=3, col=1)

    plot.add_trace(go.Scatter(x=n, y=P, name="P"), row=4, col=1)
    plot.update_xaxes(title_text="t [min]", row=4, col=1)
    plot.update_yaxes(title_text="P [W]", range=[0, 7500], row=4, col=1)

    return plot, is_open, czas_regulacji, przeregulowanie, uchyb_ustalony, dokladnosc_regulacji, koszty_regulacji

@app.callback(
    Output('the_graph1', 'figure'),
    Output('alert-auto1', 'is_open'),
    Output('czas1', 'children'),
    Output('przeregulowanie1', 'children'),
    Output('uchyb1', 'children'),
    Output('dokladnosc1', 'children'),
    Output('koszty1', 'children'),
    [Input('submit-button-state1', 'n_clicks')],
    [State('alert-auto1', 'is_open'),
    State('T1', 'value'),
    State('P1', 'value'),
    State('V1', 'value'),
    State('Tw1', 'value'),
    State('dw1', 'value'),
    State('kp1', 'value'),
    State('Ti1', 'value'),
    State('Td1', 'value'),
    State('Umax1', 'value'),
    State('t1', 'value'),
    State('Tp1', 'value')])

def update_output(n_clicks, is_open, selected_T, selected_P, selected_V, selected_Tw, selected_dw, selected_kp, selected_Ti, selected_Td, selected_Umax, selected_t, selected_Tp):

    if n_clicks == 0:
        return no_update, is_open, no_update, no_update, no_update, no_update, no_update
    if (selected_T is None) or (selected_P is None) or (selected_V is None) or (selected_Tw is None) or (selected_dw is None) or (selected_kp is None) or (selected_Ti is None) or (selected_Td is None) or (selected_Umax is None) or (selected_t is None) or (selected_Tp is None):
        return no_update, not is_open, no_update, no_update, no_update, no_update, no_update

    T_wody_wplywajacej_do_boilera = selected_Tw
    doplyw_wody = selected_dw/60000
    gestosc_cieczy = 1000
    cieplo_wlasciwe = 4190
    U_minimalne = 0
    U_maksymalne = selected_Umax

    V_boilera = selected_V / 1000  # [m3]  0.005<V<0.02
    opor = U_maksymalne**2/(1000*selected_P)  # 0 < R < 50
    T_zadane = selected_T

    T_maksymalne = 80

    kp = selected_kp
    Ti = selected_Ti
    Td = selected_Td
    Tp = selected_Tp
    czas_symulacji = selected_t*60
    n = int(czas_symulacji / Tp)
    rozmiar = n + 1

    T: List[float] = [T_wody_wplywajacej_do_boilera]
    U: List[float] = [U_minimalne]
    P: List[float] = [U_minimalne * U_minimalne / opor]

    # regulator PID
    e: List[float] = [0]
    u_min: float = 0
    u_max: float = 10
    u: List[float] = [0]
    u_unlimited: List[float] = [0]

    # model regulacyjny
    a = (U_maksymalne - U_minimalne) / (u_max - u_min)
    b = U_minimalne - u_min * a

    for x in range(n):
        e.append(T_zadane - T[-1])
        u_unlimited.append((kp * (e[-1] + (Tp / Ti) * sum(e) + (Td / Tp) * (e[-1] - e[-2]))))
        u.append(max(u_min, min(u_max, u_unlimited[-1])))
        U.append(max(U_minimalne, min(U_maksymalne, a * u[-1] + b)))
        P.append(U[-1] * U[-1] / opor)
        T.append(max(T_wody_wplywajacej_do_boilera, min(T_maksymalne, (Tp / (V_boilera * gestosc_cieczy * cieplo_wlasciwe)) * (doplyw_wody * gestosc_cieczy * cieplo_wlasciwe * (T_wody_wplywajacej_do_boilera - T[-1]) + (U[-1] * U[-1] / opor)) + T[-1])))

    for x in range(n, 1, -1):
        if (T[x] <= T_zadane * 0.95 or T[x] >= T_zadane * 1.05):
            czas_regulacji = x
            break

    czas_regulacji = czas_regulacji * Tp / 60
    print("czas regulacji ", czas_regulacji)
    czas_regulacji = round_num(czas_regulacji)
    czas_regulacji = str(czas_regulacji + " min")

    przeregulowanie = (max(T) - T_zadane) / T_zadane * 100
    print("przeregulowanie", przeregulowanie, "%")
    przeregulowanie = round_num(przeregulowanie)
    przeregulowanie = str(przeregulowanie + " %")

    uchyb_ustalony = e[-1]
    print("Uchyb ustalony", uchyb_ustalony)
    uchyb_ustalony = round_num(uchyb_ustalony)

    dokladnosc_regulacji = Tp * sum(map(abs, e))
    print("Wskaznik dokladnosci regulacji", dokladnosc_regulacji)
    dokladnosc_regulacji = round_num(dokladnosc_regulacji)

    koszty_regulacji = Tp * sum(map(abs, u))
    print("Wskaznik kosztów regulacji", koszty_regulacji)
    koszty_regulacji = round_num(koszty_regulacji)

    # skalowanie
    n = [float(x * Tp / 60) for x in range(rozmiar)]
    Tmax_list = [float(T_zadane) for _ in range(rozmiar)]
    U_koncowe = sqrt(
        (T_zadane - T_wody_wplywajacej_do_boilera) * (doplyw_wody * gestosc_cieczy * cieplo_wlasciwe * opor))
    Umax_list = [float(U_koncowe) for _ in range(rozmiar)]

    plot = make_subplots(rows=4, cols=1, subplot_titles=(
        "Zależność temperatury od czasu - T(t)", "Zależność sygnału sterującego (wielkości sterującej) od czasu - u(t)",
        "Zależność napięcia na grzałce (wielkości sterowanej) od czasu - U(t)", "Zależność mocy grzałki od czasu - P(t)"))

    plot.add_trace(go.Scatter(x=n, y=T, name="T"), row=1, col=1)
    plot.add_trace(go.Scatter(x=n, y=Tmax_list, name="T max", line=dict(dash='dash')), row=1, col=1)
    plot.update_xaxes(title_text="t [min]", row=1, col=1)
    plot.update_yaxes(title_text="T [°C]", range=[0, 85], row=1, col=1)

    plot.add_trace(go.Scatter(x=n, y=u, name="u"), row=2, col=1)
    plot.update_xaxes(title_text="t [min]", row=2, col=1)
    plot.update_yaxes(title_text="u [V]", range=[0, 12], row=2, col=1)

    plot.add_trace(go.Scatter(x=n, y=U, name="U"), row=3, col=1)
    plot.add_trace(go.Scatter(x=n, y=Umax_list, name="Umax"), row=3, col=1)
    plot.update_xaxes(title_text="t [min]", row=3, col=1)
    plot.update_yaxes(title_text="U [V]", range=[0, 250], row=3, col=1)

    plot.add_trace(go.Scatter(x=n, y=P, name="P"), row=4, col=1)
    plot.update_xaxes(title_text="t [min]", row=4, col=1)
    plot.update_yaxes(title_text="P [W]", range=[0, 7500], row=4, col=1)

    return plot, is_open, czas_regulacji, przeregulowanie, uchyb_ustalony, dokladnosc_regulacji, koszty_regulacji

if __name__ == '__main__':
    app.run_server(debug=True)
