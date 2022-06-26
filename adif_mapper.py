# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:48:20 2022
This is a beta version and is still being developed.

App to open a adif file downloaded from pskreporter.com
Save your downloaded file to the data directory as input.ddif
The app will strip the data into a pandas dataframe, lookup coordinates
for the maidenhead grid and create LL columns,
It then creates a geopandas geoframe and saves it to a shapefile or geopackage.

The program then runs as a dash app and maps the data with sliders to filter the data.

@author: brian panetta K3YZF
"""

import pandas as pd
import adif_io
import maidenhead as mh
from shapely.geometry import Point
import geopandas
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px
import datetime
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Output,Input
from dash import State
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)

#/////////////////////////////////////////////////////////////////////////////
#Functions

 
            
def cat_freq(row):  
   if row['FREQ'] >= 50 and row['FREQ'] <= 54:
       return '6'
   elif row['FREQ'] >= 28 and row['FREQ'] <= 29.7:
       return '10'
   elif row['FREQ'] >= 24.89  and row['FREQ'] <= 24.99:
       return '12'
   elif row['FREQ'] >= 21  and row['FREQ'] <= 21.45:
       return '15'
   elif row['FREQ'] >= 18.068  and row['FREQ'] <= 18.168:
       return '17'
   elif row['FREQ'] >= 14  and row['FREQ'] <= 14.350:
       return '20'
   elif row['FREQ'] >= 10.1  and row['FREQ'] <= 10.15:
       return '30'
   elif row['FREQ'] >= 7  and row['FREQ'] <= 7.3:
       return '40'
   elif row['FREQ'] >= 3.5  and row['FREQ'] <= 4:
       return '80'
   elif row['FREQ'] >= 1.8  and row['FREQ'] <= 2:
       return '160'
   return 'other'
              





#/////////////////////////////////////////////////////////////////////////////
#Nav Bar

navbar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        
    ],
    brand="QSOS Mapper and Analysis - Written by K3YZF",
    brand_href="#",
    color="info",
    # dark=True,
)

#/////////////////////////////////////////////////////////////////////////////
# Input File

qsos_raw, adfif_header = adif_io.read_from_file('data/input.adif')


qsos = pd.DataFrame(qsos_raw)
# print(qsos)

mhLL_l = []


for index, row in qsos.iterrows():
    mhg = row['MY_GRIDSQUARE']
    mhs = mhg[0 : 6]
    mhLL = mh.to_location(mhs)
    mhLL_l.append(mhLL)

LAT, LONG = zip(*mhLL_l)

qsos['LAT'] = LAT
qsos['LONG'] =LONG


qsos['Year'] = qsos.QSO_DATE.str[0:4]
qsos['Month'] = qsos.QSO_DATE.str[4:6]
qsos['Day'] = qsos.QSO_DATE.str[6:8]
qsos['Hour'] = qsos.TIME_ON.str[0:2]
qsos['Minute'] = qsos.TIME_ON.str[2:4]
qsos['Second'] = qsos.TIME_ON.str[4:6]

qsos[['Year','Month','Day','Hour','Minute','Second']] = \
    qsos[['Year','Month','Day','Hour','Minute','Second']].astype(int)


qsos['DateTime'] = pd.to_datetime(qsos[['Month','Day','Year','Hour','Minute','Second']]) 

# Set the datatime column as the new index column
qsos.index = pd.to_datetime(qsos[['Month','Day','Year','Hour','Minute','Second']])     

#Remove unwanted Columns
# qsos.drop(['Year','Month','Day','Hour','Minute','Second','CALL','COUNTRY'], axis = 1,inplace = True)
# qsos.drop(['Second'], axis = 1, inplace = True)
# print(qsos)

qsos['APP_PSKREP_SNR'].fillna(0,inplace=True)


# qsos['FREQ'] = qsos['FREQ'].astype(float)
qsos = qsos.astype({'FREQ':float, 'APP_PSKREP_BRG':float,'DISTANCE':float,
                    'MODE':str,'OPERATOR':str,'MY_GRIDSQUARE':str,
                    'APP_PSKREP_SNR':float,})
# print(qsos.dtypes)


#Clean the dataframe of error values
indexNames = qsos[qsos['FREQ'] > 55].index
qsos.drop(indexNames, inplace=True)

indexNames = qsos[qsos['FREQ'] < 1].index
qsos.drop(indexNames, inplace=True)

scaler = MinMaxScaler()
qsos[['Scaled_SNR','Scaled_Dist']] = scaler.fit_transform(qsos[['APP_PSKREP_SNR','DISTANCE']])
    
# qsos['SNR_ABS'] = abs(qsos['APP_PSKREP_SNR']) 

qsos['FREQ_str'] = qsos.apply(lambda row: cat_freq(row), axis=1)



if '6' in qsos.FREQ_str.values:
    status6 = False
else:
    status6 = True
if '10' in qsos.FREQ_str.values:
    status10 = False
else:
    status10 = True
if '12' in qsos.FREQ_str.values:
    status12 = False
else:
    status12 = True
if '15' in qsos.FREQ_str.values:
    status15 = False
else:
    status15 = True
if '17' in qsos.FREQ_str.values:
    status17 = False
else:
    status17 = True
if '20' in qsos.FREQ_str.values:
    status20 = False
else:
    status20 = True
if '30' in qsos.FREQ_str.values:
    status30 = False
else:
    status30 = True
if '40' in qsos.FREQ_str.values:
    status40 = False
else:
    status40 = True
if '80' in qsos.FREQ_str.values:
    status80 = False
else:
    status80 = True
if '160' in qsos.FREQ_str.values:
    status160 = False
else:
    status160 = True
    
    
# print('print status 20',status20)   
# print(qsos.FREQ_str)

indexNames = qsos[qsos['FREQ_str'] == 'other'].index
qsos.drop(indexNames, inplace=True)

qsos['BAND'] = qsos['FREQ_str'].astype(int)

gdf = geopandas.GeoDataFrame(qsos,geometry=geopandas.points_from_xy(qsos.LONG,qsos.LAT))
# print(gdf)
print(gdf.dtypes)

mintime = gdf.index.min()
maxtime = gdf.index.max()
# print(mintime)

minDist = gdf.DISTANCE.min()
maxDist = gdf.DISTANCE.max()

minSNR = gdf.APP_PSKREP_SNR.min()
maxSNR = gdf.APP_PSKREP_SNR.max()


# gdf.to_csv('test.csv')



#/////////////////////////////////////////////////////////////////////////////
#App


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX],
                # meta_tags=[{'name': 'viewport',
                #             'content': 'width=device-width, initial-scale=1.0'}]
                )



server = app.server

app.layout = dbc.Container([
    dbc.Row([
            navbar
        ]),
    
    dbc.Row([
        dbc.Col([

            dcc.Graph(id='xplot1',figure={}),
            dcc.Graph(id='xplot2',figure={}),
            dcc.Graph(id='xplot3',figure={}),
            
        ],width=3),
        dbc.Col([
            # dcc.Graph(id='Map',figure=fig,style={'width': '95vh', 'height': '90vh'}),
            dcc.Graph(id='map',figure={}),
            # dcc.RangeSlider(0,15,1,value=[5,7],id='rangeslider'),
            dbc.Label('Distance (miles)'),
            dcc.RangeSlider(0,maxDist,1000,value=[0,maxDist],id='rangeslider'),
            dbc.Label('SNR'),
            dcc.RangeSlider(minSNR,maxSNR,2,value=[minSNR,maxSNR],id='rangeslider_snr'),
            # dbc.Label('Hour'),
            # dcc.RangeSlider(0,24,1,value=[0,24],id='rangeslider_hour'),
            dbc.Label('Band'),
            # dbc.Checklist(
            #     options=[
            #         {'label':'6m', 'value':'6','disabled': True},
            #         {'label':'10m', 'value':'10','disabled': False},
            #         {'label':'12m', 'value':'12','disabled': False},
            #         {'label':'15m', 'value':'15','disabled': False},
            #         {'label':'17m', 'value':'17','disabled': False},
            #         {'label':'20m', 'value':'20', 'disabled': False},
            #         {'label':'30m', 'value':'30','disabled': False},
            #         {'label':'40m', 'value':'40','disabled': False},
            #         {'label':'80m', 'value':'80','disabled': False},
            #         {'label':'160m', 'value':'160','disabled': False},
            #         ],
            dbc.Checklist(
                options=[
                    {'label':'6m', 'value':'6','disabled': status6},
                    {'label':'10m', 'value':'10','disabled': status10},
                    {'label':'12m', 'value':'12','disabled': status12},
                    {'label':'15m', 'value':'15','disabled': status15},
                    {'label':'17m', 'value':'17','disabled': status17},
                    {'label':'20m', 'value':'20', 'disabled': status20},
                    {'label':'30m', 'value':'30','disabled': status30},
                    {'label':'40m', 'value':'40','disabled': status40},
                    {'label':'80m', 'value':'80','disabled': status80},
                    {'label':'160m', 'value':'160','disabled': status160},
                    ],
                    value=['6','10','12','15','17','20','30','40','80','160'],
                    id="fswitches",
                    switch=True, inline=True),
            
           
            ],width=9,
            # style={'background-color': 'lightgrey'},
            ),
    ],className='g-0',),
])
        
            
      

#/////////////////////////////////////////////////////////////////////////////
#Callbacks
            
            
@app.callback(
    Output('map', 'figure'),
    Output('xplot1', 'figure'),
    Output('xplot2', 'figure'),
    Output('xplot3', 'figure'),
    Input('rangeslider', 'value'),
    Input('rangeslider_snr', 'value'),
    Input('fswitches', 'value',)
)

def filter_gdf(value,value2,value3):
    lowerF = value[0]
    upperF = value[1]
    lowerSNR = value2[0]
    upperSNR = value2[1]
    # lowerHour = value3[0]
    # upperHour = value3[1]
    band = value3
    # print(band)
    
    # gdff = gdf[(gdf.DISTANCE > lowerF)]
    gdft = gdf[(gdf.DISTANCE > lowerF) & (gdf.DISTANCE < upperF)]
    series = gdft.FREQ_str.isin(band)
    gdft2 = gdft[series]
    print(gdft2)
    gdff= gdft2[(gdft2.APP_PSKREP_SNR > lowerSNR) & (gdft2.APP_PSKREP_SNR < upperSNR)]
    
    scaler = MinMaxScaler()
    gdff[['Scaled_SNR','Scaled_Dist']] = scaler.fit_transform(gdff[['APP_PSKREP_SNR','DISTANCE']])
    
    # print(gdff)
    
    
    fig = px.scatter_geo(
        gdff,
        lat=gdff.geometry.y,
        lon=gdff.geometry.x,
        # projection='robinson',
        projection='miller',
        # projection='mercator',
        hover_name='APP_PSKREP_SNR',
        size = 'Scaled_SNR',
        color = 'Scaled_SNR',
        color_continuous_scale='jet',
        )
    fig.update_geos(
        visible=True, 
        resolution=50,
        showcountries=True,
        countrycolor='lightgrey',
        landcolor='rgb(255,255,255)',
        coastlinecolor = 'white',
        showcoastlines = True,
        showocean = True,
        oceancolor = 'LightBlue',
        # lonaxis_range = [-15, -5],
        lataxis_range = [-40, 90],
        
        )
    fig.update_layout(
        margin=dict(l=10,r=10,b=10,t=10),
        )
    
    xplot1 = px.scatter(
        gdff,
        y=gdff.BAND,
        x=gdff.DISTANCE,
        # color=gdf.FREQ_str,
        # color_continuous_scale='jet',
        color_discrete_sequence=['rgb(228,26,28)','#109618','#0099C6','#EECA3B','#FF97FF','rgb(136,204,238)']
        
        )
    xplot1.update_layout(
        margin=dict(l=0,r=0,b=10,t=10),
        width = 200,
        height = 200,
        legend_font_size = 8,
        )

    xplot2 = px.scatter(
        gdff,
        y=gdff.APP_PSKREP_SNR,
        x=gdff.DISTANCE,
        # color=gdf.FREQ_str,
        # color_continuous_scale='jet',
        color_discrete_sequence=['rgb(228,26,28)','#109618','#0099C6','#EECA3B','#FF97FF','rgb(136,204,238)']
        
        )
    xplot2.update_layout(
        margin=dict(l=0,r=0,b=10,t=10),
        width = 200,
        height = 200,
        legend_font_size = 8,
        )
    
    
    xplot3 = px.scatter(
        gdff,
        y=gdff.BAND,
        x=gdff.APP_PSKREP_SNR,
        # color=gdf.FREQ_str,
        # color_continuous_scale='jet',
        color_discrete_sequence=['rgb(228,26,28)','#109618','#0099C6','#EECA3B','#FF97FF','rgb(136,204,238)']
        
        )
    # xplot3 = go.Figure([go.Bar(
    #     x = gdff.BAND,
    #     y=gdff.APP_PSKREP_SNR,
    #     )])
    xplot3.update_layout(
        margin=dict(l=0,r=0,b=10,t=10),
        width = 200,
        height = 200,
        legend_font_size = 8,
        )
    return fig, xplot1, xplot2, xplot3



    
    






if __name__ == '__main__':
    app.run_server(debug=False)

