import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
import dash
from dash import dcc, html
import geopandas as gpd 
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:counties = json.load(response)

# Basic Line Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Line Plot')
plt.show()

# Basic Seaborn Plot: Scatter plot with regression line
tips = sns.load_dataset('tips')

sns.lmplot(x='total_bill', y='tip', data=tips)
plt.show()

# Line Plot with Plotly
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
fig.update_layout(title='Line Plot', xaxis_title='x', yaxis_title='sin(x)')
fig.show()

# Scatter plot with Plotly Express
df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
fig.show()

# Example confusion matrix
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Example Random Forest and Partial Dependence Plot
model = RandomForestClassifier()
model.fit(X_train, y_train)

plot_partial_dependence(model, X_train, [0, 1])
plt.show()

#Creating Dashboards with Plotly
app = dash.Dash(__name__)
df = px.data.iris()

# Dash layout
app.layout = html.Div([
    dcc.Graph(
        id='example-graph',
        figure=px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    )
])

app.run_server(debug=True)

# 3D Scatter Plot
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show()

#Maps and Geospatial data
world_data = gpd.read_file(r'world.shp') 
world_data
world_data.plot() 
world_data = gpd.read_file(r'world.shp') 
world_data = world_data[['NAME', 'geometry']]
world_data['area'] = world_data.area 
world_data = world_data[world_data['NAME'] != 'Antarctica'] 
world_data[world_data.NAME=="India"].plot()
current_crs = world_data.crs 
world_data.to_crs(epsg=3857, inplace=True) 
world_data.plot(column='NAME', cmap='hsv')

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})
fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           scope="usa",
                           labels={'unemp':'unemployment rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

df = px.data.election()
geojson = px.data.election_geojson()

fig = px.choropleth(df, geojson=geojson, color="Bergeron",
                    locations="district", featureidkey="properties.district",
                    projection="mercator"
                   )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()