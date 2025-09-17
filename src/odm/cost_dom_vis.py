import pandas as pd
import plotly.graph_objects as go
import numpy as np

cost_threshold = pd.read_csv('cost_threshold.csv')
odm_journeys = pd.read_csv('odm_journeys.csv')
pt_journeys = pd.read_csv('pt_journeys.csv')

cost_threshold['time'] = pd.to_datetime(cost_threshold['time'], utc=True).dt.tz_convert('Europe/Berlin')
pt_journeys['departure'] = pd.to_datetime(pt_journeys['departure'], utc=True).dt.tz_convert('Europe/Berlin')
pt_journeys['center'] = pd.to_datetime(pt_journeys['center'], utc=True).dt.tz_convert('Europe/Berlin')
pt_journeys['arrival'] = pd.to_datetime(pt_journeys['arrival'], utc=True).dt.tz_convert('Europe/Berlin')
odm_journeys['departure'] = pd.to_datetime(odm_journeys['departure'], utc=True).dt.tz_convert('Europe/Berlin')
odm_journeys['center'] = pd.to_datetime(odm_journeys['center'], utc=True).dt.tz_convert('Europe/Berlin')
odm_journeys['arrival'] = pd.to_datetime(odm_journeys['arrival'], utc=True).dt.tz_convert('Europe/Berlin')

fig = go.Figure()

# threshold
fig.add_trace(go.Scatter(x=cost_threshold["time"], y=cost_threshold["cost"], name="Cost Threshold",
                         line=dict(color="gray")))

# pt
fig.add_trace(
    go.Scatter(x=pt_journeys["center"], y=pt_journeys["cost"], name="PT", mode="markers", marker=dict(color="blue"),
               customdata=np.stack(
                   (pt_journeys["departure"], pt_journeys["arrival"], pt_journeys["travel_time"],
                    pt_journeys["transfers"],
                    pt_journeys["cost"]), axis=-1),
               hovertemplate="<b>Departure</b>: %{customdata[0]}<br>" + "<b>Arrival</b>: %{customdata[1]}<br>" + "<b>Travel time</b>: %{customdata[2]}<br>" + "<b>Transfers</b>: %{customdata[3]}<br>" + "<b>Cost</b>: %{customdata[4]}"))
fig.add_trace(go.Scatter(x=pt_journeys["departure"], y=pt_journeys["cost"], mode="markers",
                         marker=dict(color="blue", symbol=142), hoverinfo='none', showlegend=False))
fig.add_trace(go.Scatter(x=pt_journeys["arrival"], y=pt_journeys["cost"], mode="markers",
                         marker=dict(color="blue", symbol=142), hoverinfo='none', showlegend=False))
for j in pt_journeys.itertuples():
    fig.add_shape(type="line", x0=j.departure, y0=j.cost, x1=j.arrival, y1=j.cost, line=dict(width=1, color="blue"))

# odm
fig.add_trace(go.Scatter(x=odm_journeys["center"], y=odm_journeys["cost"], name="ODM", mode="markers",
                         marker=dict(color="orange"), customdata=np.stack(
        (odm_journeys["departure"], odm_journeys["arrival"], odm_journeys["travel_time"], odm_journeys["transfers"],
         odm_journeys["odm_time"], odm_journeys["cost"]), axis=-1),
                         hovertemplate="<b>Departure</b>: %{customdata[0]}<br>" + "<b>Arrival</b>: %{customdata[1]}<br>" + "<b>Travel time</b>: %{customdata[2]}<br>" + "<b>Transfers</b>: %{customdata[3]}<br>" + "<b>ODM Time</b>: %{customdata[4]}<br>" + "<b>Cost</b>: %{customdata[5]}"))
fig.add_trace(go.Scatter(x=odm_journeys["departure"], y=odm_journeys["cost"], mode="markers",
                         marker=dict(color="orange", symbol=142), hoverinfo='none', showlegend=False))
fig.add_trace(go.Scatter(x=odm_journeys["arrival"], y=odm_journeys["cost"], mode="markers",
                         marker=dict(color="orange", symbol=142), hoverinfo='none', showlegend=False))
for j in odm_journeys.itertuples():
    fig.add_shape(type="line", x0=j.departure, y0=j.cost, x1=j.arrival, y1=j.cost, line=dict(width=1, color="orange"))

fig.show()
