import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# CSV-Daten laden
cost_threshold = pd.read_csv('cost_threshold.csv')
odm_journeys = pd.read_csv('odm_journeys.csv')
pt_journeys = pd.read_csv('pt_journeys.csv')

# Zeit-Spalten zu datetime konvertieren
cost_threshold['time'] = pd.to_datetime(cost_threshold['time'], utc=True)
odm_journeys['departure'] = pd.to_datetime(odm_journeys['departure'], utc=True)
odm_journeys['center'] = pd.to_datetime(odm_journeys['center'], utc=True)
odm_journeys['arrival'] = pd.to_datetime(odm_journeys['arrival'], utc=True)
pt_journeys['departure'] = pd.to_datetime(pt_journeys['departure'], utc=True)
pt_journeys['center'] = pd.to_datetime(pt_journeys['center'], utc=True)
pt_journeys['arrival'] = pd.to_datetime(pt_journeys['arrival'], utc=True)

# Plot erstellen
fig, ax = plt.subplots(figsize=(15, 10))

# 1. Cost Threshold als Linie
ax.plot(cost_threshold['time'], cost_threshold['cost'],
        'b-', linewidth=2, label='Cost Threshold', alpha=0.8)

# 2. ODM Journeys als horizontale Linien mit vertikalen Strichen
for i, row in odm_journeys.iterrows():
    # Horizontale Linie von departure bis arrival
    ax.plot([row['departure'], row['arrival']], [row['cost'], row['cost']],
            'r-', linewidth=3, alpha=0.7)

    # Vertikale Striche für departure und arrival (±10 units)
    ax.plot([row['departure'], row['departure']], [row['cost']-10, row['cost']+10],
            'r-', linewidth=3, alpha=0.8)
    ax.plot([row['arrival'], row['arrival']], [row['cost']-10, row['cost']+10],
            'r-', linewidth=3, alpha=0.8)
    # Quadrat für center
    ax.plot(row['center'], row['cost'], 'rs', markersize=8, alpha=0.8)

# 3. PT Journeys als horizontale Linien mit vertikalen Strichen
for i, row in pt_journeys.iterrows():
    # Horizontale Linie von departure bis arrival
    ax.plot([row['departure'], row['arrival']], [row['cost'], row['cost']],
            'g-', linewidth=3, alpha=0.7)

    # Vertikale Striche für departure und arrival (±10 units)
    ax.plot([row['departure'], row['departure']], [row['cost']-10, row['cost']+10],
            'g-', linewidth=3, alpha=0.8)
    ax.plot([row['arrival'], row['arrival']], [row['cost']-10, row['cost']+10],
            'g-', linewidth=3, alpha=0.8)
    # Quadrat für center
    ax.plot(row['center'], row['cost'], 'gs', markersize=8, alpha=0.8)

# Achsen formatieren
ax.set_xlabel('Zeit', fontsize=12)
ax.set_ylabel('Cost', fontsize=12)
ax.set_title('Journey Cost Visualization', fontsize=14, fontweight='bold')

# X-Achse Zeit-Formatierung
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='Europe/Berlin'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Legende erstellen
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='b', linewidth=2, label='Cost Threshold'),
    Line2D([0], [0], color='r', linewidth=3, label='ODM Journeys'),
    Line2D([0], [0], color='g', linewidth=3, label='PT Journeys'),
    Line2D([0], [0], color='k', linewidth=3, label='Departure/Arrival (vertikale Striche)'),
    Line2D([0], [0], marker='s', color='k', linestyle='None', markersize=8, label='Center')
]
ax.legend(handles=legend_elements, loc='upper right')

# Grid hinzufügen
ax.grid(True, alpha=0.3)

# Layout anpassen
plt.tight_layout()

# Plot anzeigen
plt.show()

# Zusätzliche Statistiken ausgeben
print("=== Datenübersicht ===")
print(f"Cost Threshold Bereich: {cost_threshold['cost'].min():.2f} - {cost_threshold['cost'].max():.2f}")
print(f"ODM Journeys Kosten: {odm_journeys['cost'].min()} - {odm_journeys['cost'].max()}")
print(f"PT Journeys Kosten: {pt_journeys['cost'].min()} - {pt_journeys['cost'].max()}")
print(f"Anzahl ODM Journeys: {len(odm_journeys)}")
print(f"Anzahl PT Journeys: {len(pt_journeys)}")