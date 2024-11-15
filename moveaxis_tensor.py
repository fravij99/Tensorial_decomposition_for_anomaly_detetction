import numpy as np
from detectorlib import detector

# Dimensioni del tensore
X, Y, Z = 200, 201, 202
det = detector()

# Genera dati con piccole anomalie
data = np.random.normal(loc=0.0, scale=1.0, size=(X, Y, Z))
det.df = data  # Assegna i dati al rilevatore

x_anomalies, y_anomalies, z_anomalies=int(2*X*0.007), int(2*Y*0.007), int(2*Z*0.007)

data = det.introduce_anomalies_2D(data, 'x', x_anomalies)
data = det.introduce_anomalies_2D(data, 'y', y_anomalies)
data = det.introduce_anomalies_2D(data, 'z', z_anomalies)

views=[X, Y, Z]

det.stamp_all_views_anomalies(views, 'IsolationForest')