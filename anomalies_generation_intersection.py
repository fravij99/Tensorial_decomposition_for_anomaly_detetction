import numpy as np
from detectorlib import detector, generator

# Dimensioni del tensore
X, Y, Z = 200, 201, 202
det = detector()
gen = generator()

data = np.random.normal(loc=0.0, scale=1.0, size=(X, Y, Z))
det.df = data  # Assegna i dati al rilevatore

# Amount of anomalies
x_anomalies, y_anomalies, z_anomalies=int(2*X*0.007), int(2*Y*0.007), int(2*Z*0.007)

data = gen.introduce_anomalies_2D(data, 'x', x_anomalies)
data = gen.introduce_anomalies_2D(data, 'y', y_anomalies)
data = gen.introduce_anomalies_2D(data, 'z', z_anomalies)

data = gen.introduce_anomalies_1D(data, 'x', x_anomalies)
data = gen.introduce_anomalies_1D(data, 'y', y_anomalies)
data = gen.introduce_anomalies_1D(data, 'z', z_anomalies)

gen.hitmaps(X, Y, Z, data, 0)
views=[X, Y, Z]

# It works also with a cubic tensor N x N x N
det.stamp_all_views_anomalies(data, 'IsolationForest', [X], [Y, Z], (0, 1, 2))
det.stamp_all_views_anomalies(data, 'IsolationForest', [Y], [X, Z], (2, 0, 1))
det.stamp_all_views_anomalies(data, 'IsolationForest', [Z], [Y, X], (1, 2, 0))

det.create_anomalies_vectors_fromfile(views)

det.find_anomalies_intersection_from_files(views, 'X')
det.find_anomalies_intersection_from_files(views, 'Y')
det.find_anomalies_intersection_from_files(views, 'Z')

det.stamp_all_views_anomalies(data, 'IsolationForest', [X, Y], [Z], (0, 1, 2))
det.stamp_all_views_anomalies(data, 'IsolationForest', [Y, Z], [X], (2, 0, 1))
det.stamp_all_views_anomalies(data, 'IsolationForest', [Z, X], [Y], (1, 2, 0))