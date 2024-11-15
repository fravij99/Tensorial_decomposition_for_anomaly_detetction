import detectorlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd


"""window=16 # Window represents the sensors
det=detectorlib.detector()
det.xlsx_path="Anomalies_hierarchy_exp_distribution"
randexp=[]
for i in range (144):
    randexp.append(det.random_exp_generator(1800))
for i in range(15):
    randexp.append(det.random_big_anomalies_generation(1800)) 
randexp.append(det.random_complete_anomalies_generation(1800)) 
randexp=np.array(randexp)
det.df=randexp



possible_shapes=[
([10, 16], [1800]), 
([16], [1800, 10]),	
([10], [1800, 16]),	
]

possible_shapes_pruned=[
([9, 16], [1800]), 
([16], [1800, 9]),	
([9], [1800, 16]),	
]

possible_shapes_antipruned=[([16], [1800])]


det.create_statistical_model('PCA')
det.stamp_all_shape_anomalies(possible_shapes)

det.df=randexp
print(det.pruning(window).shape, det.df)
det.stamp_all_shape_anomalies(possible_shapes_pruned)


det.df=randexp
print(det.antipruning(window).shape)

det.stamp_all_shape_anomalies(possible_shapes_antipruned)"""

"""for i in tqdm(range(3), desc='Generating tensors...'):
    decision=np.random.choice([1, 2])
    data=np.random.normal(loc=0.0, scale=1.0, size=(X, Y, Z))
    data=det.introduce_anomalies(data, num_anomalies=6, scale_range=[7, 15], index=i)
    if decision==1:
        print(f'Big anomaly at index:{i}')
        data=det.introduce_anomalies(data, num_anomalies=1, scale_range=[59, 60], index=i)
    data_infected.append(data)
    det.hitmaps(X, Y, Z, data, i)"""


X, Y, Z = 200, 201, 202  # Dimensioni del tensor
det=detectorlib.detector()

# Aggiungi anomalie di scala più grande
data_infected=[]

# Aggiungi piccole anomalie
data=np.random.normal(loc=0.0, scale=1.0, size=(X, Y, Z))

x_anomalies, y_anomalies, z_anomalies=int(X*0.007), int(Y*0.007), int(Z*0.007)

data = det.introduce_anomalies_2D(data, 'x', x_anomalies)
data = det.introduce_anomalies_2D(data, 'y', y_anomalies)
data = det.introduce_anomalies_2D(data, 'z', z_anomalies)

data = det.introduce_anomalies_1D(data, 'x', x_anomalies)
data = det.introduce_anomalies_1D(data, 'y', y_anomalies)
data = det.introduce_anomalies_1D(data, 'z', z_anomalies)




'''data = det.introduce_anomalies_1D(data, 'x', x_anomalies)
data = det.introduce_anomalies_1D(data, 'y', y_anomalies)
data = det.introduce_anomalies_1D(data, 'z', z_anomalies)'''

'''data=det.introduce_anomalies_3D(data, num_anomalies=6, scale_range=[2, 4], index=1)
data=det.introduce_anomalies_3D(data, num_anomalies=6, scale_range=[5, 7], index=1)'''



det.hitmaps(X, Y, Z, data, 0)

possible_shapes2D=[
    ([200], [201, 202]),
    ([201], [200, 202]),
    ([202], [201, 200])
]

possible_shapes1D=[
    ([40, 30], [20]),
    ([30, 20], [40]),
    ([20, 40], [30])
]
det.df=np.array(data)
det.xlsx_path='anomalies_cubes.xlsx'
possible_models={'PCA', 'SVM', 'IsolationForest'}
for model in (possible_models):
    det.stamp_all_shape_anomalies(possible_shapes2D, model)