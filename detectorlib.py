import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import keras
import random
import keras.layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Reshape, LSTM, GRU, Conv1D, Conv3D, MaxPooling1D, MaxPooling3D, Conv1DTranspose
from sklearn.neighbors import LocalOutlierFactor
from datetime import timedelta
from matplotlib.colors import Normalize
from plot_keras_history import plot_history
import tensorflow as tf
import matplotlib.dates as mdates
import networkx as nx
import numpy as np

'''In this class we have to set the local variables to assign ath every index of our notations'''
'''This '''
class detector():
  
  def tuple_prod(self, tuple):
    prod = 1
    for dim in tuple:
        prod *= dim
    return prod
  

  '''Reads from excel file the data and append the sheets to the third index of the tensor: (temporal samples, features, sensors)'''
  def load_preprocess(self, path, sens_num):
        self.df = []
        self.xlsx_path=path
        for sheet_num in range(sens_num): 
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            self.timestamp = sheet_df['timestamp'].dt.date
            sheet_df = sheet_df.drop(['Unnamed: 0', 'timestamp', 'sensor', 'off_ch1', 'off_ch2', 'off_ch3', 'off_ch4'], axis=1)
            self.df.append(sheet_df)
            
        self.df = np.array(self.df)
        self.df = np.nan_to_num(self.df)
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        self.df = self.df.transpose(1, 0, 2)


  '''This function is need to trat random matrix generated in the main in order to verify herarchical relationships between anomaly classes'''
  def random_matrix_loading(self):

    self.randmatrix = np.random.rand(1000, 40000)
    self.df= self.randmatrix
    possible_shapes=[([self.randmatrix.shape[0]], [int(self.randmatrix.shape[1]/20), 20]), 
                     ([self.randmatrix.shape[0], int(self.randmatrix.shape[1]/20)], [20]),
                     ([self.randmatrix.shape[0], 20], [int(self.randmatrix.shape[1]/20)])]
    possible_models={'PCA'}
    self.xlsx_path='Random_matrix_trial.xlsx'
    for model in tqdm(possible_models, desc="Creating models"):
      self.create_statistical_model(model)
      plt.imshow(self.df, cmap='hot', interpolation='nearest')
      plt.show()

      # Per la rete neurale, anche l'ordine in cui inserisco le dimensioni risulta essere importante
      self.stamp_all_shape_anomalies(possible_shapes)


  def random_walk(self, length, dim=1):
    dir_array = np.random.randint(dim, size=length)
    updown_array = np.random.choice([-1, 1], size=length)

    steps = np.zeros((length, dim))
    steps[np.arange(length), dir_array] = updown_array

    pos = np.cumsum(steps, axis=0).reshape(-1)

    return (pos/(pos.max()))


  '''This function generates some statistical distribution infecting them with noise anomalies on several hierarchic levels and launches PCA to extract from the more grain to the more fine anomaly'''
  def random_anomalies_generation(self, input_size):
    self.randexp=self.random_exp_generator(input_size)
    peak_width = int(0.1 * len(self.randexp))  
    # Random walk generation, filling the trajectory with random zeros to have sporadic random small anomalies
    self.randomwalk = self.random_walk(len(self.randexp), dim=1)
    index_zero=random.sample(range(len(self.randomwalk)), int(len(self.randomwalk)-peak_width))
    self.randomwalk[index_zero]=0

    for i in range(len(self.randexp)):
        self.randexp[i] += self.randomwalk[i]*0.1
    return self.randexp
  
  def random_exp_generator(self, input_size):
    randexp = np.exp(np.arange(0, input_size)/(input_size*0.05))
    randexp = randexp / max(randexp)
    return randexp
  
  
  '''This function generates some statistical distribution infecting them localize gaussian anomalies'''
  def random_big_anomalies_generation(self, input_size):
    self.randexp = self.random_exp_generator(input_size)
    peak_width = int(0.1 * len(self.randexp))  
    self.gauss_err = 0.5*np.exp(-np.power((np.arange(0, peak_width)), 2.)/float(peak_width*100))  
    peak_start = 650

    for i in range(len(self.gauss_err)):
        self.randexp[peak_start + i] += self.gauss_err[i]
    return self.randexp
  

  def random_complete_anomalies_generation(self, input_size):
    self.randexp = np.exp(np.arange(0, input_size)/(input_size*0.05))
    self.randexp = self.randexp / max(self.randexp)
    peak_width = int(0.1 * len(self.randexp))  
    self.gauss_err = 0.5*np.exp(-np.power((np.arange(0, peak_width)), 2.)/float(peak_width*100))  
    peak_start = 650 #np.random.randint(0, len(self.randexp) - peak_width)
    # Random walk generation, filling the trajectory with random zeros to have sporadic random small anomalies
    self.randomwalk = self.random_walk(len(self.randexp), dim=1)
    index_zero=random.sample(range(len(self.randomwalk)), int(len(self.randomwalk)-peak_width))
    self.randomwalk[index_zero]=0

    for i in range(len(self.randexp)):
        self.randexp[i] += self.randomwalk[i]*0.1
    for i in range(len(self.gauss_err)):
        self.randexp[peak_start + i] += self.gauss_err[i]
    return self.randexp

  
  '''This function generates anomalies 1D for every point of view inserted by the user'''
  def introduce_anomalies_1D(self, tensor, coordinate, num_anomalies):
    X, Y, Z = tensor.shape
    if coordinate == 'x':
        for i in tqdm(range(num_anomalies), desc='Infecting with X anomalies 1D...'):
            anomaly = np.random.normal(loc=0.0, scale=100.0, size=(X))
            y, z = np.random.randint(0, Y), np.random.randint(0, Z)
            tensor[:, y, z] += anomaly  
            print(f'X anomaly at (y, z)=({y}, {z})')
    elif coordinate == 'y':
        for i in tqdm(range(num_anomalies), desc='Infecting with Y anomalies 1D...'):
            anomaly = np.random.normal(loc=0.0, scale=100.0, size=(Y))
            x, z = np.random.randint(0, X), np.random.randint(0, Z)
            tensor[x, :, z] += anomaly 
            #print(f'Y anomaly at (x, z)=({x}, {z})')
    elif coordinate == 'z':
        for i in tqdm(range(num_anomalies), desc='Infecting with Z anomalies 1D...'):
            anomaly = np.random.normal(loc=0.0, scale=100.0, size=(Z))
            x, y = np.random.randint(0, X), np.random.randint(0, Y)
            tensor[x, y, :] += anomaly  
            #print(f'Z anomaly at (x, y)=({x}, {y})')
    return tensor

  '''This function generates anomalies 2D for every point of view inserted by the user'''
  def introduce_anomalies_2D(self, tensor, coordinate, num_anomalies):
    X, Y, Z = tensor.shape
    
    if coordinate =='x':
      for i in tqdm(range(num_anomalies), desc='Infecting Tensor with 2D anomalies YZ...'):
        anomaly=np.random.normal(loc=50, scale=1, size=(Y, Z))
        anomaly_index=np.random.randint(0, X)
        tensor[anomaly_index, :, :] += anomaly
        print('X index= ', anomaly_index)
    elif coordinate =='y':
      for i in tqdm(range(num_anomalies), desc='Infecting Tensor with 2D anomalies XZ...'):
        anomaly=np.random.normal(loc=50, scale=1, size=(X, Z))
        anomaly_index=np.random.randint(0, Y)
        tensor[:, anomaly_index, :] += anomaly
        print('Y index= ', anomaly_index)
    elif coordinate =='z':
      for i in tqdm(range(num_anomalies), desc='Infecting Tensor with 2D anomalies XY...'):
        anomaly=np.random.normal(loc=50, scale=1, size=(X, Y))
        anomaly_index=np.random.randint(0, Z)
        tensor[:, :, anomaly_index] += anomaly
        print('Z index= ', anomaly_index)
    return tensor

  """This function utilizes a random-distribuited 3D dataset and infects it with anomalies of different scales: 
     the scale can be set as a parameter of the function"""
  def introduce_anomalies_3D(self, tensor, num_anomalies, scale_range, index):
    X, Y, Z = tensor.shape
    for _ in tqdm(range(num_anomalies), desc="Infecting the tensor..."):
        # Scegli la scala dell'anomalia casualmente all'interno di un range dato
        scale_x = np.random.randint(scale_range[0], scale_range[1])
        scale_y = np.random.randint(scale_range[0], scale_range[1])
        scale_z = np.random.randint(scale_range[0], scale_range[1])
        
        # Scegli una posizione casuale all'interno del tensore
        start_x = np.random.randint(0, X - scale_x)
        start_y = np.random.randint(0, Y - scale_y)
        start_z = np.random.randint(0, Z - scale_z)
        
        # Introduci l'anomalia con un valore differente (ad esempio una distribuzione anomala)
        anomaly = np.random.normal(loc=10.0, scale=5.0, size=(scale_x, scale_y, scale_z))
        tensor[start_x:start_x + scale_x, start_y:start_y + scale_y, start_z:start_z + scale_z] = anomaly

    return tensor

  """This function realizes some slices of the starting tensor as matrix NxN slices along all the components"""
  def hitmaps(self, X, Y, Z, tensor, index):
    num_slices = 10  # numero di fette da visualizzare
    fig, axs = plt.subplots(3, num_slices, figsize=(15, 5))
    
    for i in tqdm(range(num_slices), desc="Stamping hitmaps..."):
        slice_idx = Z // (num_slices + 1) * (i + 1)  
        axs[0][i].imshow(tensor[:, :, slice_idx], cmap='hot')
        axs[0][i].set_title(f"Slice {slice_idx}")

    for i in tqdm(range(num_slices), desc="Stamping hitmaps..."):
        slice_idx = Y // (num_slices + 1) * (i + 1)  
        axs[1][i].imshow(tensor[:, slice_idx, :], cmap='hot')
        axs[1][i].set_title(f"Slice {slice_idx}")

    for i in tqdm(range(num_slices), desc="Stamping hitmaps..."):
        slice_idx = X // (num_slices + 1) * (i + 1)  
        axs[2][i].imshow(tensor[slice_idx, :, :], cmap='hot')
        axs[2][i].set_title(f"Slice {slice_idx}")
        
    plt.savefig(f'img{index}.png')
    plt.show()
    plt.close()


  '''Given the desired index from the main, it reshape the df into a tensor as the user wants'''
  def reshape_ortogonal_tensor(self, temporal_indices, spatial_indices):
    if temporal_indices[0] == 0 and temporal_indices[1] == 0:
        print('Please set at least one temporal index different from 0')
        return
    
    if spatial_indices[0] == 0 and spatial_indices[1] == 0 and spatial_indices[2] == 0:
        print('Please set at least one spatial index different from 0')
        return
    
    new_temporal_indices = [x for x in temporal_indices if x != 0]
    new_spatial_indices = [x for x in spatial_indices if x != 0]

    
    if 0 not in temporal_indices and 0 not in spatial_indices:
      self.df = self.df.reshape(temporal_indices[0], temporal_indices[1], spatial_indices[0], spatial_indices[1], spatial_indices[2])
    else:
      self.df = self.df.reshape(*new_temporal_indices, *new_spatial_indices)
  
  
  '''You have to pass the indices without zeros'''
  # devi cambiare tutte le volte la shape da cui parti per essere sicuro
  def reshape_linear_tensor(self, temporal_indices, spatial_indices, standardize=False):

    self.temporal_indices = temporal_indices
    self.spatial_indices = spatial_indices
    self.df = np.array(self.df.reshape(self.tuple_prod(temporal_indices), self.tuple_prod(spatial_indices)))
    if standardize == True:

      mean = np.mean(self.df, axis=0)
      std_dev = np.std(self.df, axis=0)
      self.df = (self.df - mean)/(std_dev)
      self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())


  def reshape_with_moveaxis(self, temporal_indices, spatial_indices, axis_order, standardize=False):

    self.temporal_indices = temporal_indices
    self.spatial_indices = spatial_indices
    # Move axes according to the specified order
    moved_tensor = np.moveaxis(self.df, source=(0, 1, 2), destination=axis_order)
    
    # Reshape based on new temporal and spatial structure to obtain a 2D array
    reshaped_tensor = moved_tensor.reshape(
        self.tuple_prod(temporal_indices),
        self.tuple_prod(spatial_indices)
    )
    
    # Standardize if needed
    if standardize:
        mean = np.mean(reshaped_tensor, axis=0)
        std_dev = np.std(reshaped_tensor, axis=0)
        reshaped_tensor = (reshaped_tensor - mean) / std_dev
        reshaped_tensor = (reshaped_tensor - reshaped_tensor.min()) / (reshaped_tensor.max() - reshaped_tensor.min())
    
    return reshaped_tensor



  def create_statistical_model(self, string_model):
      try:
        if string_model == 'KMeans':
            self.str_model=string_model
            self.model = KMeans(n_clusters=5) 
        elif string_model == 'IsolationForest':
            self.str_model=string_model
            self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.014), random_state=42) 
        elif string_model == 'SVM':
            self.str_model=string_model
            self.model = OneClassSVM(gamma='auto')  
        elif string_model == 'LOF':
            self.str_model=string_model
            self.model = LocalOutlierFactor() 
        elif string_model == 'PCA':
            self.str_model=string_model
            self.model = PCA(n_components=3)
        else:
            raise ValueError('Model name not recognized')
      except ValueError as e:
        print(f"Error creating model: {e}")
        return None


  def create_PCA(self):
    explained_variances=[]

    for i in range(10):
      pca = PCA(n_components=i+1)
      pca.fit(self.df)
      explained_variances.append(sum(pca.explained_variance_ratio_))
    self.PCA_Ncomponents = next((i for i, valore in enumerate(explained_variances) if valore > self.variance_components), len(explained_variances) - 1)+1
    self.model=PCA(n_components=self.PCA_Ncomponents)
    return


  def hyperopt_deep_model(self, string_model):
      
    self.str_model = string_model + f"_layers{self.layers}_KerStrd_{self.ker_strd}_nodes_64"
    self.model = keras.Sequential()
    for i in range(self.layers):
        self.model.add(Conv1D(filters=4, kernel_size=self.ker_strd, strides=self.ker_strd, activation='relu', padding='same', input_shape=(self.tuple_prod(self.spatial_indices), 1)))
    self.model.add(Flatten())
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dense((self.df.shape[1]), activation='linear'))
    self.model.add(Lambda(lambda x: x, output_shape=lambda s: s))
     

  
  def suitable_conv(self, filters, ker_str, dense, input):
        # Convolution
        encoded = tf.keras.layers.Conv1D(filters=filters, kernel_size=ker_str, strides=ker_str, activation='relu', padding='same')(input)
        # Downsampling
        encoded_fc1 = tf.keras.layers.Dense(dense, activation='relu')(encoded)
        encoded_fc2 = tf.keras.layers.Dense(dense, activation='relu')(encoded_fc1)
        # Deconvolution    
        decoded = tf.keras.layers.Conv1DTranspose(1, kernel_size=ker_str, strides=ker_str,  activation='relu', padding='same')(encoded_fc2)
        return decoded


  def create_deep_model(self, string_model):
    try:
      if string_model == 'conv1d':
        self.str_model=string_model + f"_nodes_20"
        input= [self.tuple_prod(self.spatial_indices), 1]
        input_img = keras.layers.Input(shape=input)
        if len(self.spatial_indices) == 1:
          self.model = keras.models.Model(input_img, self.suitable_conv(filters=2, ker_str=int(self.spatial_indices[0]/2), dense=20, input=input_img))
        else:
          self.model = keras.models.Model(input_img, self.suitable_conv(filters=2, ker_str=self.spatial_indices[0], dense=20, input=input_img))
        
      elif string_model == 'conv2d':
        self.str_model=string_model
        self.model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(self.df.shape[1] * self.df.shape[2], activation='sigmoid'),
        Reshape((self.df.shape[1], self.df.shape[2])) 
    ])
        
      elif string_model == 'conv3d':
        self.str_model=string_model
        self.model = keras.Sequential([
          Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], self.df.shape[3], 1)),
          MaxPooling3D((2, 2, 2)),
          Flatten(),
          Dense(64, activation='relu'),
    
          Dense(self.df.shape[1]*self.df.shape[2]*self.df.shape[3], activation='sigmoid'),
          Reshape((self.df.shape[1], self.df.shape[2], self.df.shape[3])),
          Lambda(lambda x: x, output_shape=lambda s: s)  
    ])
        
      elif string_model == 'GRU1D':
        self.str_model=string_model+f"_16_32"
        self.model = keras.Sequential([
          GRU(16, input_shape=(self.tuple_prod(self.spatial_indices), 1), return_sequences=True),
          GRU(32),
          Dense((self.df.shape[1]), activation='linear'),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'GRU2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          GRU(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM1D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=(self.tuple_prod(self.spatial_indices), 1), return_sequences=True),
          LSTM(32),
          Dense((self.df.shape[1]), activation='linear'),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          LSTM(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      else:
            raise ValueError('Model name not recognized')
      
    except ValueError as e:
          print(f"Error creating model: {e}")
          return None


  def fit_deep_model(self):
    self.model.compile(optimizer='adam', loss='mean_squared_error')
    if self.tuple_prod(self.spatial_indices) < 160:
      history= self.model.fit(self.df, self.df, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
    else:
      history= self.model.fit(self.df, self.df, epochs=self.tuple_prod(self.spatial_indices), batch_size=32, validation_split=0.1, verbose=1)
    plot_history(history)
    plt.savefig(f'final_validation_autoencoder_{self.xlsx_path}/training_model_{self.str_model}_{self.temporal_indices}_{self.spatial_indices}.png')
    plt.close()


  def fit_model(self):
      self.model.fit(self.df)

  def fit_linear_model(self):
    self.model.fit(self.df, self.df)


  def deep_anomalies(self):
    self.fit_deep_model()
    reconstructed = (self.model.predict(self.df)).reshape(self.df.shape)
    print(reconstructed.shape)
    mse = np.mean(np.power(self.df - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 100 - 10)
    anomalies_idx = np.where(mse > threshold)[0]
    self.anomalies_indices = anomalies_idx[np.argsort(mse[anomalies_idx])[::-1]]


  def anomalies_stat(self):
    anomaly_percentage = 0.05  # 5%


    if self.str_model == 'SVM':
        self.fit_model() 
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices

    elif self.str_model == 'KMeans':
        self.fit_model()  
        distances = self.model.transform(self.df)
        mean_distance = np.mean(np.min(distances, axis=1))
        std_distance = np.std(np.min(distances, axis=1))
        threshold = mean_distance + 2 * std_distance  
        self.anomalies_indices = np.where(np.min(distances, axis=1) > threshold)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        sorted_anomalies_indices = np.argsort(np.min(distances, axis=1))[::-1]
        self.anomalies_indices = sorted_anomalies_indices[:num_anomalies]

    elif self.str_model == 'LOF':
        self.fit_model()  
        anomalies = self.model.fit_predict(self.df)
        self.anomalies_indices = np.where(anomalies < 0)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        self.anomalies_indices = np.argsort(anomalies)[::-1][:num_anomalies]

    elif self.str_model == 'IsolationForest':
        self.fit_model()  
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices

    elif self.str_model == 'PCA':
        self.fit_model()
        anomaly_method = 'reconstruction_error'  # Modifica questo valore per provare metodi diversi

        if anomaly_method == 'reconstruction_error':
            # Metodo basato sulla ricostruzione approssimata usando inverse_transform
            X_pca = self.model.transform(self.df)
            print('Explained variance:', self.model.explained_variance_ratio_)
            X_reconstructed = self.model.inverse_transform(X_pca)
            mse = np.mean(np.square(self.df - X_reconstructed), axis=1)

        elif anomaly_method == 'residual_analysis':
            # Metodo basato sull'analisi del residuo
            X_pca = self.model.transform(self.df)
            X_projected = X_pca @ self.model.components_ + self.model.mean_
            residuals = self.df - X_projected
            mse = np.mean(residuals**2, axis=1)

        elif anomaly_method == 'low_variance_components':
            # Metodo basato sulle componenti rimanenti (meno significative)
            X_pca = self.model.transform(self.df)
            num_components = self.model.n_components_
            residual_projection = X_pca[:, num_components:]  # Considera solo le componenti scartate
            mse = np.sum(residual_projection**2, axis=1)

        elif anomaly_method == 'log_likelihood_score':
            # Metodo basato su score_samples per calcolare la verosimiglianza logaritmica
            mse = -self.model.score_samples(self.df)  # Inverti per identificare le anomalie come valori alti

        else:
            raise ValueError("Unknown anomaly detection method")

        # Calcolo del threshold per identificare le anomalie
        threshold = np.percentile(mse, 100 - anomaly_percentage * 100)
        anomaly_indices = np.where(mse > threshold)[0]

        # Ordinamento degli indici delle anomalie per l'importanza
        self.anomalies_indices = anomaly_indices[np.argsort(-mse[anomaly_indices])]


    else:
      print("Unknown model")


  def save_linear_anomaly_indices(self):
      # divido l'indice dell'anomalia per uno degli indici temporali, poi trovo l'intero piu vicino e ho fatto teoricamente  f'final_validation_autoencoder_{self.xlsx_path}

    with open(f'{self.xlsx_path}/anomalies_{self.str_model}_pruned_{self.temporal_indices}_{self.spatial_indices}.txt', 'w') as file:
        for indice in self.anomalies_indices:
          
          if len(self.temporal_indices) == 2:
            indice= int(indice/(self.temporal_indices[1])) +1, round(self.temporal_indices[1]*((indice/self.temporal_indices[1]) % 1)) +1

          elif len(self.temporal_indices) == 3:
            indice=round(self.temporal_indices[0]*((indice/(self.temporal_indices[0]*self.temporal_indices[1])) % 1)), int(self.temporal_indices[1]*((indice/(self.temporal_indices[1])) % 1)) +1, int(indice/(self.temporal_indices[1]*self.temporal_indices[0])) +1
            
          file.write(f"{indice}\n")


  def stamp_all_shape_anomalies(self, possible_shapes, model):

      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
            self.reshape_linear_tensor(temporal_indices, spatial_indices, standardize=False)
            self.create_statistical_model(model)
            self.anomalies_stat()
            self.save_linear_anomaly_indices()


  def stamp_all_views_anomalies(self, views, model):
    """
    Applies anomaly detection from multiple perspectives (X, Y, Z).
    
    Parameters:
    - views (list): A list containing dimensions of the original tensor [X, Y, Z].
    - model (str): The name of the anomaly detection model.
    """
    self.xlsx_path = 'anomalies_cubes.xlsx'
    self.create_statistical_model(model)
    
    # Partiamo dal tensore 3D originale per ciascuna prospettiva
    original_df = self.df.copy()  # Conserviamo il tensore originale

    # Prospettiva X: nessun cambio assi, solo reshape
    reshaped_data_X = self.reshape_with_moveaxis(temporal_indices=[views[0]], spatial_indices=[views[1], views[2]], axis_order=(0, 1, 2))
    print("Prospettiva X:", reshaped_data_X.shape)
    self.df = reshaped_data_X
    self.anomalies_stat()
    self.save_linear_anomaly_indices()
    self.df=original_df

    # Prospettiva Y: ruota assi, poi reshape
    reshaped_data_Y = self.reshape_with_moveaxis(temporal_indices=[views[1]], spatial_indices=[views[2], views[0]], axis_order=(2, 0, 1))
    print("Prospettiva Y:", reshaped_data_Y.shape)
    self.df = reshaped_data_Y
    self.anomalies_stat()
    self.save_linear_anomaly_indices()
    self.df=original_df

    # Prospettiva Z: ruota assi, poi reshape
    reshaped_data_Z = self.reshape_with_moveaxis(temporal_indices=[views[2]], spatial_indices=[views[0], views[1]], axis_order=(1, 2, 0))
    print("Prospettiva Z:", reshaped_data_Z.shape)
    self.df = reshaped_data_Z
    self.anomalies_stat()
    self.save_linear_anomaly_indices()
    self.df=original_df

  
  def stamp_all_shape_deep_anomalies(self, possible_shapes, model):

      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
        self.reshape_linear_tensor(temporal_indices, spatial_indices)
        self.create_deep_model(model)
        self.deep_anomalies()
        self.save_linear_anomaly_indices()


  def hyperopt_anomalies(self, possible_shapes, model, possible_combinations):
    for combinations in tqdm(possible_combinations, desc="Stamping hyperpatameters"):
      ker_strd, layers = combinations
      self.ker_strd = ker_strd
      self.layers = layers
      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
        self.reshape_linear_tensor(temporal_indices, spatial_indices)
        self.hyperopt_deep_model(model)
        self.deep_anomalies()
        self.save_linear_anomaly_indices()

  
  def pruning(self, window):
    self.df=self.df[:-window]
    return self.df
     
  def antipruning(self, window):
     self.df=self.df[-window:]
     return self.df
    


class sheet:
    def __init__(self):
        pass

    def load_timestamps(self, path, sens_num):
        self.df = []
        for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            sheet_df = sheet_df['Date Acquisition']
            self.df.append(sheet_df)
        return self.df


    def get_date(self, timestamp):
        return pd.to_datetime(timestamp, dayfirst=True).date()


    def find_discontinuity(self, *args):
        prev_date = None
        discs=[]
        dates=[]
        for idx, timestamps in enumerate(args):
            for i, timestamp in enumerate(tqdm(timestamps, desc=f'Analizing array {idx}', unit='timestamp')):
                curr_date = self.get_date(timestamp)

                if prev_date is None:
                    prev_date = curr_date
                    continue
        
                if curr_date != prev_date + timedelta(days=1) and curr_date != prev_date + timedelta(days=0):
                    print(f"Discontinuity found on array {idx} on date : {prev_date}")
                    print(f"Delta days: {abs(curr_date - prev_date).days}")
                    discs.append(idx)
                    dates.append(prev_date)
                prev_date = curr_date
            
            prev_date = None
        return discs, dates



class printer():
  def __init__(self):
        pass

  def load(self, path, sens_num):
    self.df = []
    self.xlsx_path=path
    for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            sheet_df = sheet_df.drop(['Unnamed: 0', 'off_ch1', 'off_ch2', 'off_ch3', 'off_ch4', 'timestamp'], axis=1)
            self.df.append(sheet_df)

  def print_all(self):
    sns.set_style('darkgrid')
    cmap = plt.get_cmap('rainbow')
    normalize = Normalize(vmin=0, vmax=15)
    for i in tqdm(range(len(self.df)), desc="Elaborazione"):
        plt.figure(figsize=(15,10))
        plt.rcParams.update({'font.size': 15})
        for idx, col in enumerate(self.df[i].iloc[:, 2:].columns):
            color = cmap(normalize(idx))
            plt.plot(self.df[i]['timestamp'], self.df[i][col], label=col, color=color)
        plt.title(self.df[i]['sensor'].iloc[0], fontsize=25)
        plt.xlabel('Time', fontsize=25)
        plt.ylabel('Intensity', fontsize=25)
        plt.xticks(fontsize=20, rotation=17)
        plt.yticks(fontsize=20)
        plt.legend()
        plt.savefig(f'graphs {self.xlsx_path}/sensor_{self.df[i]["sensor"].iloc[0]}.png')
        plt.close()


  def hitmap(self):


    data1 = (np.random.randint(low=1, high=200, size=(10, 10))/200).round(2)
    data2 = (np.random.randint(low=1, high=200, size=(10, 10))/200).round(2)
    data3 = (np.random.randint(low=1, high=200, size=(10, 10))/200).round(2)
    data4 = (np.random.randint(low=1, high=200, size=(10, 10))/200).round(2)
    


    # Crea e salva le heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data1, annot=True, cbar=False)
    plt.savefig('photos/hitmap1.png')
    plt.clf()

    sns.heatmap(data2, annot=True, cbar=False)
    plt.savefig('photos/hitmap2.png')
    plt.clf()

    sns.heatmap(data3, annot=True, cbar=False)
    plt.savefig('photos/hitmap3.png')
    plt.clf()

    sns.heatmap(data4, annot=True, cbar=False)
    plt.savefig('photos/hitmap4.png')
    plt.clf()



class graph():
  def __init__(self):
        pass

  def build_3_indices_graph(self, node_weights):   
    node_labels = {
    0: 'x_ijk', 
    1: 'x_ij^k', 
    2: 'x_jk^i', 
    3: 'x_ik^j', 
    4: 'x_i^jk', 
    5: 'x_j^ik', 
    6: 'x_k^ij', 
    7: 'x^ijk'
    }
    G = nx.DiGraph() 
    edges = [
      (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 5),
      (2, 6), (3, 6), (3, 4), (4, 7), (5, 7), (6, 7)
    ]

    G.add_edges_from(edges)
    pos = {
        0: (0, 2),
        1: (-1, 1),
        2: (0, 1),
        3: (1, 1),
        4: (-1, 0),
        5: (0, 0),
        6: (1, 0),
        7: (0, -1)
    }

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_weights, cmap=plt.cm.Wistia, node_size=700, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=30)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_color='black')
    plt.colorbar(nodes)
    plt.savefig('Anomaly graph intensity.png')
    plt.title("Anomalies weights on hierarchical graph", fontsize=16)
    plt.show()