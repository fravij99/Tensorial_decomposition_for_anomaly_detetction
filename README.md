

<h1 align="center"> Numerical simulation for understanding anomalies hierarchical structures </h1> <br>

<img align="right" src=https://github.com/fravij99/Numerical_simulation_anomalies_hierarchy/blob/master/images/3d_hierarchy.png  width="270">
Hey there! 

This is my master thesis future insights repository. 
This library provides tools for anomaly detection in multi-dimensional datasets and the generation of new anomalies and datasets by numerical simulation. It includes methods for data preprocessing, model creation, and anomaly detection using various algorithms.

## Usage
### Anomaly Detection Library
This Python library offers a range of functionalities for anomaly detection and sinthetic-data creation, particularly tailored for sensor data analysis. It provides tools to preprocess, reshape, and apply various statistical and deep learning models for anomaly detection.

### detector Class
* `tuple_prod(tupla)`
Computes the product of a tuple of numbers.
* `load_preprocess(path, sens_num)`
Reads data from an Excel file and preprocesses it for anomaly detection.
Appends the sheets to the third index of the tensor: (temporal samples, features, sensors).
* `reshape_tensor(temporal_indices, spatial_indices)`
Reshapes the tensor according to the desired indices.
* `create_model(string_model)`
Creates different statistical models such as KMeans, IsolationForest, SVM, LOF, and Linear Regression.
* `create_deep_model(string_model)`
Creates deep learning models such as Conv1D, Conv2D, Conv3D, GRU1D, GRU2D, LSTM1D, and LSTM2D.
* `fit_deep_model()`
Fits the deep learning model to the data.
* `fit_model()`
Fits statistical models to the data.
* `detect_deep_anomalies_unsup()`
Detects anomalies using unsupervised deep learning models.
Plots the anomaly rate trend and identifies the threshold.
* `anomalies_sup()`
Detects anomalies using supervised models such as SVM, KMeans, LOF, Isolation Forest, and Linear Regression.
* `save_linear_anomaly_indices()`
Saves the indices of detected anomalies to a text file.
* `stamp_all_shape_anomalies(possible_shapes)`
Reshapes data for all possible shapes and detects anomalies.
* `hyperopt_statistical_models(params)`
Performs hyperparameter optimization for statistical models and saves anomaly detection results.
* `hyperopt_anomalies()`
Performs hyperparameter optimization for deep learning models and saves anomaly detection results.
* `create_PCA()`
Performs PCA analysis to determine the number of components needed to explain a desired amount of variance.
* `hyperopt_deep_model and create_deep_model`
Define and create various deep learning models (Conv1D, Conv2D, Conv3D, GRU1D, GRU2D, LSTM1D, LSTM2D) for anomaly detection.
* `deep_anomalies()` and `anomalies_sup()`
Detect anomalies using deep learning and other machine learning models, respectively.
* `stamp_all_shape_deep_anomalies(possible_shapes)`
Reshapes data for all possible shapes and detects anomalies for deep models.
* `PCA_graph()`
Generates and saves plots showing PCA results, including variance explained by different numbers of components.
* `random_matrix_loading()`
Treats random matrix generated in the main in order to verify herarchical relationships between anomaly classes.
* `random_exp_generator()`
Generates a synthetic dataset following an exponential distribution.
* `random_anomalies_generation()`
Generates some statistical distribution infecting them with noise anomalies on several hierarchic levels
*`random_big_anomaliy_generation()`
Generates some statistical distribution infecting them localize gaussian anomalies

<img align="center" src=https://github.com/fravij99/Numerical_simulation_anomalies_hierarchy/blob/master/images/random_exp_distribution_demo.png
  width="370">

<img align="center" src=https://github.com/fravij99/Numerical_simulation_anomalies_hierarchy/blob/master/images/img0.png
  width="370">

<img align="center" src=https://github.com/fravij99/Numerical_simulation_anomalies_hierarchy/blob/master/images/img1.png
  width="370">

<img align="center" src=https://github.com/fravij99/Numerical_simulation_anomalies_hierarchy/blob/master/images/img2.png
  width="370">
  
### sheet Class
* `load_timestamps(path, sens_num)`
Loads timestamps from an Excel file.
* `get_date(timestamp)`
Converts timestamps to date format.
* `find_discontinuity(*args)`
Finds discontinuities in arrays of timestamps.


### printer Class
* `load(self, path, sens_num)`
Loads data from an Excel file.
* `print_all(self)`
Generates graphs for sensor data.


## Implementation
An easy implementation of every class is shown in the main.py, statistical_main.py, deep_main.py, sheet.py and printer.py files. The other folders contain the results of anomaly detection for classical models and the temporal plot of every sensor detection. 

## Installation
You can clone the repository using the command:

```
git clone git@github.com:fravij99/Vineyard-anomaly-detection.git
```

## Requirements
* Python 3.6+
* TensorFlow
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Datetime
* Keras
* Tqdm

## License
This library is provided under the MIT License.

## Contributions
Contributions and feedback are welcome! Please feel free to open issues or submit pull requests.

## Author
Developed by [Francesco Villa][fravi]

## Contact
For questions or support, contact [fravilla30@gmail.com] or [francesco.villa6@studenti.unimi.it].

[fravi]: https://github.com/fravij99
