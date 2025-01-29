import pandas as pd
from datetime import datetime
import detectorlib
# Sostituisci "file_name.xlsx" con il nome del tuo file
path="2021_All raw data_No_background.xlsx"
path2="2022_TN_SEI_field_maturation_final_matrix.xlsx"
path3="2022_LOU_AZE_field_maturation_final_matrix.xlsx"
import numpy as np


sh=detectorlib.sheet()
df=sh.load_timestamps(path3, 14)
print(type(df[0]))

all=sh.find_discontinuity(*df)
