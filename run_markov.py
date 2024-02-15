from model_ensamble_bayes import HMM_bayes
import torch
from utils import *
import numpy as np
import random
import math
import pandas as pd
import os 
import joblib

id = np.random.randint(low = 10000, high = 99999)

# Simulate date
num_neuron_list = [25, 25]
num_state_list = [3, 4]
frequency_list = [5, 10]
data_length = 5000
df_ensamble = pd.DataFrame([])
y_ensamble = np.array([])

for i in range(len(num_state_list)):
    df, y = simulate_state(num_neuron_list[i], num_state_list[i], frequency_list[i], data_length)
    if df_ensamble.empty:
        df_ensamble = df
        y_ensamble = y.reshape(-1, 1)
    else:
        df_ensamble = pd.concat([df_ensamble, df], axis = 1)
        y_ensamble = np.concatenate((y_ensamble, y.reshape(-1, 1)), axis = 1)



# Define parameteres
n_state_list = [3, 4]
m_dimensions = sum(num_neuron_list)
max_itterations = 1
tolerance = 0.00000001
lambda_tol = 0.1
n_retraining = 1

model_list = list()
score_list = list()

X = torch.tensor(df_ensamble.values)

# Train model
for _ in range(n_retraining):
    model = HMM_bayes(n_state_list, m_dimensions, max_itterations, tolerance, lambda_tol = lambda_tol)
    model.fit(X)
    
    model_list.append(model)
    score_list.append(model.forward(X).item())


best_model = model_list[np.argmax(score_list)]
best_pred = best_model.predict(X).numpy()

# Save model and data
path = f"sim_ensamble_d={data_length}_n={num_neuron_list}_s={num_state_list}_fr={frequency_list}_ms={n_state_list}_id={id}"

df_path = f'simulation/{path}/df'+ ".csv"
y_path = f'simulation/{path}/y'+ ".csv"
pred_path = f'simulation/{path}/pred'+ ".csv"

file_path = f'simulation/{path}/model.pkl'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
joblib.dump(model, file_path)

df_ensamble.to_csv(df_path, index = False)
np.savetxt(y_path, y_ensamble, delimiter=",")
np.savetxt(pred_path, best_pred, delimiter=",")

