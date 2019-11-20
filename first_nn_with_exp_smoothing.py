import time
import numpy as np
import torch
import torch.nn as nn
import database_connection
import sys


start_time = time.time()

start_date = "'2018-04-01'"
end_date = "'2018-08-31'"

# start_date = "'2019-04-01'"
# end_date = "'2019-08-31'"

articule = 361771  # 361771 519429 97143 32547 32485 117
reference_level = 10.0

conn = database_connection.DatabaseConnection()
filials_set = conn.get_filials_for_articule(articule, start_date, end_date, min_days=40)
silpo_fora_trash_filials_set = conn.get_silpo_fora_trash_filials_set()
filials_set = filials_set.intersection(silpo_fora_trash_filials_set)

print("Number of filials:", len(filials_set))
filials_list = list(filials_set)

list_of_time_series = []
list_of_promos = []
list_of_residues = []

epsilon = 0.1
for filial in filials_list:
    df = conn.get_all_sales_by_articule_and_filial_with_residues(articule, filial, start_date, end_date)
    normed_quantities = (df['quantity'].values+epsilon)/reference_level
    is_promo_day = df['is_promo_day'].values
    normed_residues = df['residue'].values/reference_level
    list_of_time_series.append(normed_quantities)
    list_of_promos.append(is_promo_day)
    list_of_residues.append(normed_residues)

start_time = time.time()

# now we want to find optimal alpha
beta = torch.tensor(0.5, requires_grad=True)
alpha = torch.sigmoid(beta)
beta_optimizer = torch.optim.Adam([beta], lr=0.01)

rnn = nn.RNN(input_size=1, hidden_size=2, num_layers=1, batch_first=False)
lin_layer = nn.Linear(in_features=2, out_features=1)
neural_net_optimizer = torch.optim.Adam(list(rnn.parameters()) + list(lin_layer.parameters()), lr=0.01)

num_epochs = 20
for i in range(num_epochs):
    for k in range(len(list_of_time_series)):
        neural_net_loss = torch.tensor(0.0, requires_grad=True)
        h = torch.tensor([0.0, 0.0], requires_grad=True)
        h = h.view(1, 1, 2)
        quantities = list_of_time_series[k]
        residues = list_of_residues[k]
        is_promo_day = list_of_promos[k]
        start_index = 0
        while start_index < len(quantities) and quantities[start_index]==0:
            start_index +=1
        if start_index >= len(quantities)-10:
            continue
        S = torch.tensor(quantities[start_index]) # requires_grad=True
        assert not np.isnan(beta.detach()), f"Sorry, beta is nan. Filial {filials_list[k]}, k={k}, i={i}"
        alpha = torch.sigmoid(beta)
        neural_net_output = torch.tensor(0.0, requires_grad=True)
        for j in range(start_index + 1, len(quantities)):
            if not is_promo_day[j] and residues[j]>0:
                current_quantity = torch.tensor(quantities[j]).view(1, 1, 1)
                neural_net_input = torch.log(current_quantity / S)
                neural_net_loss = neural_net_loss + torch.abs(neural_net_output-neural_net_input)
                S = (1 - alpha) * S + alpha * current_quantity
                neural_net_output, h = rnn(neural_net_input, h)
                neural_net_output = lin_layer(neural_net_output)
                # assert h.size() == torch.Size([1, 1, 2]), f"h size is not appropriate, k={k}, j={j}, h size: {h.size()}"
        neural_net_loss = neural_net_loss / len(quantities)
        assert not np.isnan(neural_net_loss.detach()), f"Sorry, neural_net_loss is nan. Filial {filials_list[k]}, k={k}, i={i}"
        beta_optimizer.zero_grad()
        neural_net_optimizer.zero_grad()
        neural_net_loss.backward()
        beta_optimizer.step()
        neural_net_optimizer.step()
        assert not np.isnan(beta.detach()), f"Sorry, beta is nan. Filial {filials_list[k]}, k={k}, i={i}"
alpha = torch.sigmoid(beta)
print(f"Articule: {articule} alpha: {alpha} beta: {beta}")
assert not np.isnan(beta.detach()), "Sorry, beta is nan"
assert not np.isnan(alpha.detach()), "Sorry, alpha is nan"

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Neural net learning took {hours:.2f} hours")
print(f"Neural net learning took {minutes:.2f} minutes")
print(f"Neural net learning took {seconds:.2f} seconds")
