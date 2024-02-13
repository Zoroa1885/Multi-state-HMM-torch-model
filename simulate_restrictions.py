from itertools import product
from functools import reduce
import torch

def simulate_restriktions(n_state_list):
    tot_comb = reduce(lambda x, y: x * y, n_state_list)
    N = min(n_state_list)
    count = 0
    
    for prod in product(range(N+1), repeat=tot_comb):
        if not sum(prod) == tot_comb:
            continue
        tensor = torch.tensor(list(prod)).view(tuple(n_state_list))
        ok = True
        for i in range(len(n_state_list)):
            n = n_state_list[i]
            sums = torch.sum(tensor, dim = i)
            if not torch.all(sums == n):
                ok = False
        
        if ok:
            print(tensor)
            count += 1
    
    return count

     
     