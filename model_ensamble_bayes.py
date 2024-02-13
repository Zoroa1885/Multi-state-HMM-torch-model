import torch
import torch.distributions as dist
import torch.optim as optim
import itertools
from functools import reduce
from torch import nn

class HMM_bayes(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """
    def __init__(self, n_state_list, m_dimensions, max_itterations = 100, tolerance = 0.1, verbose = True, lambda_max_itter = 100, lambda_tol = 10, use_combine = True ):
        super(HMM_bayes, self).__init__()
        self.n_state_list = n_state_list  # number of states
        self.T_max = None # Max time step
        self.m_dimensions = m_dimensions 
        self.num_laten_variables = len(self.n_state_list)
        self.dim_to_reduce = tuple(range(1, self.num_laten_variables))
        
        self.lambda_max_itter = lambda_max_itter
        self.lambda_tol = lambda_tol
        
        self.max_itterations = max_itterations
        self.tolerance = tolerance
        
        self.verbose = verbose
        
        # A
        self.transition_matrix_list = []
        self.log_transition_matrix_list = []
        for N in n_state_list:
            transition_matrix = torch.nn.functional.softmax(torch.randn(N,N)*10, dim = 0)
            self.transition_matrix_list.append(transition_matrix)
            self.log_transition_matrix_list.append(transition_matrix.log())
        self.transition_matrix_combo = None
        
        # b(x_t)
        ## Only lambdas are parameters that needs to be optimized
        self.lambda_list = []
        for N in n_state_list:
            lambda_i = torch.nn.Parameter(torch.exp(torch.rand(N, m_dimensions)*10))
            self.lambda_list.append(lambda_i)
        self.log_emission_matrix = None
        self.emission_matrix = None

        # If using the algorithm which fits the combined lambdas
        self.use_combine = use_combine
        if use_combine:
            self.lambda_combined = self.combine_lambdas()
        

        # pi
        self.state_priors_list = []
        self.log_state_priors_list = []
        for N in n_state_list:
            state_priors = torch.nn.functional.log_softmax(torch.rand(N)*10)
            self.state_priors_list.append(state_priors)
            self.log_state_priors_list.append(state_priors.exp())
        

        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.cuda()

    def emission_model(self, x, log = True):
        """
        x: LongTensor of shape (T_max, m_dimensions)

        Get observation log probabilities
        Returns:
            log_probabilities: FloatTensor of shape (T_max, self.n_states_list)
        """
        # Compute Poisson log probabilities for each lambda in parallel
        if self.use_combine:
            combined_lambdas = self.lambda_combined
        else:
            combined_lambdas = self.combine_lambdas()
        
        poisson_dist = dist.Poisson(combined_lambdas)
        dim_probs = (x.shape[0],) + tuple(self.n_state_list)
        log_probabilities = torch.zeros(dim_probs)
        for t in range(x.shape[0]):
            # The last dimension specify the dimension of the data.
            # We sum this dimension togheter.
            log_probabilities[t,:] = poisson_dist.log_prob(x[t,:]).sum(dim=-1)
        if log:
            return log_probabilities
        
        return log_probabilities.exp()
        
    
    def alpha_calc(self):
        alpha_list = []
        for i in range(len(self.emission_matrix_list)):
            emission_matrix = self.emission_matrix_list[i]
            alpha = torch.zeros(self.T_max, self.n_state_list[i]).float()
            if self.is_cuda:
                alpha = alpha.cuda()
            
            alpha[0,:] = emission_matrix[0,:] + self.log_state_priors
            
            for t in range(1, self.T_max):
                alpha[t, :] = emission_matrix[t, :] * torch.matmul(self.transition_matrix, alpha[t-1, :])
            
            return alpha_list
    
    def beta_calc(self):
        beta = torch.ones(self.T_max, self.n_states).float()
        if self.is_cuda:
            beta = beta.cuda()
        
        for t in range(self.T_max - 2, -1, -1):
            beta_t_s = torch.zeros(self.n_states).float()
            for s in range(self.n_states):
                for k in range(self.n_states):
                    beta_t_s[s] += beta[t+1,k]*self.transition_matrix[s,k]*self.emission_matrix[t+1,k]
            beta[t,:] = beta_t_s
        
        return beta
                    
    def log_alpha_calc(self):
        """
        self.log_emission_matrix: longTensor of shape (T_max, n_states)

        Returns:
            log_alpha: floatTensor of shape (T_max, n_states)
        """
        assert self.log_emission_matrix is not None, "No emission matrix"
        assert self.T_max is not None, "No maximum time"
        
        tot_dim = reduce(lambda x, y : x * y, self.n_state_list)
        dim_alpha = (self.T_max,) + tuple(self.n_state_list)
        log_alpha = torch.zeros(dim_alpha).float()
        if self.is_cuda:
            log_alpha = log_alpha.cuda()

        log_alpha[0, :] = self.log_emission_matrix[0, :] + self.log_state_priors
        
        # log_alpha[1:self.T_max,:] = log_emission_matrix[1:self.T_max,:] + log_transition_matrix[0:(self.T_max-1), :]
        for t in range(1, self.T_max):
            # Creat a list of sums with log_alpha and transition_matrix_combos
            log_alpha_trans_sum = self.transition_matrix_combo + log_alpha[t-1,:].repeat(tot_dim)
            # Take logsumexp over all dimension except the first.
            # We now have vector with length equal to the product of all dimensions
            lats = log_alpha_trans_sum.logsumexp(dim = self.dim_to_reduce)
            # We then reshape the vector to a tensor with dimensions equal to the different number of states
            mat_prod = lats.view(tuple(self.n_state_list))
            # We then sum this tensor with the emission probabilities to 
            log_alpha[t, :] = self.log_emission_matrix[t, :] + mat_prod
        
        return log_alpha
    
    def log_beta_calc(self):
        assert self.log_emission_matrix_list is not None, "No emission matrix"
        
        tot_dim = reduce(lambda x, y : x * y, self.n_state_list)
        
        dim_beta = (self.T_max,) + tuple(self.n_state_list)
        log_beta = torch.zeros(dim_beta).float()
        if self.is_cuda:
            log_beta = log_beta.cuda()
    
        for t in range(self.T_max -2, -1, -1):
            log_beta_trans_sum = self.transition_matrix_combo + log_beta[t+1,:].repeat(tot_dim) + self.log_emission_matrix[t+1,:]
            sum_reduced = log_beta_trans_sum.logsumexp(dim=self.dim_to_reduce)
            log_beta[t,:] = sum_reduced.view(tuple(self.n_state_list))
        
        return log_beta
        
    
    def forward(self, x):
        """
        x: IntTensor of shape (T_max, m_dimensions)

        Compute log p(x)
        """
        self.T_max = x.shape[0]
        self.log_emission_matrix = self.emission_model(x)
        log_alpha = self.log_alpha_calc()

        log_prob = log_alpha[self.T_max-1, :].logsumexp()
        return log_prob
    
    def get_lambdas(self):
        return self.lambdas_list
            
    
    def fit(self, x):
        """ Estimates optimal transition matrix and lambdas given the data x.

        Args:
            x (torch): T_max x m_dimensions
            log_alpha (torch) : T_max x N
            log_beta (torch) : T_max x N
        """
        
        self.T_max = x.shape[0]
        prev_log_likelihood = float('-inf')
        log_x = torch.log(x + 1e-16)
        
        for iteration in range(self.max_itterations):
            # Get emission matrix
            self.log_emission_matrix = self.emission_model(x)
            
            self.transition_matrix_combo = self.combine_transition_matrices()
            # E step
            ## Calculate log_alpha
            log_alpha = self.log_alpha_calc()
            
            ## Caculcate log_beta
            log_beta = self.log_beta_calc()
            
            # Chack for tolerance
            log_likelihood = log_alpha[self.T_max - 1, :].logsumexp()
            log_likelihood_change = log_likelihood - prev_log_likelihood
            prev_log_likelihood = log_likelihood
            if self.verbose:
                if log_likelihood_change > 0:
                    print(f"{iteration + 1} {log_likelihood:.4f}  +{log_likelihood_change}")
                else:
                    print(f"{iteration + 1} {log_likelihood:.4f}  {log_likelihood_change}")
            
            if log_likelihood_change < self.tolerance and log_likelihood_change > 0:
                if self.verbose:
                    print("Converged (change in log likelihood within tolerance)")
                break
            
            ## Calculate log_gamma
            gamma_numerator = log_alpha + log_beta
            gamma_denominator = gamma_numerator.logsumexp(dim=self.dim_to_reduce, keepdim=True)
            
            log_gamma = gamma_numerator - gamma_denominator.expand_as(gamma_numerator)
            
            log_gamma_list = []
            log_emission_matrix_list = []
            log_alpha_list = []
            log_beta_list = []
            for i in range(self.num_laten_variables):
                dim_reduce_i = tuple(list(self.dim_to_reduce).remove(i+1))
                
                log_gamma_i = log_gamma.logsumexp(dim = dim_reduce_i) 
                log_gamma_list.append(log_gamma_i)
                
                log_emission_matrix_i = self.log_emission_matrix.logsumexp(dim = dim_reduce_i)
                log_emission_matrix_list.append(log_emission_matrix_i)
                
                log_alpha_i = log_alpha.logsumexp(dim = dim_reduce_i)
                log_alpha_list.append(log_alpha_i)
                
                log_beta_i = log_beta.logsumexp(dim = dim_reduce_i)
                log_beta_list.append(log_beta_i)
                
            
            
            ## Calculate log_xi
            log_xi_list = []
            for i in range(self.num_laten_variables):
                xi_numerator_i = (log_alpha_list[i][:-1, :, None] + self.log_transition_matrix[None, :, :] + log_beta_list[i][1:, None, :] + log_emission_matrix_list[i][1:, None, :])
                xi_denominator = xi_numerator_i.logsumexp(dim = (1,2), keepdim=True)
                
                log_xi_i = xi_numerator_i - xi_denominator
                log_xi_list.append(log_xi_i)
                
            # M step
            ## Update pi
            for i in range(len(self.n_state_list)):
                log_gamma_i = log_gamma_list[i]
                self.log_state_priors_list[i] = log_gamma_i[0,] - log_gamma_i.logsumexp(dim = 0)
            
            ## Updaten transition matrix
            for i in range(self.num_laten_variables):
                trans_numerator_i = log_xi_list[i].logsumexp(dim = 0)
                trans_denominator_i = log_gamma_list[i][0:(self.T_max-1),:].logsumexp(dim = 0)
                
                self.log_transition_matrix_list[i] =  trans_numerator_i - trans_denominator_i.view(-1, 1)
            
            ## Update lambda
            ### Optimizing every combination of lambdas.
            if self.use_combined:
                lambda_numerator = log_domain_matmul(log_gamma.t(), log_x, dim_1=False)
                lambda_denominator = log_gamma.logsumexp(dim = 0)
                self.lambda_combined = torch.exp(lambda_numerator - lambda_denominator.view(-1,1))
            
            ### Optimizing indvidual lambdas
            else:
                optimizer = optim.Adam(self.parameters(), lr = 0.01)
                lambda_loss = log_likelihood
                for epoch in range(self.lambda_max_itter):
                    # Optimize
                    optimizer.zero_grad()
                    lambda_loss.backward()
                    optimizer.step()
                    
                    # Calculate new loss
                    new_lambda_loss = self.forward(x)
                    lambda_likelihood_change = lambda_loss - new_lambda_loss
                    if lambda_likelihood_change > 0 and lambda_likelihood_change < self.lambda_tol:
                        break
                    lambda_loss = new_lambda_loss
            
            if self.verbose and iteration == self.max_itterations -1:
                print("Max itteration reached.")
        
        # Do a gradient search to find best approximation of seperated lambdas. 
        optimizer = optim.Adam(self.parameters(), lr = 0.01)
        lambda_loss = self.seperation_loss()
        for epoch in range(self.lambda_max_itter):
            # Optimize
            optimizer.zero_grad()
            lambda_loss.backward()
            optimizer.step()
            
            # Calculate new loss
            new_lambda_loss = self.seperation_loss()
            lambda_likelihood_change = lambda_loss - new_lambda_loss
            if lambda_likelihood_change > 0 and lambda_likelihood_change < self.lambda_tol:
                break
            lambda_loss = new_lambda_loss
    
    def predict(self, x):
        """
        x: IntTensor of shape (T_max, m_dimensions)

        Find argmax_z log p(z|x)
        """
        if self.is_cuda:
            x = x.cuda()

        T_max = x.shape[0]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_delta = torch.zeros(T_max, self.n_states).float()
        psi = torch.zeros(T_max, self.n_states).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        self.log_emission_matrix = self.emission_model(x)
        
        log_delta[0, :] = self.log_emission_matrix[0,:] + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = log_domain_matmul(self.log_transition_matrix, log_delta[t-1,:], max = True)
            log_delta[t, :] = self.log_emission_matrix[t,:] + max_val
            psi[t, :] = argmax_val

        z_star = torch.zeros(T_max).long()
        z_star[T_max-1] = log_delta[T_max-1, :].argmax()
        for t in range(T_max-2, -1, -1):
            z_star[t] = psi[t+1, z_star[t+1]]

        return z_star

    def get_transition_matrix(self):
        return torch.exp(self.log_transition_matrix)
    
    def combine_lambdas(self):
        """Combines every possible combination of lambdas and adds them togheter

        Returns:
            lambda_combined: FloatTensor of shape (self.n_state_list)
        """
        dim_tuple = tuple(self.n_state_list) + (self.m_dimensions,)
        lambda_combined = torch.zeros(dim_tuple, dtype=torch.float)

        # For each time step the data is on the form x_t = (x_t1,...,x_tm)
        # We itterate over each dimension and caclulate the combined lambdas
        # for each dimension
        for m in range(self.m_dimensions):
            # A list of all lambdas for that dimension in every statespace
            sets = [element[:, m] for element in self.lambda_list]

            # Find every possible combination of lambdas
            combinations = itertools.product(*sets)
            # Each combination is sumed togheter and stacked into a tensor
            combined_set = torch.stack([torch.sum(list(combo)) for combo in combinations])
            # Reshape the tensor such that it maches the dimensions of the number of states
            lambda_combined[:, m] = combined_set.view(tuple(self.n_state_list))
        
        return lambda_combined

    def combine_transition_matrices(self):
        tensor_list = []
        sets = []
        for trans_mat in self.log_transition_matrix_list:
            trans_mat_rows = [row for row in trans_mat]
            sets.append(trans_mat_rows)
        combinations = itertools.product(*sets)
        for combo in combinations:
            sub_combo = itertools.product(*list(combo))
            combined_set = torch.stack([torch.sum(list(c)) for c in sub_combo])
            tensor_list.append(combined_set.view(tuple(self.n_state_list)))
        return torch.tensor(tensor_list)
    
    
    def seperation_loss(self, h = 1):
        combine_lambdas = self.combine_lambdas()
        loss = nn.MSEloss(combine_lambdas, self.combine_lambdas) 
        for l in self.lambda_list:
            loss += h*torch.norm(l)    
        return loss
    
        
        
def log_matrix_multiply(log_A, log_B):
    # Ensure that the dimensions match for element-wise addition
    assert log_A.shape[1] == log_B.shape[0], "Inner dimensions do not match for matrix multiplication"

    # Perform element-wise addition in log-space
    log_result = log_A.unsqueeze(2) + log_B.unsqueeze(0)

    # Calculate the log of the sum of exponentiated values (equivalent to log-domain matrix multiplication)
    log_result = torch.logsumexp(log_result, dim=1)

    return log_result


def log_domain_matmul(log_A, log_B, dim_1 = True, max = False):
    """
    log_A: m x p
    log_B: n x p
    output: m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    if not dim_1:
        m = log_A.shape[0] 
        n = log_A.shape[1]
        p = log_B.shape[1]

        log_A = torch.stack([log_A] * p, dim=2)
        log_B = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A + log_B
    if max:
        out1, out2 = torch.max(elementwise_sum, dim = 1)
        return out1, out2
    
    out = torch.logsumexp(elementwise_sum, dim=1)
    return out

def maxmul(log_A, log_B):
    elementwise_sum = log_A + log_B
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2

def combination_addition(mat1, mat2):
    assert mat1.shape[1] == mat2.shape[1], "Different number of dimensions"
    n = mat1.shape[0]
    m = mat2.shape[0]
    dim = mat1.shape[1]
    comb_mat = torch.tensor(n, m, dim)
    
    for i in range(n):
        for j in range(m):
            comb_mat[i,j,:] = mat1[i,:] + mat2[j,:]
            
def combine_lambdas(lambda_list, m_dimensions):
    dim_tuple = tuple(self.n_state_list)
    lambda_combined = torch.tensor()
    for m in m_dimensions:
        sets = []
        for element in lambda_list:
            sets.append(element[m])
        
        combinations = []
        for combo in itertools.product(*sets):
            combinations.append(torch.stack(list(combo)).sum())
        
 