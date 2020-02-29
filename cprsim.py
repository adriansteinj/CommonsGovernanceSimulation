import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from scipy import stats
import scipy as sp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class CPRSim:
    '''
    Runs simulations of a specified number of agents using the resource according to
    the use rules.
    '''
    def __init__(self, network_size = 20, network = 'directed', mechanism = 'combined', rate_of_return = .85, needs = .5, ID = 0.2, AC = 0.2, neighborhood_size = 5, prosociality = .6, sanction_cost = .1, p_interaction = 0.1, markup = False):
        '''
        Inputs:
            network_size (int) The number of agents in the community.
              Default: 20
            
            network (str) The type of network structure that the community possessses. The 
              simulator can choose between three types: a small-world, undirected Barabasi-Albert 
              graph (BA), a small-world, directed random k-out-degree graph (directed), and 
              a complete network where all agents are connected to each other (complete).
              Default: directed
            
            identity_mechanism (str) The type of cooperation mechanism that the community uses to 
              coordinate the resource use. There are three options:
              - 'identity', which activates the affective responses to neighbor actions of the identity mechanism
              - 'reputation', which activates the reflective evaluation of neighbors through the reputation mechanism
              - 'combined', which activates both.
              Default: 'combined', activating both
              
            rate_of_return (float) The rate at which the resource yields returns in 
              one period of time. Depending on the common resource analyzed, this can 
              include the replenishment of a natural resource, the social surplus created 
              by community services, institutions or cultural events, or the network effects 
              that arise from information aggregation in knowledge commons.
              Default: .9
            
            needs (float) The need for the resource by each agent. It is determined 
              by micro-situational economic and social factors not modeled herein, 
              such as family size and availability of the resource for living.
              Default: .6
              
            neighborhood_size (float) The number of other agents the average individual is able
              to observe in any given round. Observable agents are called neighbors. In BA networks, 
              the agents' neighborhood sizes are distirbuted with a power law distribution around a 
              mean of 5, while all agents can observe 5 neighbors in the directed network. In the 
              complete network, neighborhood size is irrelevant since all agents are connected.
              Default: 5
            
            prosociality(float) The mean altruistic tendency of the individuals 
              in the community. This parameter is used as the prior for individual social
              norm beliefs.
              Default: .6
              
            diversity (float) The diversity of the group at the start, which informs the 
              variance of the identification of agents with the community. 
              Default: .3
            
            sanction_cost (float) The cost of a sanction, which the punisher incurs and which
              predicts the size of the sanction to the defector.
              Default: 0.1
            
            p_interaction(float) The probability of having a social interaction with another
              person during the time period. Social interaction can enhance the agent's 
              sense of belonging and thereby boost affective commitment.
              Default: 0.1
              
            markup (bool) Turns the observation of the results on and off.
              Default: False
        
        Variables:
            resource_stock (int) The stock variable of the common resource which all individuals use.
              Default: 0
        '''
        #setting network level variables
        self.network_type = network
        self.reputation_mechanism = True
        self.identity_mechanism = True
        if mechanism == 'identity':
            self.reputation_mechanism = False
        if mechanism == 'reputation':
            self.identity_mechanism = False
        self.network_size = network_size
        self.resource_stock = 0
        self.rate_of_return = rate_of_return
        self.neighborhood_size = neighborhood_size
        self.prosociality = prosociality
        self.diversity = 0.3
        self.needs = needs
        self.sanction_cost = sanction_cost
        self.p_interaction = p_interaction
        self.markup = markup
        
        #the following arrays initialize and store the agent variables
        self.lambdas = np.linspace(0.01, .99, 100) #guessed social value orientation for Bayesian updating
        self.social_norms = np.empty((self.network_size,100)) #stores social norm distributions
        self.social_norms_variance = np.empty((self.network_size,100)) #stores variance of the social norm
        self.identification = np.full(self.network_size,ID) #stores identification with the group
        self.affective_commitment = np.full(self.network_size, AC)
        self.accounts = np.full(self.network_size, 0.) #stores the economic units each agent accumulates
        
        #the following variables and arrays store system-level variables over time
        self.sanctions = 0 #counts the total number of sanctions
        self.step = 0 #counts the time steps
        self.time = [] #stores the time steps
        self.wealth = [np.mean(self.accounts)] #how wealth or debt grows over time in each agent
        self.resource_over_time = [self.resource_stock]  #how the resource develops over time
        self.id_over_time = [np.mean(self.identification)] #collects identification averages
        self.ac_over_time = [np.mean(self.affective_commitment)] #collects commitment averages
      
    
    def likelihood(self,param_vals,data):
        '''
        Takes a set of evidence and evaluates the probability
        of these outcomes over the set of possible parameter values for the variable of
        interest that the dataset captures.
        '''
        return stats.binom(n=len(data),p=param_vals).pmf(data)

    def compute_posterior(self, parameter_values, prior, likelihood, data):
        '''
        Performs Bayesian inference on a prior distribution, the social norm distribution,
        and updates it with the observations of neighbor's cooperation, computing a 
        posterior distribution over all probabilities of cooperation.
        '''
        log_prior = np.log(prior)
        log_likelihood = np.array([
            np.sum(np.log(likelihood(param, data)))
            for param in parameter_values])
        unnormalized_log_posterior = log_prior + log_likelihood
        unnormalized_log_posterior -= max(unnormalized_log_posterior)
        unnormalized_posterior = np.exp(unnormalized_log_posterior)
        area = sp.integrate.trapz(unnormalized_posterior, parameter_values)
        posterior = unnormalized_posterior / area
        return posterior

    
    def initialize(self):
        '''
        Initializes the network according to the chosen structure.
        Community members have a given social value orientation. 
        
        For the start of the identity mechanism, if active, agents have 
        a first exchange about it with initializes their social norms.
        
        For the start of the reputation mechanism, if active, the
        agents have arrays storing the reputation of neighbors.
        The links between agents are weighted according to the 
        average trust between two nodes.
        '''
        #initialize the network as a Barabasi Albert graph with a power law node degree distribution
        if self.network_type == 'BA':
            self.graph = nx.barabasi_albert_graph(self.network_size, self.neighborhood_size)
        elif self.network_type == 'complete':
            self.graph = nx.complete_graph(self.network_size)
        elif self.network_type == 'directed':
            self.graph = nx.random_k_out_graph(self.network_size, self.neighborhood_size, alpha= 1, self_loops=False)
 
        #setting social value orientations
        social_value_orientations = np.full(self.network_size,1)
        social_value_orientations[0:int(self.network_size * (1-self.prosociality))] = 0.01
        for node in self.graph.nodes:
            self.graph.nodes[node]['current_use'] = social_value_orientations[node]
        
        #observing the social value orientation of neighbors and initiating one's social norm expectations and identification
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            norms_distribution = []
            norms_differences = []
            for i in neighbors: #recording the prosociality of neighbors and the deviation from one's own norms
                norms_distribution.append(self.graph.nodes[i]['current_use'])
                norms_differences.append(abs(self.graph.nodes[i]['current_use'] - self.graph.nodes[node]['current_use']))
            
            #updating social norms with the observation of neighbors' SVOs
            norms_distribution = np.array(norms_distribution) * len(norms_distribution)
            prior = stats.beta(a=(1.01-self.graph.nodes[node]['current_use']),b=0.8).pdf(self.lambdas)
            posterior = self.compute_posterior(self.lambdas, prior, self.likelihood, norms_distribution)    
            self.social_norms[node] = posterior
            
            #updating the expected social norm variation and induced identification with the distribution of SVO differences
            norms_differences = np.array(norms_differences) * len(norms_differences)
            var_prior = stats.truncnorm(a=(self.identification[node]-1)/self.diversity,b= self.identification[node]/self.diversity,loc=(1-self.identification[node]),scale=self.diversity).pdf(self.lambdas)
            var_posterior = self.compute_posterior(self.lambdas,var_prior,self.likelihood,norms_differences)
            self.social_norms_variance[node] = var_posterior
            expected_var = []
            for _ in range(1000):
                expected_var.append(np.random.choice(self.lambdas,p=self.social_norms_variance[node]/np.sum(self.social_norms_variance[node])))
            self.identification[node] = 1 - np.mean(expected_var)
            
        #plotting the social norms of all agents
        if self.markup:
            for n in self.social_norms:
                plt.plot(self.lambdas, n, label='social norms')
            plt.title('Initial social norm distributions of all agents in the network')
            plt.xlabel('Expected cooperation')
            plt.ylabel('PDF')
            plt.show()

        self.layout = nx.spring_layout(self.graph)  # Initial visual layout

        #set reputations of neighbors according to agent's social value orientation
        for node in self.graph.nodes:
            self.graph.nodes[node]['reputation'] = np.empty(self.network_size)
            for i in self.graph.neighbors(node):
                self.graph.nodes[node]['reputation'][i] = social_value_orientations[node]
        self.layout = nx.spring_layout(self.graph)  # Initial visual layout
            
    def draw_network(self):
        '''
        Draw the connections between nodes in the network visually.
        '''
        self.layout = nx.spring_layout(self.graph, pos = self.layout, iterations=5)
        plt.figure()
        plt.clf()
        plt.title('Step: ' + str(self.step))
        print('Resource level: ',self.resource_stock)
        
    
    def show_results(self):
        '''
        Draw figures of social norm distributions at the last step, as well as
        the evolution of average resource and wealth levels, and average 
        idenfication and affective commitment over time. Lastly, draw the 
        distributions of wealth, ID and AC after the last step.
        '''
        for n in self.social_norms:
            plt.plot(self.lambdas, n, label='social norms')
        plt.title('Social norm distributions of all agents in the network')
        plt.xlabel('Expected cooperation')
        plt.ylabel('PDF')
        
        fig, axs = plt.subplots(2, 2, sharex=True)
        plt.subplots_adjust(right = 1.7, bottom=.1,top=.9, wspace = .3, hspace=.2)
        axs[0,0].plot(self.resource_over_time,color='darkgreen')
        axs[0,0].set(title = 'Resource levels and mean id over time',ylabel ='resource')
        
        axs[0,1].plot(self.wealth,color = 'midnightblue')
        axs[0,1].set(title = 'Wealth and mean ac over time',ylabel ='wealth of agents')
        
        axs[1,0].plot(self.id_over_time,color='royalblue')
        axs[1,0].set(ylim=([-0.1,1.1]),ylabel='mean id',xlabel='steps')

        axs[1,1].plot(self.ac_over_time, color='firebrick')
        axs[1,1].set(ylim=([-0.1,1.1]),ylabel ='mean ac',xlabel='steps')
                
        fig, axs = plt.subplots(1, 3)
        plt.subplots_adjust(left=.2,right=3, wspace=.1)      
        axs[0].hist(self.accounts,color='silver')
        axs[0].set(ylabel = 'PDF',xlabel ='distribution of wealth')

        axs[1].hist(self.identification,bins=20,range=(0,1),color='lightsteelblue')
        axs[1].set(xlabel='distribution of ID',title='Final distribution of wealth, ID, and AC')

        axs[2].hist(self.affective_commitment,bins=20,range=(0,1), color='lightcoral')
        axs[2].set(xlabel ='distribution of AC')
        plt.show()
        
        print("Final resource level:",round(self.resource_stock,2))
        print("Final average wealth:",round(self.wealth[-1],2))
        print("Final mean identification:",round(self.id_over_time[-1],2))
        print("Final mean affective commitment:",round(self.ac_over_time[-1],2))
        print("Sanctions per round per person:",round(self.sanctions/(self.network_size*self.step),2))
    
    def use_resource(self):
        '''
        The agent uses the resource according to their social norms and their 
        affective commitment towards the group.
        
        Their current account changes by how much they used the resource minus their
        current need, which they used up and cannot keep. If the use is smaller than 
        the need, they lose wealth or go into debt. 
        '''
        for node in self.graph.nodes:
            #the dictator game for each agent's resource use in each round
            expected_norm = []
            for _ in range(1000):
                expected_norm.append(np.random.choice(self.lambdas,p=self.social_norms[node]/np.sum(self.social_norms[node])))
            self.graph.nodes[node]['current_use'] = 1 - np.mean(expected_norm) * self.affective_commitment[node]
            self.resource_stock += self.rate_of_return - self.graph.nodes[node]['current_use']
            self.accounts[node] += self.graph.nodes[node]['current_use'] - self.needs
            self.graph.nodes[node]['identity_loss'] = 0
            
        #measure the use steps
        self.step += 1
        self.time.append(self.step)
        self.resource_over_time.append(self.resource_stock)
    
    def update_beliefs(self):
        '''
        First, each agent looks at the behavior of the agents they are connected to.
        Norm-violations by neighbors can lead to an affective response,
        if affective commitment is high enough. For a cooperator observing a defector, 
        AC motivates punishment, which will in turn create guilt in the defector 
        and increase their cooperative motivation. Sanctions cost the punisher.

        Second, each agent updates their social norm beliefs based on their observations. 
        
        Third, each agent reflects on the fairness of appropriation in their group
        by comparing their appropriation to that of those around them, which updates
        their identification with group goals. The update in identification also creates 
        changes in affective commitment.        
        '''
        for node in self.graph.nodes:
            
            #observing the current use of neighbors
            neighbors = list(self.graph.neighbors(node))
            cooperators = 0
            use_differences = []
            norms_distribution = self.social_norms[node]/np.sum(self.social_norms[node])
            for i in neighbors:     
                #observe differences between own use and neighbor
                use_difference = self.graph.nodes[i]['current_use'] - self.graph.nodes[node]['current_use']
                use_differences.append(use_difference)
                
                #find cooperators and defectors compared to own social norms and adjust their reputation
                if use_difference < 0:
                    cooperators += 1
                    self.graph.nodes[node]['reputation'][i] += 1
                    continue
                if norms_distribution[int(self.graph.nodes[i]['current_use'] * 100)] > 0.05:
                    cooperators += 1
                    self.graph.nodes[node]['reputation'][i] += 1
                    continue
                self.graph.nodes[node]['reputation'][i] -= 1
                
                #if affective commitment is high enough, punish the defectors, which produces a loss of sanctioning and an increase in AC
                if self.identity_mechanism and random.random() < self.affective_commitment[node]:
                    self.graph.nodes[i]['identity_loss'] += 1
                    self.affective_commitment[i] += (1 - self.affective_commitment[i]) / (1 / self.sanction_cost)
                    self.accounts[node] -= self.sanction_cost
                    self.sanctions += 1
                    continue
                else: #if viewing defector but do not respond, incur a loss in AC
                    self.affective_commitment[node] -= self.affective_commitment[node] / (1 / self.sanction_cost)
            
                #reduce subjective reputation of defectors and punish them if it is too low
                if self.reputation_mechanism and self.graph.nodes[node]['reputation'][i] < 0:
                    self.graph.nodes[i]['identity_loss'] += 1
                    self.accounts[node] -= self.sanction_cost
                    self.sanctions += 1
            
            #Bayesian social norm updating
            observations = np.full(len(neighbors),int((cooperators+self.graph.nodes[node]['identity_loss'])*self.identification[node]))
            posterior = self.compute_posterior(self.lambdas, self.social_norms[node], self.likelihood, observations)    
            self.social_norms[node] = posterior
            
            #update identification based on differences to others, fairness and coverage of needs
            var_observations = np.array([int(abs(u) * len(neighbors) + np.mean(use_differences)) + 1 if (self.needs - self.graph.nodes[node]['current_use'] > 0) else 0 for u in use_differences])            
            var_posterior = self.compute_posterior(self.lambdas, self.social_norms_variance[node], self.likelihood, var_observations)    
            self.social_norms_variance[node] = var_posterior
            
            expected_var = []
            for _ in range(1000):
                expected_var.append(np.random.choice(self.lambdas,p=self.social_norms_variance[node]/np.sum(self.social_norms_variance[node])))
            self.identification[node] = 1 - np.mean(expected_var) 
            
            #update affective commitment based on current identification
            if self.identity_mechanism:
                if random.random() < self.p_interaction:
                    self.affective_commitment[node] += (1-self.affective_commitment[node]) / 2
                
            if self.reputation_mechanism:
                if self.identification[node] - 0.5 > 0:
                    self.affective_commitment[node] += (self.identification[node]-0.5) * (1-self.affective_commitment[node]) / 2
                else:
                    self.affective_commitment[node] += (self.identification[node]-0.5) * self.affective_commitment[node] / 2

        #record changes in wealth, identification and affective commitment after the current round
        self.wealth.append(np.mean(self.accounts))
        self.id_over_time.append(np.mean(self.identification))
        self.ac_over_time.append(np.mean(self.affective_commitment))
    
    
    def simulate(self,trials):
        '''
        Runs simulations of the resource system.
        '''
        self.initialize()
        for i in range(trials):
            self.use_resource()
            self.update_beliefs()
        if self.markup:
            self.show_results()
