**System and agents**

Simulating a complex system with ecological, social, economic, and cognitive variables requires making some simplifications and assumptions, which follow from the literature. The model looks at a resource-using community with clear boundaries to its users, as Ostrom (1990) suggests. There are several micro-situational variables that are set to fixed or initial levels. These include:
- Network size, which is bounded at $ N = 20 $, to implement a community small enough to show cooperation without system-wide communication, as predicted in Aflagah et al. (2019). 
- Resource stock, which starts at level $ R = 100 $
- The rate of return $ r_i $, denoting the natural increase in the resource per community member at each time period. For sake of simplicity, the model assumes $r_i=1$ resource unit. As a result, the resource naturally increases at the rate $ R(t) = R(t-1) + N r_i $

Following the cognitive-behavioral framework of CPR governance, each agent holds a number of beliefs which produce behavior. The core variables encoding these beliefs are:
- Social norms distribution of expected cooperation $ p (c) $
- Affective commitment towards cooperation $ A $
- Identification with the community $ Q $

**Note to reader**

The current implementation does not yet implement a fully fleshed-out reputation mechanism and can thus not compare it to the social norm mechanism. Furthermore, it does not yet include a formula for personal interaction and its effects on affective commitment.

import matplotlib.pyplot as plt
%matplotlib inline
import networkx as nx
import numpy as np
import random
from scipy import stats
import scipy as sp

class CPRSim:
    '''
    Runs simulations of a specified number of agents using the resource according to
    the use rules.
    '''
    def __init__(self, network = 'BA', needs = 1, ID = 0.5, AC = 0.5, neighborhood_size = 5, prosociality = .6, sanction_cost = .1, p_interaction = 0.1, markup = False):
        '''
        Inputs:              
            network (str) The type of network structure that the community possessses. The 
              simulator can choose between three types: a small-world, undirected Barabasi-Albert 
              graph (BA), a small-world, directed random k-out-degree graph (directed), and 
              a complete network where all agents are connected to each other (complete).
              Default: BA
            
            needs (float) The need for the resource by each agent. The need is a percentage
              of the rate_of_return, and is determined by micro-situational economic and social
              variables, such as family size and availability of resource for living.
              Default: 1
              
            neighborhood_size (float) The number of other agents the average individual is able
              to observe in any given round. Observable agents are called neighbors. In BA networks, 
              the agents' neighborhood sizes are distirbuted with a power law distribution around a 
              mean of 5, while all agents can observe 5 neighbors in the directed network. In the 
              complete network, neighborhood size is irrelevant since all agents are connected.
              Default: 5
            
            prosociality(float) The mean altruistic tendency of the individuals 
              in the community. This parameter is used as the prior for individual social
              norm beliefs.
              Default: 1   
            
            sanction_cost (float) The cost of imposing a sanction on a defector, as a ratio
              of the rate_of_return.
              Default: 0.1
            
            p_interaction(float) The probability of having a social interaction with another
              person during the time period. Social interaction can enhance the agent's 
              sense of belonging and thereby boost affective commitment.
              Default: 0.1
              
            markup (bool) Turns the observation of the results on and off.
              Default: False
        
        Variables:
            network_size (int) The number of agents in the community.
              Default: 50
            
            resource_stock (int) The stock variable of the common resource which all individuals use.
              Default: 100
              
            rate_of_return (float) The rate at which the resource yields returns in 
              one period of time. Depending on the common resource analyzed, this can 
              include the replenishment of a natural resource, the social surplus created 
              by community services, institutions or cultural events, or the network effects 
              that arise from information aggregation in knowledge commons.
              Default: 1

        '''    
        #setting network level variables
        self.network_type = network
        self.network_size = 20
        self.resource_stock = 100
        self.rate_of_return = 1
        self.neighborhood_size = neighborhood_size
        self.prosociality = prosociality
        self.needs = needs
        self.sanction_cost = sanction_cost
        self.p_interaction = p_interaction
        self.markup = markup
        
        #the following arrays initialize and store the agent variables
        self.svo_lambdas = np.linspace(0.01, .99, 100) #guessed social value orientation for Bayesian updating
        self.social_norms = np.empty((self.network_size,100)) #stores social norm distributions
        self.identification = np.full(self.network_size,ID) #stores identification with the group
        self.affective_commitment = np.full(self.network_size, AC) #stores AC towards the group
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
        Initializes the network according to the chosen network type.
        Community members then have a first exchange about their 
        prosociality to initialize their social norm distributions.
        '''
        #initialize the network as a Barabasi Albert graph with a power law node degree distribution
        if self.network_type == 'BA':
            self.graph = nx.barabasi_albert_graph(self.network_size, self.neighborhood_size)
        elif self.network_type == 'complete':
            self.graph = nx.complete_graph(self.network_size)
        elif self.network_type == 'directed':
            self.graph = nx.random_k_out_graph(self.network_size, self.neighborhood_size, alpha= 1, self_loops=False)
 
        #set node parameter of current resource use to their social value orientation
        current_use = np.full(self.network_size,1)
        current_use[0:int(self.network_size * (1-self.prosociality))] = 0.1
        for node in self.graph.nodes:
            self.graph.nodes[node]['current_use'] = current_use[node]
        
        #observing the social value orientation of neighbors and initiating one's social norm expectations
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            norms_distribution = []
            for i in neighbors:
                norms_distribution.append(self.graph.nodes[i]['current_use'])
            norms_distribution = np.array(norms_distribution)* len(norms_distribution)
            observations = norms_distribution
            prior = stats.beta(a=(1.1-self.graph.nodes[node]['current_use']),b=0.8).pdf(self.svo_lambdas)
            posterior = self.compute_posterior(self.svo_lambdas, prior, self.likelihood, observations)    
            self.social_norms[node] = posterior
        
        if self.markup:
            for n in self.social_norms:
                plt.plot(self.svo_lambdas, n, label='social norms')
            plt.title('Initial social norm distributions of all agents in the network')
            plt.xlabel('Expected cooperation')
            plt.ylabel('PDF')
            plt.show()
        
        for edge in self.graph.edges:
            self.graph.edges[edge]['weight'] = 1
        self.layout = nx.spring_layout(self.graph)  # Initial visual layout
    
        
    def draw_network(self):
        '''
        Draw the state of the network.
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
            plt.plot(self.svo_lambdas, n, label='social norms')
        plt.title('Social norm distributions of all agents in the network')
        plt.xlabel('Expected cooperation')
        plt.ylabel('PDF')
        
        fig, axs = plt.subplots(4, 1, sharex=True)
        plt.subplots_adjust(bottom=.3,top=2, hspace=.2)
        axs[0].plot(self.resource_over_time,color='darkgreen')
        axs[0].set(title = 'Resource levels, wealth, mean id and ac over time',ylabel ='resource')
        
        axs[1].plot(self.wealth,color = 'midnightblue')
        axs[1].set(ylabel ='wealth of agents')
        
        axs[2].plot(self.id_over_time,color='royalblue')
        axs[2].set(ylim=([-0.1,1.1]),ylabel='mean id')

        axs[3].plot(self.ac_over_time, color='firebrick')
        axs[3].set(ylim=([-0.1,1.1]),ylabel ='mean ac')
        
        fig, axs = plt.subplots(1, 3)
        plt.subplots_adjust(left=.2,right=3, wspace=.1)      
        axs[0].hist(self.accounts,color='silver')
        axs[0].set(ylabel = 'PDF',xlabel ='distribution of wealth')

        axs[1].hist(self.identification,range=(0,1),color='lightsteelblue')
        axs[1].set(xlabel='distribution of ID',title='Final distribution of wealth, ID, and AC')

        axs[2].hist(self.affective_commitment,range=(0,1), color='lightcoral')
        axs[2].set(xlabel ='distribution of AC')

        plt.show()
    
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
            expected_norm = np.random.choice(self.svo_lambdas,p=self.social_norms[node]/np.sum(self.social_norms[node]))
            self.graph.nodes[node]['current_use'] = self.rate_of_return - expected_norm * self.affective_commitment[node]
            self.resource_stock += self.rate_of_return - self.graph.nodes[node]['current_use']
            self.accounts[node] += self.graph.nodes[node]['current_use'] - self.needs
            
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
                use_difference = self.graph.nodes[i]['current_use'] - self.graph.nodes[node]['current_use']
                use_differences.append(use_difference)
                #find cooperators compared to own social norms
                if use_difference < 0:
                    cooperators += 1
                    continue
                if norms_distribution[int(self.graph.nodes[i]['current_use'] * 100)] > 0.05:
                    cooperators += 1
                    continue
                #if affective commitment is high enough, punish the defectors, which increases their AC for the next round
                if random.random() < self.affective_commitment[node]:
                    self.affective_commitment[i] += (1 - self.affective_commitment[i]) / 2
                    self.accounts[node] -= self.sanction_cost
                    self.sanctions += 1
                else: #if viewing defector but do not respond, incur a loss in AC
                    self.affective_commitment[node] -= self.affective_commitment[node] / 2
            
            #Bayesian social norm updating
            observations = np.full(len(neighbors),int(cooperators*self.identification[node]))
            posterior = self.compute_posterior(self.svo_lambdas, self.social_norms[node], self.likelihood, observations)    
            self.social_norms[node] = posterior
            
            #update identification of agent based on fairness of appropriation amongst connections
            self.identification[node] -= np.mean(use_differences)
            if self.identification[node] > .99:
                self.identification[node] = .99
            if self.identification[node] < .01:
                self.identification[node] = .01
            
            #update affective commitment based on current identification
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
            

sim = CPRSim(network = 'directed',needs=0.7,markup=True)
sim.simulate(10)
sim.sanctions
