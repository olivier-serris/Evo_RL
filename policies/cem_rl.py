import torch
from salina import instantiate_class,get_class
import random 
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from salina import Agent

import copy

# from run_launcher.cem_rl.debug_pytorch import load_state_dict

class CemRl:

    def __init__(self,cfg) -> None:

        # debug hyper-parameters : 
        self.rl_active = cfg.algorithm.rl_algorithm.active
        self.es_active = cfg.algorithm.es_algorithm.active

        # hyper-parameters: 
        self.pop_size = cfg.algorithm.es_algorithm.pop_size
        self.initial_buffer_size = cfg.algorithm.initial_buffer_size
        self.n_rl_agent = cfg.algorithm.n_rl_agent

        # RL objects:
        self.rl_learner =  get_class(cfg.algorithm.rl_algorithm)(cfg)

        # CEM objects
        actor_weights = self.rl_learner.get_acquisition_actor().parameters()
        self.centroid = copy.deepcopy(parameters_to_vector(actor_weights).detach())
        code_args = {'num_params': len(self.centroid),'mu_init':self.centroid}
        kwargs = {**cfg.algorithm.es_algorithm, **code_args}
        self.es_learner = get_class(cfg.algorithm.es_algorithm)(**kwargs)

        self.pop_weights = self.es_learner.ask(self.pop_size)

        # vector_to_parameters does not seem to work when module are in different processes
        # the transfert agent is used to transfert vector_to_parameters in main thread
        # and then transfert the parameters to another agent in another process.
        self.param_transfert_agent = copy.deepcopy(self.rl_learner.get_acquisition_actor())

    def get_acquisition_actor(self,i) -> Agent:
        actor = self.rl_learner.get_acquisition_actor()
        weight = copy.deepcopy(self.pop_weights[i]) # TODO: check if necessary
        vector_to_parameters(weight,self.param_transfert_agent.parameters())
        actor.load_state_dict(self.param_transfert_agent.state_dict())
        
        return actor

    def update_acquisition_actor(self,actor,i) -> None:
        weight = copy.deepcopy(self.pop_weights[i]) # TODO: check if necessary
        vector_to_parameters(weight,self.param_transfert_agent.parameters())        
        actor.load_state_dict(self.param_transfert_agent.state_dict())

    def train(self,acq_workspaces,n_total_actor_steps,logger) -> None:

        # Fitness of population
        n_actor_all_steps = 0
        fitness = torch.zeros(len(acq_workspaces))
        for i,workspace in enumerate(acq_workspaces):
            n_actor_all_steps += (
                workspace.time_size() - 1
            ) * workspace.batch_size()
            done = workspace['env/done']
            cumulated_reward = workspace['env/cumulated_reward']
            fitness[i] = cumulated_reward[done].mean()
        
        if self.es_active:
            self.es_learner.tell(self.pop_weights,fitness) #  Update CEM
            self.pop_weights = self.es_learner.ask(self.pop_size) # Generate new population

        # RL update : 
        if self.rl_active:
            self.rl_learner.workspace_to_replay_buffer(acq_workspaces)
            if self.rl_learner.replay_buffer.size() < self.initial_buffer_size: # shouldn't access directly to replay buffer 
                return
            selected_actors = random.sample(range(0,self.pop_size),self.n_rl_agent) # take a half of the population at random
            n_step_per_actor = n_actor_all_steps//len(selected_actors)
            for i in selected_actors:
                logger.debug(f"agent {i}")
                weight = copy.deepcopy(self.pop_weights[i]) # TODO: check if copy necessary
                self.rl_learner.set_actor_params(weight)

                self.rl_learner.train_critic_and_actor(n_step_per_actor,n_total_actor_steps,logger)
                n_total_actor_steps+=n_step_per_actor

                # TODO: check if i really need to use transfert agent here ? 
                actor_state_dict = self.rl_learner.get_state_dict()
                self.param_transfert_agent.load_state_dict(actor_state_dict)
                vector_param = torch.nn.utils.parameters_to_vector(self.param_transfert_agent.parameters())
                self.pop_weights[i] = copy.deepcopy(vector_param.detach()) # TODO: check if copy necessary
