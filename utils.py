from collections import deque
import os
import torch
import numpy as np
from ddpg_agent import Agent
from ddpg_agent import ReplayBuffer
import matplotlib.pyplot as plt

def moving_average(sig, n=100):
    """ moving average over 100 episodes """
    window = deque(maxlen=n)  # last n scores
    sig_ma = []
    for i in range(len(sig)):
        window.append(sig[i])
        sig_ma.append(np.mean(window))
    return sig_ma

def get_numberedfilename(filename, ext):
    i = 0
    while os.path.exists('{}_{:d}.{}'.format(filename, i, ext)):
        i += 1
    return '{}_{:d}.{}'.format(filename, i, ext)

def plot_actions_episode(actions_epi):
    plt.subplot(121)
    plt.plot(actions_epi[0])
    plt.subplot(122)
    plt.plot(actions_epi[1])
    plt.show()

def save_agentcheckpoint(state_size, action_size, agents):
    ''' save neural network weights of critic and actor as checkpoint  '''
    for i in range(len(agents)):
        checkpoint = {'input_size': state_size+action_size,
                      'output_size': 1,
                      'hidden_layers': [each.out_features for each in agents[i].qnetwork_local.hidden_layers],
                      'state_dict': agents[i].qnetwork_local.state_dict()}
        torch.save(checkpoint, './data/checkpoint_agent_{}_qnetwork_local.pth'.format(i))

        checkpoint = {'input_size': state_size+action_size,
                      'output_size': 1,
                      'hidden_layers': [each.out_features for each in agents[i].qnetwork_local.hidden_layers],
                      'state_dict': agents[i].qnetwork_local.state_dict()}
        torch.save(checkpoint, './data/checkpoint_agent_{}_qnetwork_target.pth'.format(i))

        checkpoint = {'input_size': state_size,
                      'output_size': action_size,
                      'hidden_layers': [each.out_features for each in agents[i].munetwork_local.hidden_layers],
                      'state_dict': agents[i].munetwork_local.state_dict()}
        torch.save(checkpoint, './data/checkpoint_agent_{}_munetwork_local.pth'.format(i))

        checkpoint = {'input_size': state_size,
                      'output_size': action_size,
                      'hidden_layers': [each.out_features for each in agents[i].munetwork_target.hidden_layers],
                      'state_dict': agents[i].munetwork_target.state_dict()}
        torch.save(checkpoint, './data/checkpoint_agent_{}_munetwork_target.pth'.format(i))

def save_agenttrained(state_size, action_size, agents):
    ''' save final neural network weights of critic and actor '''
        for i in range(len(agents)):
        checkpoint = {'input_size': state_size+action_size,
                      'output_size': 1,
                      'hidden_layers': [each.out_features for each in agents[i].qnetwork_local.hidden_layers],
                      'state_dict': agents[i].qnetwork_local.state_dict()}
        torch.save(checkpoint, './data/trained_agent_{}_qnetwork_local.pth'.format(i))

        checkpoint = {'input_size': state_size+action_size,
                      'output_size': 1,
                      'hidden_layers': [each.out_features for each in agents[i].qnetwork_local.hidden_layers],
                      'state_dict': agents[i].qnetwork_local.state_dict()}
        torch.save(checkpoint, './data/trained_agent_{}_qnetwork_target.pth'.format(i))

        checkpoint = {'input_size': state_size,
                      'output_size': action_size,
                      'hidden_layers': [each.out_features for each in agents[i].munetwork_local.hidden_layers],
                      'state_dict': agents[i].munetwork_local.state_dict()}
        torch.save(checkpoint, './data/trained_agent_{}_munetwork_local.pth'.format(i))

        checkpoint = {'input_size': state_size,
                      'output_size': action_size,
                      'hidden_layers': [each.out_features for each in agents[i].munetwork_target.hidden_layers],
                      'state_dict': agents[i].munetwork_target.state_dict()}
        torch.save(checkpoint, './data/trained_agent_{}_munetwork_target.pth'.format(i))

def create_agents(state_size, action_size, num_agents, memory):
    # create agents
    agents= []
    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, 1, memory))
    return agents

def load_agents(state_size, action_size, num_agents, memory):
    # create agents
    agents= []
    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, 1, memory))
    # load checkpoints
    for i in range(len(agents)):
        checkpoint = torch.load('./data/checkpoint_agent_{}_qnetwork_local.pth'.format(i))
        agents[i].qnetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_qnetwork_target.pth'.format(i))
        agents[i].qnetwork_target.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_munetwork_local.pth'.format(i))
        agents[i].munetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_munetwork_target.pth'.format(i))
        agents[i].munetwork_target.load_state_dict(checkpoint['state_dict'])
    return agents

def load_trainedagents(state_size, action_size, num_agents, memory):
    # create agents
    agents= []
    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, 1, memory))
    # load checkpoints
    for i in range(len(agents)):
        checkpoint = torch.load('./data/trained_agent_{}_qnetwork_local.pth'.format(i))
        agents[i].qnetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_qnetwork_target.pth'.format(i))
        agents[i].qnetwork_target.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_munetwork_local.pth'.format(i))
        agents[i].munetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_munetwork_target.pth'.format(i))
        agents[i].munetwork_target.load_state_dict(checkpoint['state_dict'])
    return agents

def load_networks(agents):
    # load checkpoints
    for i in range(len(agents)):
        checkpoint = torch.load('./data/checkpoint_agent_{}_qnetwork_local.pth'.format(i))
        agents[i].qnetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_qnetwork_target.pth'.format(i))
        agents[i].qnetwork_target.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_munetwork_local.pth'.format(i))
        agents[i].munetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/checkpoint_agent_{}_munetwork_target.pth'.format(i))
        agents[i].munetwork_target.load_state_dict(checkpoint['state_dict'])

def load_trainednetworks(agents):
    # load checkpoints
    for i in range(len(agents)):
        checkpoint = torch.load('./data/trained_agent_{}_qnetwork_local.pth'.format(i))
        agents[i].qnetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_qnetwork_target.pth'.format(i))
        agents[i].qnetwork_target.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_munetwork_local.pth'.format(i))
        agents[i].munetwork_local.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./data/trained_agent_{}_munetwork_target.pth'.format(i))
        agents[i].munetwork_target.load_state_dict(checkpoint['state_dict'])
        
def print_info(i_episode, scores_window, steps_window):
    print(('\rEpisode {}\t Score Min/Mean/Max: {:.2f} / ' + color.BOLD + '{:.2f}' + color.END + 
           ' / {:.2f} \t Steps Min/Mean/Max: {} / ' + color.BOLD + '{:.1f} ' + color.END + '/ {}')
          .format(i_episode, np.min(scores_window), np.mean(scores_window), np.max(scores_window),
          np.min(steps_window), np.mean(steps_window), np.max(steps_window)), end="")
        
class color:
    ''' for printing in color, bold, ... '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
  
