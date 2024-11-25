from trpo import *

from datetime import datetime

def train_policy_trpo():
    value_learning_rate = 0.005
    hidden_dim = 32

    env = create_halfcheetah_env(render=False, forward_reward_weight=1)

    policy_net = CreatePolicyNet(env, hidden_dim)
    value_net = CreateValueNet(env, hidden_dim, value_learning_rate)

    hyperparams = {"epochs":2000, 
                    "num_episodes":50, 
                    "max_num_steps":200, 
                    "max_d_kl" : 0.01, 
                    "damping_coeff" : 0.1, 
                    "entr_coeff" : 0.05}
    # hyperparams = {"epochs":1000, 
    #                 "num_episodes":10, 
    #                 "max_num_steps":200, 
    #                 "max_d_kl" : 0.01, 
    #                 "damping_coeff" : 0.1, 
    #                 "entr_coeff" : 0.05}

    trpo = TRPO(env, policy_net, value_net, hyperparams) 
    trpo.train_trpo()

    time = datetime.now().strftime("%d_%m_%H_%M_%S")
    torch.save(policy_net, f"trpo_weights/policy_net_weights_{time}.pth")



if __name__ == "__main__":

