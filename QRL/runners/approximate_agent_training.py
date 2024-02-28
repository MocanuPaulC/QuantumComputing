from QRL.agent.agent import QuantumAgent
from QRL.environment.openai import FrozenLakeEnvironment, CartPoleEnvironment
from QRL.learning.approximate.deep_qlearning import DeepQLearning
import pickle

if __name__ == '__main__':
    # example use of the code base
    environment = CartPoleEnvironment(render=True)

    deep_q_learning= DeepQLearning(environment,hidden_size=50, batch_size=128,α=0.1, λ=0.01, γ=0.618, t_max=1000, ϵ_min=0.000001, ϵ_max=1)
    # print(environment.map)

    agent = QuantumAgent(render=False, environment=environment, learning_strategy=deep_q_learning, n_episodes=500)  # Set a reasonable number of episodes for evaluation
    agent.train()

    # save the agent to pickle format
    with open('agent.pickle', 'wb') as f:
        pickle.dump(agent, f)


