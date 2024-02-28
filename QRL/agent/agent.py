import threading
from abc import abstractmethod

from QRL.agent.episode import Episode
from QRL.agent.percept import Percept
from QRL.environment.environment import Environment
from QRL.learning.approximate.deep_qlearning import DeepQLearning
from QRL.learning.learningstrategy import LearningStrategy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.cum_rewards = 0
        self.win_rate = []
        self.average_ep_length = []
        self.episode_count = [0]
        self.percepts_per_episode = []
        self.loss_per_episode = []
        self.update_counter = 0
        self.count_update_counter = 15

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def learning_strat(self):
        """ Getter for learning_strategy """
        return self.learning_strategy

    @property
    def environment(self):
        """ Getter for environment """
        return self.env

    @property
    def done(self):
        return max(self.episode_count) > self.n_episodes


class QuantumAgent(Agent):

    def __init__(self, render: bool, environment: Environment, learning_strategy: DeepQLearning,
                 n_episodes=500) -> None:
        super().__init__(environment, learning_strategy, n_episodes)
        self.cum_rewards_per_episode = []
        self.ep_cum_rewards = 0
        self.render = render
        self.run_animation()


    @property
    def set_render(self):
        return self.render

    def run_animation(self):
        # Create a thread that runs the _run_animation method
        animation_thread = threading.Thread(target=self._run_animation)
        animation_thread.daemon = True  # Optional: makes the thread exit when the main thread does
        animation_thread.start()

    def _run_animation(self):
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        lines = {
            "reward": axs[0].plot([], [], 'r-')[0],
            "success": axs[1].plot([], [], 'g-')[0],
            "loss": axs[2].plot([], [], 'y-')[0]
        }

        def init():
            axs[0].set_title('Percepts per Episode')
            axs[0].set_xlabel('Episodes')
            axs[0].set_ylabel('Average Reward')
            axs[0].set_xlim(0, 500)
            axs[0].set_ylim(0, 500)

            axs[1].set_title('Average percepts per Episode')
            axs[1].set_xlabel('Episodes')
            axs[1].set_ylabel('Success Rate')
            axs[1].set_xlim(0, 500)
            axs[1].set_ylim(0, 500)

            axs[2].set_title('Loss')
            axs[2].set_xlabel('Episodes')
            axs[2].set_ylabel('Loss')
            axs[2].set_xlim(0, 500)
            axs[2].set_ylim(0, 1)
            return [lines["reward"], lines["success"], lines["loss"]]

        def update(frame):
            print("update")

            if self.update_counter > self.count_update_counter:
                self.count_update_counter = self.update_counter + 15

                ep_count = len(self.episode_count) - 2
                ep = self.episode_count[0:ep_count]
                percepts = self.percepts_per_episode[0:ep_count]
                average_ep_length = self.average_ep_length[0:ep_count]
                loss_per_episode = self.loss_per_episode[0:ep_count]
                print(f"percepts: {percepts}")
                print(f"average_ep_length: {average_ep_length}")
                print(f"loss_per_episode: {loss_per_episode}")
                print(f"ep: {ep}")
                lines["reward"].set_data(ep, percepts)
                lines["success"].set_data(ep, average_ep_length)
                lines["loss"].set_data(ep, loss_per_episode)
            return [lines["reward"], lines["success"], lines["loss"]]

        ani = FuncAnimation(fig, update, init_func=init, interval=1000, repeat=False, blit=True)
        plt.show()

    def train(self) -> None:
        super(QuantumAgent, self).train()
        episode_cum_reward = 0
        steps_to_update_target_model = 0

        # as longs as the agents hasn't reached the maximum number of episodes
        # This is the same as for episode in range(self.n_episodes)
        while not self.done:
            total_training_rewards = 0
            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state = self.env.reset()[0]
            # reset the learning strategy for the new episode
            self.learning_strategy.on_episode_start()

            # while the episode isn't finished by length
            while not self.learning_strategy.done():
                steps_to_update_target_model += 1
                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action
                # step method returns a tuple with values (s', r, terminated, truncated, info)
                if self.render:
                    self.env.render()
                # terminated means the the agent reached a terminal state meaning the episode is over
                # if terminated is True, you need to call reset
                next_state, r, terminated, truncated, info = self.env.step(action)

                # ignore values truncated and info
                percept = Percept((state, action, r, next_state, terminated))

                # add the newly created Percept to the Episode
                episode.add(percept)

                # learn from Percepts in Episode
                self.learning_strategy.learn(episode)

                # update Agent's state
                state = percept.next_state
                # learn from one or more Percepts in the Episode
                # self.learning_strategy.learn(episode)

                self.cum_rewards += percept.reward
                episode_cum_reward += percept.reward
                # break if episode is over and inform learning strategy
                if percept.done:
                    self.learning_strategy.on_episode_end()
                    break

            # end episode
            self.learning_strategy.decay()
            self.episode_count.append(self.episode_count[-1] + 1)
            self.percepts_per_episode.append(len(episode.all_percepts()))
            # print("hat")
            # print(f"percepts: {self.percepts_per_episode}")
            self.loss_per_episode.append(self.learning_strategy.get_loss())

            # get count of all percepts
            count_of_percepts = 0
            for episode in self.episodes:
                count_of_percepts += len(episode.all_percepts())

            self.average_ep_length.append(count_of_percepts / len(self.episodes))
            # print(f"average_ep_length: {self.average_ep_length}")
            self.update_counter += 1
            # print(f"episode count: {self.episode_count[-1]}")
            # print(f"loss: {self.loss_per_episode[-1]}")
            print()

        self.env.close()

