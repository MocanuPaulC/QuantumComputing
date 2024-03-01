# from keras import Model, Sequential
# from keras.src.layers import Dense, Activation, Dropout, Flatten
# from keras.src.optimizers.adam import Adam
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from QRL.agent.percept import Percept
from QRL.agent.episode import Episode
from QRL.environment.environment import Environment
from QRL.learning.learningstrategy import LearningStrategy
from QRL.learning.approximate.quantum_model import get_quantum_neural_network


class DeepQLearning(LearningStrategy):
    """
    Two neural nets model en target_model are trained together and used to predict the best action.
    These nets are denoted as model and target_model in the pseudocode.
    This class is INCOMPLETE.
    """
    count_next_actions = 0

    # decay rate = λ
    # discount factor = γ

    def __init__(self, environment: Environment, hidden_size=50, batch_size=128, α=0.7, ddqn=False, λ=0.0005, γ=0.99,
                 t_max=1000,
                 ϵ_min=0.0005,
                 ϵ_max=1.0) -> None:
        super().__init__(environment, α, λ, γ, t_max, ϵ_min=ϵ_min, ϵ_max=ϵ_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        print("Action Space: {}".format(environment.action_space))
        print("State space: {}".format(environment.observation_space))
        self.q1 = get_model(
            input_size=environment.observation_space.shape,
            output_size=environment.action_space.n,
            hidden_size=hidden_size
        )
        self.q2 = get_model(
            input_size=environment.observation_space.shape,
            output_size=environment.action_space.n,
            hidden_size=hidden_size
        )
        self.q2.set_weights(self.q1.get_weights())

        self.counter = 0
        self.C = 100  # update target network every C steps
        self.replay_memory = deque(maxlen=50_000)

    def on_episode_start(self):
        self.t = 0
        pass

    def next_action(self, state):
        """ Epsilon greedy policy """
        randomvalue = random.random()
        if randomvalue <= self.ϵ:
            # Explore
            action = self.env.action_space.sample()
        else:
            # Exploit best known action
            # model dims are (batch, env.observation_space.n)
            encoded = state
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = self.q1.predict(encoded_reshaped, verbose=0).flatten()
            action = np.argmax(predicted)
        return action

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        # Add last percept of episode to replay memory
        self.replay_memory.append(episode.percepts(1)[-1])
        # If replay memory is full, sample batch and learn from it
        if len(self.replay_memory) > self.batch_size and self.t % 10 == 0:
            # indices = np.random.choice(range(len(done_history)), size=batch_size)

            percepts = random.sample(self.replay_memory, self.batch_size)
            self.learn_from_batch(percepts)

        super().learn(episode)
        pass

    def learn_from_batch(self, percepts):
        states, q_values = self.build_training_set(percepts)
        self.train_network(states, q_values)
        self.counter += 1
        if self.counter % self.C == 0:
            print("Transer weights")
            self.counter = 0
            self.q2.set_weights(self.q1.get_weights())

    def get_loss(self):
        """
        Calculate the loss using the .evaluate() function.
        """
        if len(self.replay_memory) < self.batch_size:
            # Not enough samples to evaluate
            return None

        # Randomly sample a batch of percepts from the replay memory
        percepts = random.sample(self.replay_memory, self.batch_size)

        # Prepare states and target Q-values
        states, target_q_values = self.build_training_set(percepts)

        # Calculate loss using the .evaluate() function
        loss = self.q1.evaluate(np.array(states), np.array(target_q_values), verbose=0)

        return loss

    def build_training_set_for_loop(self, percepts: [Percept]):
        state_list = []
        action_list = []
        for percept in percepts:
            state = percept.state
            action = percept.action
            reward = percept.reward
            next_state = percept.next_state
            done = percept.done

            q_values = self.q1.predict(state, verbose=0)
            future_q_values = max(self.q2.predict(next_state, verbose=0))

            if done:
                max_future_q = reward
            else:
                max_future_q = reward + self.γ * np.max(future_q_values)

            q_values[action] = max_future_q

            state_list.append(state)
            action_list.append(q_values)

        return state_list, action_list

    def build_training_set(self, percepts: [Percept]):
        """ Build training set from episode """

        state_list = []
        action_list = []

        states = np.array([percept.state for percept in percepts])

        q_values = self.q1.predict(states, verbose=0)

        next_states = np.array([percept.next_state for percept in percepts])

        future_q_values = self.q2.predict(next_states, verbose=0)

        a_star_values = None
        if self.ddqn:
            a_star_values = self.q1.predict(next_states, verbose=0, batch_size=self.batch_size)

        for i, percept in enumerate(percepts):
            if self.ddqn:
                a_star = np.argmax(a_star_values[i])
                future_q_value = future_q_values[i][a_star]
            else:
                future_q_value = max(future_q_values[i])

            if percept.done:
                max_future_q = percept.reward
            else:
                max_future_q = percept.reward + self.γ * future_q_value

            current_qs = q_values[i]
            current_qs[percept.action] = (1 - self.α) * current_qs[percept.action] + self.α * max_future_q

            state_list.append(percept.state)
            action_list.append(current_qs)

        return state_list, action_list

    def train_network(self, states, q_values):
        # training_set=np.array(training_set)
        self.q1.fit(np.array(states), np.array(q_values), batch_size=self.batch_size, verbose=0)


def get_model(input_size, hidden_size,output_size, loss='mse'):
    pass
    # model = Sequential()
    # model.add(Dense(units=hidden_size, input_shape=input_size,
    #                 activation='relu'))
    # model.add(Dense(units=hidden_size * 2, activation='relu'))
    # model.add(Dense(units=output_size, activation='linear'))
    # model.compile(loss=loss, optimizer=optimizer)

    # return model


class TorchDeepQLearning(LearningStrategy):
    """
    Two neural nets model en target_model are trained together and used to predict the best action.
    These nets are denoted as model and target_model in the pseudocode.
    This class is INCOMPLETE.
    """


    count_next_actions = 0

    def __init__(self, environment: Environment, hidden_size=50, batch_size=128, α=0.7, ddqn=False, λ=0.0005, γ=0.99,
                 t_max=1000,
                 ϵ_min=0.0005, ϵ_max=1.0) -> None:
        super().__init__(environment, α, λ, γ, t_max, ϵ_min=ϵ_min, ϵ_max=ϵ_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        print(f"Action Space: {environment.action_space.n}")
        print(f"State space: {environment.state_size}")
        self.q1 = get_quantum_neural_network(n_qubits=environment.state_size,n_layers=6, enable_rx=False, enable_ry=True, enable_rz=True)
        self.q2 = get_quantum_neural_network(n_qubits=environment.state_size,n_layers=6, enable_rx=False, enable_ry=True, enable_rz=True)
        self.q2.load_state_dict(self.q1.state_dict())

        self.counter = 0
        self.C = 100  # update target network every C steps
        self.replay_memory = deque(maxlen=50_000)

        self.optimizer = optim.Adam(self.q1.parameters(), lr=λ)
        self.criterion = nn.SmoothL1Loss()

    def on_episode_start(self):
        self.t = 0
        pass

    def next_action(self, state):
        """ Epsilon greedy policy """
        randomvalue = random.random()
        if randomvalue <= self.ϵ:
            # Explore
            action = self.env.action_space.sample()
        else:
            # Exploit best known action
            print("state",state)
            # state_tensor = torch.tensor(state)
            with torch.no_grad():
                print("state_tensor",state)
                predicted = self.q1(state)
            action = predicted.argmax().item()
        return action

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        self.replay_memory.append(episode.percepts(1)[-1])
        if len(self.replay_memory) > self.batch_size and self.t % 10 == 0:
            percepts = random.sample(self.replay_memory, self.batch_size)
            self.learn_from_batch(percepts)

        super().learn(episode)
        pass

    def get_loss(self):
        if len(self.replay_memory) < self.batch_size:
            # Not enough samples to evaluate
            return None

        percepts = random.sample(self.replay_memory, self.batch_size)
        states, target_q_values = self.build_training_set(percepts)

        states_tensor = torch.FloatTensor(states)
        target_q_values_tensor = torch.FloatTensor(target_q_values)

        with torch.no_grad():
            predicted_q_values = self.q1(states_tensor)
            loss = self.criterion(predicted_q_values, target_q_values_tensor)

        return loss.item()

    def learn_from_batch(self, percepts):
        states, q_values = self.build_training_set(percepts)
        states_tensor = torch.FloatTensor(states)
        q_values_tensor = torch.FloatTensor(q_values)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.q1(states_tensor)
        loss = self.criterion(outputs, q_values_tensor)

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        self.counter += 1
        if self.counter % self.C == 0:
            print("Transfer weights")
            self.q2.load_state_dict(self.q1.state_dict())

    def build_training_set(self, percepts):
        states = np.array([percept.state for percept in percepts])
        next_states = np.array([percept.next_state for percept in percepts])

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)

        with torch.no_grad():
            q_values = self.q1(states_tensor)
            next_q_values = self.q2(next_states_tensor)
            if self.ddqn:
                next_q_actions = self.q1(next_states_tensor).max(1)[1].unsqueeze(1)
                next_q_values = next_q_values.gather(1, next_q_actions).squeeze(1)

        for i, percept in enumerate(percepts):
            q_update = percept.reward
            if not percept.done:
                q_update += self.γ * next_q_values[i].max().item()
            q_values[i][percept.action] = q_update

        return states, q_values.numpy()





def get_torch_model(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size * 2),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, output_size)
    )
    return model
