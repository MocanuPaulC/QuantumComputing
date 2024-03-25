import gymnasium
import numpy as np
import torch
import quantum_model as qm
import concurrent.futures


def demo(trained: bool):
    rewards = []

    if trained:
        model = qm.get_quantum_neural_network(4, 8)
        model.load_state_dict(torch.load("./best_average_score2.pth"), strict=True)
        print(model.state_dict())
    else:
        model = qm.get_quantum_neural_network(4, 8)
    env = gymnasium.make("CartPole-v1", render_mode="human")
    for episode in range(10000):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for step in range(80000):
            epsilon = 0
            obs, reward, done, truncated, info = qm.play_one_step(env, obs, epsilon, model)
            if isinstance(obs, tuple):
                obs = obs[0]
            if done:
                rewards.append(step)
                break

    print(f"Average reward {'trained' if trained else 'untrained'}:", np.mean(rewards))


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(demo, trained=True), executor.submit(demo, trained=False)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
