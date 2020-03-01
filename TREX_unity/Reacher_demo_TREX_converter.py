"""
This script oads, cleans and converts the Reacher demonstrations to a TREX compatible format
(Also provided in ml-agents/mlagents/trainers/Reacher_demo_converter.py)
"""

import os
import numpy as np
import demo_loader

train_samples_dir = './samples/train_data'
val_samples_dir = './samples/val_data'

if not os.path.isdir(train_samples_dir):
    os.makedirs(train_samples_dir)
if not os.path.isdir(val_samples_dir):
    os.makedirs(val_samples_dir)

sequence_length = 500
n_demonstrations = 10

for model in ['5', '20', '35', '50', '65', '80', '100', '115', '145', '160', '180', '200', '210', '230', '250', '260', '270']:

    print("Processing model_" + model)
    file_path = str(
        os.getcwd()) + "/../../../UnitySDK/Assets/Demonstrations/Reacher/demonstrations" + model + ".demo"
    _, demo_buffer, _ = demo_loader.demo_to_buffer(file_path, sequence_length)

    all_eps_observations = np.array(
        demo_buffer.update_buffer['vector_obs'])[:sequence_length * n_demonstrations, :]

    all_eps_rewards = np.array(
        demo_buffer.update_buffer['rewards'])[:sequence_length * n_demonstrations]

    val = np.random.randint(0, n_demonstrations)

    for demonstration in range(n_demonstrations):

        ep_observation = all_eps_observations[
                         demonstration * sequence_length: (demonstration + 1) * sequence_length, :]
        ep_reward = all_eps_rewards[
                    demonstration * sequence_length:(demonstration + 1) * sequence_length]
        cumulative_reward = ep_reward.sum()

        print("Model_", model, " ---> demonstration_", demonstration, ": ", sequence_length, " steps with reward: ", cumulative_reward)

        if demonstration == val:
            np.savez(os.path.join(val_samples_dir, 'Step_%s_Ep_%02d_Reward_%.2f' % (int(model), demonstration,
                                                                                    cumulative_reward)),
                     states=ep_observation)
        else:
            np.savez(os.path.join(train_samples_dir, 'Step_%s_Ep_%02d_Reward_%.2f' % (int(model), demonstration,
                                                                                      cumulative_reward)),
                     states=ep_observation)

        print("Processed demonstration ", demonstration, "of size", np.shape(ep_observation))
