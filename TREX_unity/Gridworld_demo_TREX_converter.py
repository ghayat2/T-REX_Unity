"""
This script loads, cleans and converts the Gridworld demonstrations to a TREX compatible format
(Also provided in ml-agents/mlagents/trainers/Gridworld_demo_converter.py)
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

sequence_length = 400
n_demonstrations = 20

for model in ['40', '80', '120', '160', '200', '240', '280', '320', '360', '400', '440', '480', '520', '560', '600']:

    print("Processing model_" + model)
    file_path = str(
        os.getcwd()) + "/../../../../ml-agents-master/ml-agents-master/UnitySDK/Assets/Demonstrations/demonstration" + model + ".demo"
    _, demo_buffer, _ = demo_loader.demo_to_buffer(file_path, sequence_length)

    all_eps_observations = np.array(
        demo_buffer.update_buffer['visual_obs0'])[:sequence_length * n_demonstrations, :, :, :]

    all_eps_rewards = np.array(
        demo_buffer.update_buffer['rewards'])[:sequence_length * n_demonstrations]

    val = np.random.randint(0, n_demonstrations)

    for demonstration in range(n_demonstrations):
        i = 0
        while not np.any(all_eps_observations[demonstration * sequence_length + i, :, :, :]):
            i += 1

        ep_observation = all_eps_observations[
                         demonstration * sequence_length + i: (demonstration + 1) * sequence_length, :, :, :]
        ep_reward = all_eps_rewards[
                    demonstration * sequence_length + i: (demonstration + 1) * sequence_length]
        cumulative_reward = ep_reward.sum()

        print("Model_", model, " ---> demonstration_", demonstration, ": ", sequence_length - i, " steps with reward: ", cumulative_reward)

        if demonstration == val:
            np.savez(os.path.join(val_samples_dir, 'Step_%s_Ep_%02d_Reward_%.2f' % (int(model), demonstration,
                                                                                    cumulative_reward)),
                     states=ep_observation)
        else:
            np.savez(os.path.join(train_samples_dir, 'Step_%s_Ep_%02d_Reward_%.2f' % (int(model), demonstration,
                                                                                      cumulative_reward)),
                     states=ep_observation)

        print("Processed demonstration ", demonstration, "of size", np.shape(ep_observation))
