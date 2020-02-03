from tqdm import tqdm
import tensorflow as tf
from utils import process_screen
import threading

def test_worker_no_threading(tn, epoch, test_run_no):
    states = list(map(lambda env: process_screen(env.reset()), tn.environments))
    done = [False for _ in range(tn.threads)]
    if tn.render_testing:
        [env.render() for env in tn.environments]
    try:
        tn.actor_critic.reset_states()
    except:
        pass

    while True:
        # print(tf.convert_to_tensor(states).dtype)
        output_namespace = tn.actor_critic(tf.convert_to_tensor(states))
        actor_policy, critic_value = output_namespace.actor, output_namespace.critic
        actions = tf.squeeze(tf.random.categorical(actor_policy, 1))

        step_results = [env.step(action) if not done_ else (None, None, True, None) for env, action, done_ in zip(tn.environments, actions, done)]

        if tn.render_testing:
            [env.render() for env in tn.environments]

        states = [process_screen(step_result[0]) if not done_ else tf.zeros(tn.environments[0].observation_space.shape) for step_result, done_ in zip(step_results, done)]
        rewards = [step_result[1] if not done_ else None for step_result, done_ in zip(step_results, done)]
        done = [step_result[2] for step_result in step_results]

        if not False in done:
            break


def test_worker(tn, actor_critic, target_network):
    states = list(map(lambda env: process_screen(env.reset()), tn.environments))
    done = [False for _ in range(tn.threads)]
    if tn.render_testing:
        [env.render() for env in tn.environments]
    try:
        tn.actor_critic.reset_states()
    except:
        pass

    while True:
        # print(tf.convert_to_tensor(states).dtype)
        output_namespace = tn.actor_critic(tf.convert_to_tensor(states))
        actor_policy, critic_value = output_namespace.actor, output_namespace.critic
        actions = tf.squeeze(tf.random.categorical(actor_policy, 1))

        step_results = [env.step(action) if not done_ else (None, None, True, None) for env, action, done_ in zip(tn.environments, actions, done)]

        if tn.render_testing:
            [env.render() for env in tn.environments]

        states = [process_screen(step_result[0]) if not done_ else tf.zeros(tn.environments[0].observation_space.shape) for step_result, done_ in zip(step_results, done)]
        rewards = [step_result[1] if not done_ else None for step_result, done_ in zip(step_results, done)]
        done = [step_result[2] for step_result in step_results]

        if not False in done:
            break

def run_training_procedure(tn): # tn is training_namespace
    for epoch in range(tn.epochs):
        for episode_batch in tqdm(range(tn.episodes_per_epoch), desc = 'Epoch {:10d}'.format(epoch)):
            run_episode_batch(tn)

        for test_run_no in tqdm(range(tn.tests_per_epoch), desc = 'Test {:11d}'.format(epoch)):
            test_run(tn, epoch, test_run_no)
