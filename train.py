from tqdm import tqdm
import tensorflow as tf
from utils import process_screen
import threading
import os
import imageio

def test_worker(tn, thread_number):
    environment = tn.environments[thread_number]
    state = process_screen(environment.reset())
    done = False
    tn.actor_critic.reset_thread_states()
    if tn.render_testing:
        environment.render()
    try:
        tn.actor_critic.reset_states()
    except:
        pass

    done = False
    while not done:
        actor_policy, critic_value = tn.actor_critic(state, thread_number)
        action = tf.squeeze(tf.random.categorical(actor_policy, 1))

        state, reward, done, _ = environment.step(action)
        tn.total_reward += reward
        if tn.render_testing:
            environment.render()
        state = process_screen(state)

def worker_process(tn, thread_number):
    environment = tn.environments[thread_number]
    counter = 0
    while True:
        counter += 1
        if counter % tn.gifs_save_interval == 0:
            images = []
        state = environment.reset()
        if counter % tn.gifs_save_interval == 0:
            images.append(state)
        state = process_screen(state)
        done = False
        tn.actor_critic.reset_thread_states()

        try:
            tn.actor_critic.reset_states()
        except:
            pass

        done = False
        while not done:
            actor_policy, critic_value = tn.actor_critic(state, thread_number)
            action = tf.squeeze(tf.random.categorical(actor_policy, 1))

            state, reward, done, _ = environment.step(action)
            tn.total_reward += reward
            if counter % tn.gifs_save_interval == 0:
                images.append(state)
            state = process_screen(state)

        if counter % tn.gifs_save_interval == 0:
            imageio.mimsave(os.path.join(tn.gifs_dir, 'episode-{}_thread-{}.gif'.format(str(counter), thread_number)), images, duration = 0.02)
            images = []


def run_training_procedure(tn): # tn is training_namespace
    # get threads to work
    thread_list = [threading.Thread(target = worker_process, args = (tn, i)) for i in range(tn.threads)]
    tn.total_reward = 0.0
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
