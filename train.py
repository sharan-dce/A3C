from tqdm import tqdm
import tensorflow as tf
from utils import process_screen
import threading


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



def run_training_procedure(tn): # tn is training_namespace
    # get threads to work
    thread_list = [threading.Thread(target = test_worker, args = (tn, i)) for i in range(tn.threads)]

    for epoch in range(tn.epochs):
        # for episode_batch in tqdm(range(tn.episodes_per_epoch), desc = 'Epoch {:10d}'.format(epoch)):
        #     run_episode_batch(tn)

        for test_run_no in tqdm(range(tn.tests_per_epoch), desc = 'Test {:11d}'.format(epoch)):
            tn.total_reward = 0.0
            # thread_list = [threading.Thread(target = test_worker, args = (tn, i)) for i in range(tn.threads)]
            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()
        print(tn.total_reward / (tn.tests_per_epoch * tn.threads))
