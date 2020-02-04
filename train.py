from tqdm import tqdm
import tensorflow as tf
from utils import process_screen
import threading
import os
from imageio import mimsave


def manage_network_update(actor_loss, critic_loss, tn, tape):
    actor_grads = tape.gradient(actor_loss, tn.actor_critic.trainable_variables)
    critic_grads = tape.gradient(critic_loss, tn.actor_critic.trainable_variables)
    tn.optimizer.apply_gradients(zip(actor_grads, tn.actor_critic.trainable_variables))
    tn.optimizer.apply_gradients(zip(critic_grads, tn.actor_critic.trainable_variables))
    del tape

def worker_process(tn, thread_number):
    environment = tn.environments[thread_number]
    parameter_updates, update_counter, last_update, episode_count = (0, 0, 0, 0)
    episode_reward = 0.0
    state = environment.reset()
    if thread_number == 0:
        images = [state]
    tn.actor_critic.reset_thread_states(thread_number)
    tn.target_network.reset_thread_states(thread_number)
    state = process_screen(state)
    while True:
        actor_loss, critic_loss = 0.0, 0.0
        with tf.GradientTape(persistent = True) as tape:
            update_point = False
            while not update_point:
                actor_policy, critic_value = tn.actor_critic(state, thread_number)
                action = tf.squeeze(tf.random.categorical(actor_policy, 1))
                new_state, reward, done, _ = environment.step(action)
                if thread_number == 0:
                    images.append(new_state)
                episode_reward += reward
                new_state = process_screen(new_state)

                if done:
                    if thread_number == 0:
                        episode_count += 1
                        with tn.summary_writer.as_default():
                            print('Writing log info')
                            tf.summary.scalar('average-episode-reward', episode_reward, step = episode_count)
                        if episode_count % tn.checkpoint_save_interval == 0:
                            tn.actor_critic.save_weights(os.path.join(tn.checkpoint_dir, 'AC_' + str(episode_count)))
                        if episode_count % tn.gifs_save_interval == 0:
                            print('Saving GIF')
                            mimsave(exportname = os.path.join(tn.checkpoint_dir, 'AC_' + str(episode_count)), frames = images, format = 'GIF', duration = 0.2)
                        images = []

                    episode_reward = 0.0
                    target_value = reward
                    advantage = target_value - critic_value
                    state = environment.reset()
                    if thread_number == 0:
                        images = [state]
                    tn.actor_critic.reset_thread_states(thread_number)
                    tn.target_network.reset_thread_states(thread_number)
                    state = process_screen(state)
                else:
                    target_value = reward + tn.gamma * tn.target_network(new_state, thread_number)[1]
                    advantage = target_value - critic_value
                    state = new_state

                actor_loss -= advantage.numpy() * tf.math.log(actor_policy[0][action] + 1e-5)
                critic_loss += tf.square(advantage)
                update_counter += 1
                tn.global_update_counter += 1

                if (update_counter - thread_number + tn.threads) % tn.threads == 0:
                    update_point = True

        print('Update {} by thread {}'.format(parameter_updates, thread_number))
        manage_network_update(actor_loss = actor_loss, critic_loss = critic_loss, tn = tn, tape = tape)
        parameter_updates += 1
        if thread_number == 0 and tn.global_update_counter - last_update >= tn.target_update_interval:
            print('Update to target network by thread 0')
            tn.target_network.set_weights(tn.actor_critic.get_weights())
            last_update = tn.global_update_counter




def run_training_procedure(tn): # tn is training_namespace
    tn.total_reward = 0.0
    tn.global_update_counter = 0
    # get threads to work
    thread_list = [threading.Thread(target = worker_process, args = (tn, i)) for i in range(tn.threads)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
