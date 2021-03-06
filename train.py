from tqdm import tqdm
import tensorflow as tf
from utils import process_screen
import threading
import os
from imageio import mimsave

def process_gradients(actor_grads, critic_grads, tn):
    grads = []
    for a, c in zip(actor_grads, critic_grads):
        grad = a + tn.critic_coefficient * c
        if tn.gradient_clipping != None:
            grad = tf.clip_by_norm(grad, tn.gradient_clipping)
        grads.append(grad)
    return grads

def manage_network_update(actor_loss, critic_loss, tn, tape):
    actor_grads = tape.gradient(actor_loss, tn.actor_critic.trainable_variables, unconnected_gradients = 'zero')
    critic_grads = tape.gradient(critic_loss, tn.actor_critic.trainable_variables, unconnected_gradients = 'zero')
    grads = process_gradients(actor_grads, critic_grads, tn)
    tn.optimizer.apply_gradients(zip(grads, tn.actor_critic.trainable_variables))
    del tape

def worker_process(tn, thread_number):
    environment = tn.environments[thread_number]
    parameter_updates, update_counter, last_update, episode_count = (0, 0, 0, 0)
    episode_reward = 0.0
    state = environment.reset()
    if tn.render and thread_number == 0:
        environment.render()
    if thread_number == 0:
        images = [state]
    tn.actor_critic.reset_thread_states(thread_number)
    state = process_screen(state)
    while True:
        actor_loss, critic_loss = 0.0, 0.0
        with tf.GradientTape(persistent = True) as tape:
            update_point = False
            while not update_point:
                actor_policy, critic_value = tn.actor_critic(state, thread_number)
                action = tf.squeeze(tf.random.categorical(tf.math.log(actor_policy), 1))
                new_state, reward, done, _ = environment.step(tf.stop_gradient(action))
                # print(reward)
                if tn.render and thread_number == 0:
                    environment.render()
                if thread_number == 0:
                    images.append(new_state)
                episode_reward += reward
                new_state = process_screen(new_state)

                if done:
                    print('Thread {} Episode done'.format(thread_number))
                    episode_count += 1

                    if thread_number == 0:
                        with tn.summary_writer.as_default():
                            tf.summary.scalar('average-episode-reward', episode_reward, step = episode_count)
                        if tn.checkpoint_save_interval != None and episode_count % tn.checkpoint_save_interval == 0:
                            tn.actor_critic.save_weights(os.path.join(tn.checkpoint_dir, 'AC_' + str(episode_count)))
                        if tn.gifs_save_interval != None and episode_count % tn.gifs_save_interval == 0:
                            print('Saving GIF')
                            mimsave(os.path.join(tn.gifs_dir, 'AC_' + str(episode_count)) + '.gif', images, duration = 0.1)
                        images = []

                    episode_reward = 0.0
                    target_value = reward
                    advantage = target_value - critic_value
                    state = environment.reset()
                    if tn.render and thread_number == 0:
                        environment.render()
                    if thread_number == 0:
                        images = [state]
                    tn.actor_critic.reset_thread_states(thread_number)
                    state = process_screen(state)
                else:
                    target_value = reward + tn.gamma * tf.stop_gradient(tn.actor_critic(new_state, thread_number)[1])
                    advantage = target_value - critic_value
                    state = new_state

                actor_loss -= tf.stop_gradient(advantage) * tf.math.log(actor_policy[0][action] + 1e-5)
                critic_loss += tf.square(advantage)
                update_counter += 1

                if (update_counter - thread_number + tn.update_intervals) % tn.update_intervals == 0:
                    update_point = True

        print('Update {} by thread {}'.format(parameter_updates, thread_number))
        manage_network_update(actor_loss = actor_loss, critic_loss = critic_loss, tn = tn, tape = tape)
        parameter_updates += 1
        tn.global_update_counter += 1
        print('Global updates: ' + str(tn.global_update_counter))




def run_training_procedure(tn): # tn is training_namespace
    tn.total_reward = 0.0
    tn.global_update_counter = 0
    # get threads to work
    thread_list = [threading.Thread(target = worker_process, args = (tn, i)) for i in range(1, tn.threads)]
    for thread in thread_list:
        thread.start()
    worker_process(tn, 0)
