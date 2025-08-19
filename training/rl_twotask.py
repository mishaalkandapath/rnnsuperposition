import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.rnn import RNN
from datasets.task_datasets import TwoStepRLEnvironment   
from training.train_utils import SignalManager

class ActorCriticTrainer:
    """
    Trainer class implementing advantage actor-critic algorithm for the two-step RL task.
    """
    
    def __init__(self, model, env, lr=7e-4, lr_init= 0, gamma=0.9, 
                 beta_entropy=0.005, beta_critic=0.05, logger=None, device=None):
        """
        Initialize trainer. Defaults as in Wang et. al (2017)
        
        Args:
            model: RNN model that outputs both policy logits and value estimates
            env: TwoStepRLEnvironment instance
            lr_actor: Learning rate for actor (policy) parameters
            lr_critic: Learning rate for critic (value) parameters  
            gamma: Discount factor
            beta_entropy: Entropy regularization coefficient
            beta_critic: Critic loss coefficient
            max_episode_length: Maximum steps per episode (t_max in pseudocode)
        """
        self.model = model
        self.env = env
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.beta_critic = beta_critic
    
        if model.learn_init:
            self.optimizer = torch.optim.Adam([
                {'params': [p for name, p in model.named_parameters() 
                            if 'initial_states' not in name], 
                'lr': lr},
                
                {'params': model.initial_states.parameters(), 
                'lr': lr_init} 
            ], lr=lr)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.training_stats = defaultdict(list)
        self.logger = logger

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model.to(self.device)
        
    def get_model_outputs(self, input_tensor, hidden_state):
        """
        Get policy logits and value estimate from model.
        Assumes model outputs (batch, seq, features) where features == 4.
        """
        outputs, new_hidden = self.model(input_tensor, hidden_state)
        
        policy_logits = outputs[0, 0, :3]  # Actions: fixation, action1, action2
        value_estimate = outputs[0, 0, 3]
        
        return policy_logits, value_estimate, new_hidden
    
    def select_action(self, policy_logits, inference=False, phase=-1):
        """
        Select action based on policy logits.
        
        Args:
            policy_logits: Logits for each action
            inference: If True, use greedy selection; if False, sample
        """
        if inference and phase >= 0: # only sample compatible actions
            if phase != 1:
                return 0
            else:
                return torch.argmax(policy_logits[:, :, 1:]).item() + 1
        elif inference:        
            return torch.argmax(policy_logits).item()
        else:
            action_probs = F.softmax(policy_logits, dim=0)
            return int(torch.multinomial(action_probs, 1).item())
    
    def select_action_batched(self, policy_logits, phase=-1):
        """
        Select action based on policy logits.
        
        Args:
            policy_logits: Logits for each action
            inference: If True, use greedy selection; if False, sample
        """
        #TODO: currently only for inference
        actions = torch.zeros(policy_logits.size(0)).to(policy_logits.device)
        if phase == 1:
            actions = torch.argmax(policy_logits[:, :, 1:], dim=-1)[:, 0] + 1
        return actions
    
    def compute_returns(self, rewards, values, is_terminal=True):
        """
        Compute discounted returns using bootstrapping from pseudocode.
        
        From pseudocode:
        R = { 0           for terminal s_t
            { V(s_t, θ_v) for non-terminal s_t  //Bootstrap from last state
        
        Then work backwards: R ← r_i + γR for i ∈ {t-1, ..., t_start}
        """
        returns = []
        
        if is_terminal:
            R = 0.0
        else:
            R = values[-1].item() if len(values) > 0 else 0.0
        
        # Work backwards: R ← r_i + γR
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)
            
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def compute_advantage_loss(self, policy_logits_history, actions_history, 
                              returns, values_history):
        """
        Compute actor-critic loss components.
        """
        values = torch.stack(values_history)
        advantages = returns - values
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # Actor loss: -∇_θ log π(a|s; θ) * (R - V(s; θ_v))
        log_probs = []
        entropies = []
        
        for i, logits in enumerate(policy_logits_history):
            dist = torch.distributions.Categorical(logits=logits)
            a = torch.tensor(actions_history[i], dtype=torch.long, device=logits.device)
            log_prob = dist.log_prob(a)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        actor_loss = -(log_probs * advantages.detach()).mean() - self.beta_entropy * entropies.mean()
        
        # Critic loss: (R - V(s; θ_v))^2
        critic_loss = 0.5 * F.mse_loss(values, returns)

        # print(f"Mean ret: {returns.mean()} Std Ret: {returns.std()} Adv Mean :{advantages.mean()} Adv Std: {advantages.std()} Entropy: {entropy.item()}")
        
        return actor_loss, critic_loss, advantages.detach()
    
    def run_episode(self, trials_per_episode=100, inference=False):
        """
        Run a single episode and collect experience.
        
        Args:
            trials_per_episode: Number of trials in this episode
            inference: If True, run inference mode; if False, collect training data
            
        Returns:
            episode_data: Dictionary with trajectory data and statistics
        """
        states_history = []
        actions_history = []
        rewards_history = []
        policy_logits_history = []
        values_history = []
        
        episode_reward = 0
        hidden_state = None
        
        for _ in range(trials_per_episode):
            self.env.reset_trial()
            
            for _ in range(3):  # DELAY_1, GO, DELAY_2
                self.env.get_observation()
                input_vec = self.env.get_input_vector()
                input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                policy_logits, value_estimate, hidden_state = self.get_model_outputs(input_tensor, hidden_state)
                
                action = self.select_action(policy_logits, inference=inference)
                
                self.env.step(action)
                reward = getattr(self.env, 'last_reward', 0)
                
                if not inference:
                    states_history.append(input_vec)
                    actions_history.append(action)
                    rewards_history.append(reward)
                    policy_logits_history.append(policy_logits)
                    values_history.append(value_estimate)
                
                episode_reward += reward
        
        episode_data = {
            'episode_reward': episode_reward,
            'states': states_history,
            'actions': actions_history, 
            'rewards': rewards_history,
            'policy_logits': policy_logits_history,
            'values': values_history
        }
        
        return episode_data
    
    def train_step(self, episode_data):
        """
        Perform one training step using collected episode data.
        """
        if len(episode_data['rewards']) == 0:
            return 0, 0
            
        returns = self.compute_returns(
                                     episode_data['rewards'], 
                                     episode_data['values']
                                    )
        actor_loss, critic_loss, advantages = self.compute_advantage_loss(
            episode_data['policy_logits'],
            episode_data['actions'],
            returns,
            episode_data['values']
        )
        
        self.optimizer.zero_grad()
        (actor_loss + self.beta_critic * critic_loss).backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, num_episodes=15000, trials_per_episode=100, 
              eval_every=-1, eval_episodes=10, run_name="models"):
        """
        Main training loop following the advantage actor-critic algorithm.
        
        Args:
            num_episodes: Total number of training episodes (E_max in pseudocode)
            trials_per_episode: Trials per episode
            eval_every: Evaluate model every N episodes
            eval_episodes: Number of episodes for evaluation
        """
        self.model.train()
        
        print(f"Starting training for {num_episodes} episodes...")
        pbar = tqdm(range(num_episodes))
        for episode in pbar:
            self.env.reset_trial(reset_r=True)
            episode_data = self.run_episode(trials_per_episode=trials_per_episode, 
                                          inference=False)
            
            actor_loss, critic_loss = self.train_step(episode_data)
            
            self.training_stats['episode_rewards'].append(episode_data['episode_reward'])
            self.training_stats['actor_losses'].append(actor_loss)
            self.training_stats['critic_losses'].append(critic_loss)
            
            if eval_every > 0 and (episode + 1) % eval_every == 0:
                self.evaluate_model(eval_episodes, trials_per_episode)
            
            eval_range = min(50, episode)
            recent_reward = np.mean(self.training_stats['episode_rewards'][-eval_range:])
            recent_actor_loss = np.mean(self.training_stats['actor_losses'][-eval_range:])
            recent_critic_loss = np.mean(self.training_stats['critic_losses'][-eval_range:])
                
            pbar.set_description(f"R={recent_reward:6.2f}, aloss={recent_actor_loss:6.3f}, closs={recent_critic_loss:6.3f}")

            if self.logger:
                run.log({"avg ep reward":recent_reward, 
                         "Actor loss":recent_actor_loss,
                          "Critic Loss": recent_critic_loss })
        
        torch.save(self.model.state_dict(), f"{run_name}/{run_name.split('/')[-1]}.ckpt")
        return self.training_stats
    
    def evaluate_model(self, num_episodes=300, trials_per_episode=100):
        """
        Evaluate model performance using greedy inference.
        """
        self.model.eval()
        eval_rewards = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                episode_data = self.run_episode(trials_per_episode=trials_per_episode,
                                              inference=True)
                eval_rewards.append(episode_data['episode_reward'])
        
        self.model.train()
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        self.training_stats['eval_rewards'].append(mean_reward)
        self.training_stats['eval_stds'].append(std_reward)
        
        return mean_reward, std_reward
    
    def run_inference(self, episodes=300, trials_per_episode=100):
        """
        Wrapper for pure inference (reusing existing inference logic).
        """
        results = {
            'episode_rewards': [],
            'episode_data': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(episodes):
                episode_data = self.run_episode(trials_per_episode=trials_per_episode,
                                              inference=True)
                results['episode_rewards'].append(episode_data['episode_reward'])
                results['episode_data'].append(episode_data)
        
        results['mean_reward'] = np.mean(results['episode_rewards'])
        results['std_reward'] = np.std(results['episode_rewards'])
        
        return results
    
if __name__ =="__main__":
    import argparse
    import os
    import wandb
    import pickle

    parser = argparse.ArgumentParser(description="Train RL two step meta learner")

    parser.add_argument("--d_hidden", type=int, required=True, help="Hidden layer size")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of RNN layers")
    parser.add_argument("--run_name", type=str, required=True, help="Name of run")
    parser.add_argument("--ctd_from", type=str, default=None)
    parser.add_argument("--gru", action="store_true", help="Use GRU?")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--lr_init", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--beta_entropy", type=float, default=0.001)
    parser.add_argument("--beta_critic", type=float, default=0.1)
    parser.add_argument("--learn_init", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.run_name, exist_ok=True)
    run = wandb.init(
        entity="mishaalkandapath",
        project="rnnsuperpos",
        config={
            "lr": args.lr,
            "lr_init": args.lr_init,
            "gamma": args.gamma,
            "beta_entropy": args.beta_entropy,
            "beta_critic": args.beta_critic,
            "n_hidden": args.d_hidden
        },
    )
    # run = None
    torch.manual_seed(2)

    env = TwoStepRLEnvironment()
    env.get_observation()
    env.reset_trial(reset_r=True)
    in_size = len(env.get_input_vector())
    model = RNN(input_size=in_size, hidden_size=48, out_size=4,
                 out_act=lambda x: x, use_gru=args.gru, learn_init=args.learn_init)
    
    if args.ctd_from:
        model.load_state_dict(torch.load(f"models/rl_train/{args.ctd_from}/{args.ctd_from}.ckpt"))
    
    trainer = ActorCriticTrainer(model, env, 
                                 lr=args.lr, 
                                 lr_init=args.lr_init,
                                 gamma=args.gamma,
                                 beta_critic=args.beta_critic,
                                 beta_entropy=args.beta_entropy,
                                 logger=run)

    sig_handler = SignalManager()
    sig_handler.set_training_context(trainer, args.run_name)
    sig_handler.register_handler()

    train_stats = trainer.train(run_name=args.run_name)
    f = open(f"{args.run_name}/stats_{args.run_name.split('/')[-1]}.p", "wb")
    pickle.dump(train_stats, f)
    f.close()