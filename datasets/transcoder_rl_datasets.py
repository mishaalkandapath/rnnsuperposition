from typing import Dict, Tuple, List
import os
import random
import numpy as np
from collections import defaultdict, Counter, OrderedDict

import torch
from torch.utils.data import ConcatDataset, StackDataset, Subset
from tqdm import tqdm

from models.rnn import RNN
from datasets.task_datasets import TwoStepRLEnvironment
from training.rl_twotask import ActorCriticTrainer

# naively, i can just sample trials from the environment in varying lengths and collect data and train 
# but the phenomena of interest is when an uncommon transition is followed by a reward. -- which is the case for when you choose A2, go to S1 (0.2) and then a reward with 0.9 chance. and then there is the 0.025 chance of switching the value of states themselves. 
# i need to make sure -- uncommon transitions getting rewarded is well represented, and so are behaviours after value reversal. 
"""
In general there are 6 types of trials:
1. Start trials, where they do not get a reward (generally means started with reversal, or an uncommon transition ocucred, or failed to get reward in normal setting -- ) 
2. Start trials, where they do (generally, normal setting, or did reverse by uncommon transition, or reverse and failed to get reward because of prob)
3. Mid trials, preceded by reward, and no change in reward stat (normal setting)
4. Mid trials, preceded by no reward, no change in reward stat (common transition was unrewarded)
3. Mid trials, preceded by reward, and but change in reward stat
4. Mid trials, preceded by no reward, but change in reward stat
Data Variations:
1. Episodes that start with (0.9, 0.1)
2. Episodes that start with (0.1, 0.9)
3. Episodes where a switch happens atleast n times (n = 1....5)
4. Episodes with no switch
----
For every n episode, there are 100-n trials where no change has occurred
Within each uninterrupted chain -- an uncommon transition happens 20% of the time. 
Balance trials so that uncommon transition related states are represented equally as well as otherwise
Balance trials so that for every n=1, there are 2 n=2s, 3 n=3, 4 n=4s, 5 n=5s. 
---
Structure: 
1. 100 trials per episode
2. 50% of data points belong to episodes that start one way, and the other 50 for the other. 
3. Within each type of episode, 
"""

class RLTranscoderDataGenerator:
    """Generate training data for RNN transcoders from RL task"""
    
    def __init__(self, rl_agent: ActorCriticTrainer, device='cpu'):
        """
        Args:
            rl_agent: Trained RL agent with RNN model
            device: Device to run computations on
        """
        self.rl_agent = rl_agent
        self.device = device
        self.rl_agent.model.eval()
        
    def generate_transcoder_dataset(self, 
                                    n_episodes: int,
                                    trials_per_episode: int = 100,
                                    balance_patterns: bool = True,
                                    max_extra_episodes: int = None,
                                    initial_config: str = "high_first") -> Tuple[StackDataset, StackDataset]:
        """
        Generate dataset for training transcoders from RL episodes
        
        Args:
            n_episodes: Number of episodes to generate
            trials_per_episode: Number of trials per episode
            balance_patterns: Whether to balance uncommon transition/reward patterns
            max_extra_episodes: Maximum extra episodes to collect for balancing
            
        Returns:
            Tuple of (update_gate_dataset, hidden_context_dataset)
        """
        
        # Generate initial episodes with controlled reversals
        episodes_data, by_pattern_counts = self._generate_controlled_episodes(
                                                    n_episodes,
                                                    trials_per_episode,
                                                    initial_config=initial_config
                                                )
        
        if balance_patterns:
            episodes_data = self._balance_patterns(episodes_data, 
                                                   by_pattern_counts,
                                                   initial_config,
                                                   trials_per_episode, max_extra_episodes)
        
        update_dataset, hidden_dataset = self._extract_transcoder_data(episodes_data)
        
        return update_dataset, hidden_dataset
    
    def _generate_controlled_episodes(self, n_episodes: int, 
                                      trials_per_episode: int, 
                                      initial_config: str) -> Dict[str, List]:
        assert initial_config in ['high_first', 'low_first']
        episodes_data = []
        all_patterns_count = [0, 0, 0, 0]
        # Equal representation of reversal counts (1, 2, 3, 4)
        episodes_per_reversal = n_episodes // 4
        reversal_counts = [1, 2, 3, 4] * episodes_per_reversal
        
        for episode_idx in tqdm(range(n_episodes)):
            num_reversals = reversal_counts[episode_idx]

            reversal_indices = sorted(random.sample(range(1, trials_per_episode-1), num_reversals))
            
            self.rl_agent.env.reset_trial(reset_r=True)
            episode_data, patterns_count, _ = self._run_single_episode(
                trials_per_episode, 
                reversal_indices, 
                initial_config
            )
            all_patterns_count = (patterns_count if not all_patterns_count else [all_patterns_count[i] + list(patterns_count.values())[i] for i in range(4)])
            episodes_data = (episode_data if not episodes_data else {k: torch.concat([episodes_data[k], episode_data[k]]) for k in episodes_data})
            
        return (
            episodes_data, 
            OrderedDict([(["commonp", "common(1-p)",
                            "uncommonp", "uncommon(1-p)"][i], all_patterns_count[i]) for i in range(4)
                        ]))
    
    def _run_single_episode(self, trials_per_episode: int, 
                            reversal_indices: List[int], 
                            initial_config: str) -> Dict:
        
        #the hidden state of the curren trial must update in light of transition -- common or uncommon
        # the hidden state of the next trial D1 phase will change based on the reward given -- it'll also influence the action and what not. 
        # so both trials, collectively, inform a type of transition. 
        
        by_pattern = OrderedDict([("commonp", 0), ("common(1-p)", 0),
                                  ("uncommonp", 0), ("uncommon(1-p)", 0)])
        if initial_config == 'high_first':
            self.rl_agent.env.reward_probs[1] = 0.9  # State 1
            self.rl_agent.env.reward_probs[2] = 0.1  # State 2
        else:
            self.rl_agent.env.reward_probs[1] = 0.1
            self.rl_agent.env.reward_probs[2] = 0.9
            
        actions_history = []
        
        # RNN internal state
        hidden_states = []
        update_gates = []
        reset_gates = []
        new_contexts = []
        inputs = []
        
        hidden_state = None
        reversal_idx = 0
        state_pattern_mask = []
        prev_trial_key = ""
        for trial in (range(trials_per_episode)):
            if reversal_idx < len(reversal_indices) and trial == reversal_indices[reversal_idx]:
                self.rl_agent.env.reset_trial(manual_switch=True)
                reversal_idx += 1
            else:
                self.rl_agent.env.reset_trial()
            curr_trial_key = ""
            for phase in range(3):  # DELAY_1, GO, DELAY_2
                # Get observation and convert to input tensor
                self.rl_agent.env.get_observation()
                input_vec = self.rl_agent.env.get_input_vector()
                input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                if phase == 2:
                    if self.rl_agent.env.current_state == actions_history[-1]:
                        curr_trial_key+="common"
                    else:
                        curr_trial_key+="uncommon"
                    cur_rel_state = self.rl_agent.env.current_state
                
                with torch.no_grad():
                    policy_logits, value_estimate, new_hidden_state, gate_data = self._forward_with_gates(
                        input_tensor, hidden_state
                    )
                
                action = self.rl_agent.select_action(policy_logits,
                                                     inference=True,
                                                     phase=phase)
                # take that action
                self.rl_agent.env.step(action)
                reward = getattr(self.rl_agent.env, 'last_reward', 0)

                if phase == 2:
                    p_reward = self.rl_agent.env.reward_probs[cur_rel_state]
                    curr_trial_key += "p" if (reward > 0 and p_reward > 0.5) else "(1-p)"
                    by_pattern[curr_trial_key] += 1
                    state_pattern_mask += [list(by_pattern.keys()).index(curr_trial_key)]
                    prev_trial_key = curr_trial_key

                actions_history.append(action)
                inputs.append(input_tensor.squeeze(0))
                update_gates.append(gate_data['update_gate'].squeeze(0))
                reset_gates.append(gate_data['reset_gate'].squeeze(0))
                new_contexts.append(gate_data['new_context'].squeeze(0))
                hidden_states.append(hidden_state[0] if hidden_state is not None else self.rl_agent.model.initial_states[0].detach().broadcast_to(new_hidden_state[0].shape)) # bcz learned init state

                if phase != 2 and prev_trial_key:
                    state_pattern_mask += [list(by_pattern.keys()).index(prev_trial_key)]
                elif phase != 2: # only on initial transition
                    state_pattern_mask += [-1]

                hidden_state = new_hidden_state
        
        return {
            'inputs': torch.concat(inputs),
            'hidden_states': torch.concat(hidden_states),
            'update_gates': torch.concat(update_gates),
            'reset_gates': torch.concat(reset_gates),
            'new_contexts': torch.concat(new_contexts),
        }, by_pattern, torch.tensor(state_pattern_mask)
    
    def _forward_with_gates(self, input_tensor, hidden_state):        
        model = self.rl_agent.model

        new_logits, new_hidden, r_records, z_records, h_new_records, h_records = model(input_tensor, hidden_state, record_gates=True)
    
        # Get policy and value outputs
        policy_logits = new_logits[:, :, :3]
        value_estimate = new_logits[:, :, 3]
        gate_data = {
            'update_gate': z_records[0],
            'reset_gate': r_records[0],
            'new_context': h_new_records[0]
        }
        
        return policy_logits, value_estimate, new_hidden, gate_data
    
    def _balance_patterns(self, episodes_data: Dict[str, List], 
                          pattern_counts: Dict[str, int],
                          initial_config:str, 
                          trials_per_episode: int, 
                          max_extra_episodes: int = None) -> List[Dict]:
        """Balance uncommon transition and reward patterns distribution in the dataset"""
        
        if max_extra_episodes is None:
            max_extra_episodes = 40*len(episodes_data)
        
        print("Initial pattern counts:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")
        
        target_count = max(pattern_counts.values())
        left_counts = OrderedDict([(k, target_count - pattern_counts[k]) for k in pattern_counts])
        extra_episodes = 0
        og_count = max(left_counts.values())
        pbar = tqdm(total=og_count)
        while extra_episodes < max_extra_episodes:
            if all(count >= target_count for count in pattern_counts.values()):
                break
                
            num_reversals = random.choice([1, 2, 3, 4])
            reversal_indices = sorted(random.sample(range(1, trials_per_episode), num_reversals))
            
            episode_data, episode_patterns, pattern_mask = self._run_single_episode(trials_per_episode, reversal_indices, initial_config)

            # tally patterns, add when necessary
            update=False
            for pattern in left_counts:
                if left_counts[pattern]:
                    update=True
                    num_suitable = episode_patterns[pattern]
                    take = min(num_suitable, left_counts[pattern])
                    left_counts[pattern] -= take
                    pattern_counts[pattern] += take
                    indices = torch.where(pattern_mask == list(left_counts.keys()).index(pattern))[0][:take]

                    for key in episodes_data:
                        episodes_data[key] = torch.concat([episodes_data[key], episode_data[key][indices]], dim=0)
            if update:
                pbar.update(og_count - max(left_counts.values()))
                og_count = max(left_counts.values())
        
        print(f"Generated {extra_episodes} additional episodes for balancing")
        print("Final pattern counts:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")
        
        return episodes_data
    
    def _extract_transcoder_data(self, episodes_data: Dict[str, List]) -> Tuple[StackDataset, StackDataset]:
        """Extract transcoder training data from episodes"""

        inputs = episodes_data['inputs']
        hidden_states = episodes_data['hidden_states']
        update_gates = episodes_data['update_gates']
        reset_gates = episodes_data['reset_gates']
        new_contexts = episodes_data['new_contexts']

        inputs_update = torch.concat([inputs, hidden_states], dim=-1)
        targets_update = update_gates
        inputs_hidden = torch.concat([inputs, hidden_states * reset_gates], dim=-1)
        targets_hidden = new_contexts

        update_dataset = StackDataset(
            input=inputs_update,
            output=targets_update
        )
        
        hidden_dataset = StackDataset(
            input=inputs_hidden,
            output=targets_hidden
        )
        
        print(f"Generated transcoder dataset with {len(update_dataset)} samples")
        
        return update_dataset, hidden_dataset


def create_rl_transcoder_dataloaders(dataset: ConcatDataset,
                                   batch_size: int = 256,
                                   train_split: float = 0.9,
                                   shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val dataloaders for transcoder training"""
    
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=len(os.sched_getaffinity(0)), persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RL Transcoder Dataset")
    
    parser.add_argument("--n_episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--trials_per_episode", type=int, default=100, help="Trials per episode")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained RL agent")
    parser.add_argument("--balance_patterns", action='store_true', help="Balance transition/reward patterns")
    parser.add_argument("--max_extra_episodes", type=int, default=None, help="Max extra episodes for balancing")
    parser.add_argument("--initial_config", type=str, required=True)
    
    args = parser.parse_args()
    device = torch.device("cpu")
    env = TwoStepRLEnvironment()
    env.get_observation()
    env.reset_trial(reset_r=True)
    in_size = len(env.get_input_vector())
    model = RNN(input_size=in_size, hidden_size=48, out_size=4,
                 out_act=lambda x: x, use_gru=True, learn_init=True)
    model.load_state_dict(torch.load(args.model_path)) 
    agent = ActorCriticTrainer(model, env, device=device)
    
    generator = RLTranscoderDataGenerator(agent, device=device)
    update_dataset, hidden_dataset = generator.generate_transcoder_dataset(
        n_episodes=args.n_episodes,
        trials_per_episode=args.trials_per_episode,
        balance_patterns=args.balance_patterns,
        max_extra_episodes=args.max_extra_episodes,
        initial_config=args.initial_config
    )
    
    os.makedirs("/w/nobackup/436/lambda/data/rl_transcoder/", exist_ok=True)
    torch.save(update_dataset, f"/w/nobackup/436/lambda/data/rl_transcoder/{args.dataset_name}_update_gate.pt")
    torch.save(hidden_dataset, f"/w/nobackup/436/lambda/data/rl_transcoder/{args.dataset_name}_hctx.pt")