from typing import Dict, Tuple, List
import os
import random
import numpy as np
from collections import defaultdict, Counter, OrderedDict

import torch
from torch.utils.data import ConcatDataset, StackDataset, Subset
from tqdm import tqdm

from models.rnn import RNN
from datasets.task_datasets import TwoStepRLEnvironment, BatchedTwoStepRLEnvironment
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
    
    def __init__(self, rl_agent: ActorCriticTrainer, batch_size: int, device='cpu'):
        """
        Args:
            rl_agent: Trained RL agent with RNN model
            device: Device to run computations on
        """
        self.rl_agent = rl_agent
        self.device = device
        self.rl_agent.model.eval()
        self.batch_size = batch_size
        
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
        episodes_data, by_pattern_counts, sequences = self._generate_controlled_episodes(
                                                    n_episodes,
                                                    trials_per_episode,
                                                    initial_config=initial_config
                                                )
        
        if balance_patterns:
            episodes_data = self._balance_patterns(episodes_data, 
                                                   by_pattern_counts,
                                                   initial_config,
                                                   trials_per_episode, max_extra_episodes)
        
        update_dataset, hidden_dataset, sequence_datasetes = self._extract_transcoder_data(episodes_data, sequences)
        
        return update_dataset, hidden_dataset, sequence_datasetes
    
    def _generate_controlled_episodes(self, n_episodes: int, 
                                      trials_per_episode: int, 
                                      initial_config: str) -> Dict[str, List]:
        assert initial_config in ['high_first', 'low_first']
        all_episodes_data = []
        all_patterns_count = [0, 0, 0, 0]
        # Equal representation of reversal counts (1, 2, 3, 4)
        n_batches = (n_episodes + self.batch_size - 1) // self.batch_size
        episodes_per_reversal = n_episodes // 4
        reversal_counts = [1, 2, 3, 4] * episodes_per_reversal
        
        all_batch_sequences = OrderedDict(
            [(c, defaultdict(torch.Tensor))for c in range(4)] 
                                           )
        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_episodes)
            current_batch_size = end_idx - start_idx

            batch_reversal_counts = reversal_counts[start_idx:end_idx]
            batch_data, batch_patterns, _, batch_sequences = self._run_batch_episodes(
                current_batch_size,
                batch_reversal_counts,
                trials_per_episode, 
                initial_config,
                make_sequences=True
            )
            least_sequences = min(batch_sequences[i]["inputs"].size(0) for i in range(4))
            if not all_batch_sequences[0]: 
                all_batch_sequences = {pattern: {k: batch_sequences[pattern][k][:least_sequences] for k in batch_sequences[pattern]} for pattern in batch_sequences}
            else:
                all_batch_sequences = {pattern: {k: 
                                                torch.concat([
                                                    batch_sequences[pattern][k][:least_sequences],all_batch_sequences[pattern][k]])
                                                    for k in batch_sequences[pattern]} 
                                                    for pattern in batch_sequences}

            all_episodes_data.append(batch_data)
            all_patterns_count = [all_patterns_count[i] + batch_patterns[i] for i in range(4)]
        
        episodes_data = self._concatenate_batch_data(all_episodes_data)
        return (
            episodes_data, 
            OrderedDict([(["commonp", "common(1-p)",
                            "uncommonp", "uncommon(1-p)"][i], all_patterns_count[i]) for i in range(4)
                        ]),
            all_batch_sequences)

    def _concatenate_batch_data(self, all_batch_data: List[Dict]) -> Dict:
        if not all_batch_data:
            return {}
        
        keys = all_batch_data[0].keys()
        concatenated = {}
        
        for key in keys:
            concatenated[key] = torch.cat([batch_data[key] for batch_data in all_batch_data], dim=0)
        
        return concatenated
    
    def _run_batch_episodes(self, batch_size: int, reversal_counts: List[int], 
                            trials_per_episode: int, initial_config: str,
                            make_sequences=False) -> Dict:
        #the hidden state of the curren trial must update in light of transition -- common or uncommon
        # the hidden state of the next trial D1 phase will change based on the reward given -- it'll also influence the action and what not. 
        # so both trials, collectively, inform a type of transition. 
        env_batch = self.rl_agent.env
        batch_reversal_indices = torch.zeros((batch_size, trials_per_episode))
        for i in range(batch_size):
            num_reversals = reversal_counts[i]
            reversal_indices = sorted(random.sample(range(1, trials_per_episode-1), num_reversals))
            batch_reversal_indices[i][reversal_indices] = 1
        
        batch_patterns = [0, 0, 0, 0]
        
        #  sequence data for feature analysis and circuit vizualization
        # type -> list of sequences by batch
        sequence_dict = defaultdict(lambda: [[] for _ in range(batch_size)])
        
        # Context window for each episode (max 3 trials)
        context_windows = [[] for _ in range(batch_size)]
        # Track previous trial types for each episode
        prev_trial_types = [None for _ in range(batch_size)]
        
        env_batch.reset_trial_batch(reset_r=True)
        if initial_config == 'high_first':
            env_batch.reward_probs[:, 1] = 0.9  # State 1
            env_batch.reward_probs[:, 2] = 0.1  # State 2
        else:
            env_batch.reward_probs[:, 1] = 0.1
            env_batch.reward_probs[:, 2] = 0.9
            
        actions_history = [[] for _ in range(batch_size)]
        batch_pattern_mask = [[] for _ in range(batch_size)] 
        # RNN internal state
        batch_inputs = []
        batch_hidden_states = []
        batch_update_gates = []
        batch_reset_gates = []
        batch_new_contexts = []
        
        hidden_states = None
        for trial in (range(trials_per_episode)):
            manual_switch = torch.zeros(batch_size, dtype=torch.bool)
            switch_mask = batch_reversal_indices[:, trial] != 0
            manual_switch[switch_mask] = 1
            env_batch.reset_trial_batch(manual_switch=manual_switch)
            
            # Store trial data for each episode
            trial_data = [{} for _ in range(batch_size)]
            
            for phase in range(3):  # DELAY_1, GO, DELAY_2
                # Get observation and convert to input tensor
                input_vectors = env_batch.get_input_vector_batch()
                input_tensor = input_vectors.unsqueeze(1).to(self.device).to(dtype=torch.float32)
                
                with torch.no_grad():
                    policy_logits, value_estimate, new_hidden_states, gate_data = self._forward_with_gates(
                        input_tensor, hidden_states
                    )
                
                actions = self.rl_agent.select_action_batched(policy_logits,
                                                            phase=phase)
                # take that action
                _, rewards = env_batch.step_batch(actions)
                
                # Store phase data for each episode
                for ep_idx in range(batch_size):
                    if make_sequences:
                        if 'inputs' not in trial_data[ep_idx]:
                            trial_data[ep_idx]['inputs'] = []
                            trial_data[ep_idx]['h_prevs'] = []
                            trial_data[ep_idx]['z_ts'] = []
                            trial_data[ep_idx]['r_ts'] = []
                            trial_data[ep_idx]['h_new_ts'] = []
                            trial_data[ep_idx]['outputs'] = []
                        
                        trial_data[ep_idx]['inputs'].append(input_tensor[ep_idx].squeeze(0))
                        trial_data[ep_idx]['z_ts'].append(gate_data['update_gate'][ep_idx].squeeze(0))
                        trial_data[ep_idx]['r_ts'].append(gate_data['reset_gate'][ep_idx].squeeze(0))
                        trial_data[ep_idx]['h_new_ts'].append(gate_data['new_context'][ep_idx].squeeze(0))
                        trial_data[ep_idx]['outputs'].append(torch.concat([policy_logits[ep_idx], value_estimate[ep_idx].unsqueeze(0)], dim=-1).squeeze(0))
                        
                        if hidden_states is not None:
                            trial_data[ep_idx]['h_prevs'].append(hidden_states[0][ep_idx])
                        else:
                            initial_hidden = self.rl_agent.model.initial_states[0].detach()
                            trial_data[ep_idx]['h_prevs'].append(initial_hidden)
                        
                    if phase == 2:
                        current_state = env_batch.current_state[ep_idx].item()
                        last_action = actions_history[ep_idx][-1] if actions_history[ep_idx] else 0
                        reward = rewards[ep_idx].item()
                        
                        # Determine current trial type
                        if current_state == last_action:
                            pattern_type = 0 if reward > 0 else 1  # common
                        else:
                            pattern_type = 2 if reward > 0 else 3  # uncommon
                        
                        batch_patterns[pattern_type] += 1
                        batch_pattern_mask[ep_idx].append(pattern_type)
                        
                        if make_sequences:
                            # 1. If there's a previous trial type, add current trial to that sequence
                            trial_stacked = {key: torch.stack(val) for key, val in trial_data[ep_idx].items()}
                            if prev_trial_types[ep_idx] is not None:
                                # Find the most recent sequence of the previous trial type
                                prev_type_sequences = sequence_dict[prev_trial_types[ep_idx]][ep_idx]
                                if prev_type_sequences:  # If there are existing sequences for this type
                                    # Add current trial to the most recent sequence
                                    last_sequence = prev_type_sequences[-1]
                                    assert len(last_sequence) < 5  # Max length is 5
                                    last_sequence.append(trial_stacked)
                        
                            # 2. Create new sequence starting with context + current trial
                            new_sequence = []
                            
                            # Add context trials (up to 3 previous trials)
                            for context_trial in context_windows[ep_idx]:
                                new_sequence.append(context_trial)
                            
                            # Add current trial as the "target" trial
                            new_sequence.append(trial_stacked)
                            sequence_dict[pattern_type][ep_idx].append(new_sequence)
                            
                            # 3. Update context window (max 3 trials)
                            context_windows[ep_idx].append(trial_stacked)
                            if len(context_windows[ep_idx]) > 3:
                                context_windows[ep_idx].pop(0)  # Remove earliest trial
                            
                            # Update previous trial type
                            prev_trial_types[ep_idx] = pattern_type
                    else:
                        batch_pattern_mask[ep_idx].append(batch_pattern_mask[ep_idx][-1] if trial else -1)
                    actions_history[ep_idx].append(actions[ep_idx].item())

                batch_inputs.append(input_tensor.squeeze(1))
                batch_update_gates.append(gate_data['update_gate'].squeeze(1))
                batch_reset_gates.append(gate_data['reset_gate'].squeeze(1))
                batch_new_contexts.append(gate_data['new_context'].squeeze(1))
                if hidden_states is not None:
                    batch_hidden_states.append(hidden_states[0])
                else:
                    initial_hidden = self.rl_agent.model.initial_states[0].detach()
                    batch_hidden_states.append(initial_hidden.broadcast_to(new_hidden_states[0].shape))

                hidden_states = new_hidden_states    
        ret_dict = {
            'inputs': torch.concat(batch_inputs),
            'hidden_states': torch.concat(batch_hidden_states),
            'update_gates': torch.concat(batch_update_gates).to(batch_inputs[0].device),
            'reset_gates': torch.concat(batch_reset_gates).to(batch_inputs[0].device),
            'new_contexts': torch.concat(batch_new_contexts).to(batch_inputs[0].device),
        }
        if make_sequences: 
            # make sequences dicts:
            keys = ["inputs", "outputs", "h_prevs",
                        "h_new_ts", "z_ts", "r_ts"]
            for pattern in sequence_dict:
                for batch_idx in range(len(sequence_dict[pattern])):
                    for sequence_idx in range(len(sequence_dict[pattern][batch_idx])):
                        temp_dict = {}
                        for key in keys:
                            zero = [torch.zeros(sequence_dict[pattern][batch_idx][sequence_idx][0][key].shape).to(self.device) + 5]
                            left = 5 - len(sequence_dict[pattern][batch_idx][sequence_idx])
                            zero *= left
                            assert sequence_dict[pattern][batch_idx][sequence_idx]
                            temp_dict[key] = torch.stack(zero + [s[key].to(self.device) for s in sequence_dict[pattern][batch_idx][sequence_idx]])
                        sequence_dict[pattern][batch_idx][sequence_idx] = temp_dict
                    if not sequence_dict[pattern][batch_idx]: continue
                    temp_dict = {} 
                    for key in keys:
                        temp_dict[key] = torch.stack([s[key] for s in sequence_dict[pattern][batch_idx]])
                    sequence_dict[pattern][batch_idx] = temp_dict
                assert sequence_dict[pattern]
                temp_dict = {}
                for key in keys:
                    temp_dict[key] = torch.concat([s[key] for s in sequence_dict[pattern] if s])
                sequence_dict[pattern] = temp_dict

            return ret_dict, batch_patterns, torch.tensor(sum(batch_pattern_mask, start=[])).to(batch_inputs[0].device), sequence_dict
        return ret_dict, batch_patterns, torch.tensor(sum(batch_pattern_mask, start=[])).to(batch_inputs[0].device)
    
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
            max_extra_episodes = 40*len(episodes_data["inputs"])// (trials_per_episode * 3)
        
        print("Initial pattern counts:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")
        
        target_count = max(pattern_counts.values())
        left_counts = [target_count - pattern_counts[k] for k in pattern_counts]
        extra_episodes = 0

        og_count = max(left_counts)
        pbar = tqdm(total=og_count)
        while extra_episodes < max_extra_episodes and max(left_counts):
            batch_reversal_counts = [random.choice([1, 2, 3, 4]) for _ in range(self.batch_size)]

            batch_data, batch_patterns, batch_pattern_mask = self._run_batch_episodes(
                self.batch_size,
                batch_reversal_counts,
                trials_per_episode, 
                initial_config
            )

            # tally patterns, add when necessary
            update=False
            for pattern_idx in range(4):
                if left_counts[pattern_idx]:
                    num_suitable = batch_patterns[pattern_idx]
                    take = min(num_suitable, left_counts[pattern_idx])
                    left_counts[pattern_idx] -= take
                    pattern_counts[["commonp", "common(1-p)",
                                    "uncommonp", "uncommon(1-p)"][pattern_idx]] += take
                    update = take > 0 if not update else update
                    pattern_mask = batch_pattern_mask == pattern_idx
                    for key in episodes_data:
                        episodes_data[key] = torch.concat([episodes_data[key], batch_data[key][pattern_mask][:take]], dim=0)

            if update:
                pbar.update(og_count - max(left_counts))
                og_count = max(left_counts)
            extra_episodes += batch_size
        
        print(f"Generated {extra_episodes} additional episodes for balancing")
        print("Final pattern counts:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")
        
        return episodes_data
    
    def _extract_transcoder_data(self, episodes_data: Dict[str, List],
                                 sequences: Dict[str, 
                                                       Dict[str, torch.Tensor]]) -> Tuple[StackDataset, StackDataset]:

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
        print(f"Generated Sequences have {sequences[0]["inputs"].size(0)} each")

        sequence_datasets = []
        for pattern in sequences:
            sequence_datasets.append(StackDataset(**sequences[pattern]))
        
        return update_dataset, hidden_dataset, sequence_datasets


def create_rl_transcoder_dataloaders(dataset: ConcatDataset,
                                   batch_size: int = 256,
                                   train_split: float = 0.9,
                                   shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0#, persistent_workers=True
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
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    batch_size = 1024
    env = BatchedTwoStepRLEnvironment(batch_size, reward_switch_prob=0, device=device)
    in_size = 8
    model = RNN(input_size=in_size, hidden_size=48, out_size=4,
                 out_act=lambda x: x, use_gru=True, learn_init=True)
    model.load_state_dict(torch.load(args.model_path)) 
    agent = ActorCriticTrainer(model, env, device=device)
    
    generator = RLTranscoderDataGenerator(agent, batch_size, device=device)
    update_dataset, hidden_dataset, sequence_datasets = generator.generate_transcoder_dataset(
        n_episodes=args.n_episodes,
        trials_per_episode=args.trials_per_episode,
        balance_patterns=args.balance_patterns,
        max_extra_episodes=args.max_extra_episodes,
        initial_config=args.initial_config, 
    )
    
    os.makedirs("/w/nobackup/436/lambda/data/rl_transcoder/", exist_ok=True)
    torch.save(update_dataset, f"/w/nobackup/436/lambda/data/rl_transcoder/{args.dataset_name}_update_gate.pt")
    torch.save(hidden_dataset, f"/w/nobackup/436/lambda/data/rl_transcoder/{args.dataset_name}_hctx.pt")
    for i, dataset in enumerate(sequence_datasets):
        prefix = ["commonp", "common_p", "uncommonp", "uncommon_p"][i]
        torch.save(dataset, f"/w/nobackup/436/lambda/data/rl_transcoder/{args.dataset_name}_{prefix}_{args.initial_config}_rl.pt")
 
# ecah trial will have a context window of upto 3 trials before it, itself, and the trial after it. 
# what is interesting is the relationship between the trials before it and itself.
# it can be that the trial is consisten -- or inconsistent (common w reward and uncommon without as opposed to common without reward and uncommon with)