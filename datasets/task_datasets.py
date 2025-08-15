import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
# torch.serialization.add_safe_globals([TensorDataset])
import math
import random

def generate_sparse_copyset(n_features, feature_prob, copy_length, batch_size):
    """Generate sparse copy dataset"""
    feature_prob = 1 - math.pow(1 - feature_prob, 1/copy_length)
    batch_shape = (batch_size, copy_length, n_features)
    feat_mag = torch.rand(batch_shape)
    feat_seeds = torch.rand(batch_shape)
    return torch.where(feat_seeds <= feature_prob, feat_mag, 0.0)

def generate_token_copyset(n_tokens, batch_size, max_len, min_len=2):
    batch_size, max_len = int(batch_size), int(max_len)
    sequence_indices = torch.randint(0, n_tokens, 
                                            (batch_size, max_len))
    sequence_one_hot = torch.nn.functional.one_hot(sequence_indices, 
                                                    num_classes=n_tokens).float()
    sequence_lengths = torch.randint(min_len, max_len+1, (batch_size,))
    length_mask = torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    sequence_one_hot *= length_mask.unsqueeze(-1)
    return sequence_one_hot, length_mask

def generate_unique_test_set(n_tokens, test_size, 
                             max_len, min_len, train_indices):
    # Convert train sequences (int form) to a set for fast lookup
    train_set = {tuple(row.tolist()) for row in train_indices}
    seen = set(train_set)  # start with all train sequences

    test_indices = []
    test_masks = []

    while len(test_indices) < test_size:
        seq = torch.randint(0, n_tokens, (max_len,))
        seq_len = torch.randint(min_len, max_len+1, (1,))
        mask = torch.arange(max_len) < seq_len
        seq = seq * mask  # zero out pads if needed
        tup = tuple(seq.tolist())

        if tup not in seen:
            seen.add(tup)
            test_indices.append(seq)
            test_masks.append(mask)

    test_indices = torch.stack(test_indices)
    test_masks = torch.stack(test_masks)
    test_one_hot = torch.nn.functional.one_hot(test_indices, num_classes=n_tokens).float()

    return test_one_hot, test_masks

def add_delimiter_dimension(data, concat_del=True):
    """
    Add delimiter dimension to data
    Args:
        data: (batch_size, seq_len, n_features)
        delimiter_pos: position to place delimiter, if None adds at end
    Returns:
        enhanced_data: (batch_size, seq_len + 1, n_features + 1) if delimiter added
                      or (batch_size, seq_len, n_features + 1) if delimiter_pos specified
    """
    batch_size, seq_len, n_features = data.shape
    
    # Add delimiter at the end
    # Expand original data with 0s in delimiter dimension
    expanded_data = torch.cat([data, torch.zeros(batch_size, seq_len, 1).to(data.device)], dim=-1).to(data.device)
    if not concat_del: return expanded_data
    
    # Create delimiter token: all zeros except delimiter dimension = 1
    delimiter = torch.zeros(batch_size, 1, n_features + 1).to(data.device)
    delimiter[:, :, -1] = 1  # Set delimiter dimension to 1
    
    # Concatenate original data + delimiter
    enhanced_data = torch.cat([expanded_data, delimiter], dim=1)
    return enhanced_data.to(data.device)

def sort_seq_by_len(seq_one_hot, loss_mask, max_len, min_len, 
                    add_delim=False, make_copy=False):
    if add_delim:
        seq_one_hot = add_delimiter_dimension(seq_one_hot)
    sequence_lengths = loss_mask.sum(dim=-1)
    sequences_by_length = [[] for _ in range(min_len, max_len+1)]
    targets_by_length = [[] for _ in range(min_len, max_len+1)]
    n_sequences = seq_one_hot.size(0)

    for seq_idx in range(n_sequences):
        actual_length = sequence_lengths[seq_idx].item()
        sequence = seq_one_hot[seq_idx][:actual_length+add_delim]
        sequence[-1, -1] = 1
        sequences_by_length[actual_length-min_len].append(sequence.repeat(1+make_copy, 1)[:-2]) # last two are not reqd coz last token + del
        targets_by_length[actual_length-min_len].append(torch.cat([torch.zeros((sequence.size(0)-1)), sequence.argmax(-1), sequence[-1:, :].argmax(-1)], dim=0)[:-2])
    return sequences_by_length, targets_by_length

def make_copy_targets(seq_one_hot, loss_mask, max_len, min_len):
    # seq_one_hot: batch_size, max_len, 30
    sequences_by_length, targets_by_length = sort_seq_by_len(seq_one_hot, loss_mask, 
                                         max_len, min_len, 
                                         add_delim=True, make_copy=True)

    all_seqs = sum(sequences_by_length, start=[])
    all_targs = sum(targets_by_length, start=[])
    final_lengths = torch.tensor(
                    [seq.size(0) for seq in all_seqs]
                )
    padded_seqs = pad_sequence(all_seqs, batch_first=True, padding_value=0.0)
    padded_targs = pad_sequence(all_targs, batch_first=True, padding_value=0.0)
    pad_masks = ((final_lengths.unsqueeze(1)/2)<= torch.arange(2*max_len).unsqueeze(0)) & (torch.arange(2*max_len).unsqueeze(0) < final_lengths.unsqueeze(1))
    return padded_seqs, pad_masks, padded_targs

def generate_token_copy_dataset(n_tokens, train_size, test_size, 
                                max_len, min_len=2):
    train_seq_one_hot, train_loss_mask = generate_token_copyset(n_tokens,
                                                                train_size,
                                                                max_len,
                                                                min_len=min_len
                                                                )
    train_seq_indices = train_seq_one_hot.argmax(-1)
    test_seq_one_hot, test_loss_mask = generate_unique_test_set(n_tokens,
                                                                test_size,
                                                                max_len,
                                                                min_len, 
                                                                train_seq_indices)
    
    #make copy mechanism 
    train_seq_one_hot, train_loss_mask, train_targs = make_copy_targets(train_seq_one_hot,
                                                            train_loss_mask,
                                                            max_len,
                                                            min_len)

    
    return TensorDataset(train_seq_one_hot, train_loss_mask, train_targs.long()), TensorDataset(test_seq_one_hot, test_loss_mask)

""" 
--- Two-step RL Task ---
1. initial observation state - choose one of two actions. 
2. Each action triggers one of two possible transitions -- ie. moves one of two states. 
3. For each action, there is a high probability and low prob transition (symmetric)
4. The choice of an action -> induces transition -> receives reward 
5. Reward can be high or low, and this can vary trial to trial. 

-- Specifics --
p(S1|a1) = p(S2|a2) = 0.8
p(S2|a1) = p(S1|a2) = 0.2
p(r|S1) = 0.9, p(r|S2) = 0.1
2.5% of this reward allocation switching trial to trial 
Wang: train 10K episodes, 100 trials each. Tested w fixed weights on 300 eps


RNN Train Setup:
1. Learned initial state representation
2. Input at t time consists, of observation ot, a_t-1, and varying reward:
Delay 1:
r -> from your choice last trial 
a -> fixation action
o -> curernt fixation cue state (no go signal)
outputs:
value and action, correct action says fixation actions

Go:
r -> 0
a -> fixation action
o -> fixation cue state (w go signal)
output: value and action, correct action is one of a1 or a2

Delay 2:
r -> 0 
a -> last action
o -> resultant state
value and action, correct action is fixation action
----
Incorrect choices of fixation action leads to a -1 reward. i.e the 0s get -0.1. if you choose fixation in go, that gets a similar L, but ig the following state would be the fixation state. 
----
Output vector has 4 logits, first three are for the three legal actions 
----
Inputs are concat one-hot vectors, except for the scalar reward. For Advantage actor critic, you sample the actions based off the softmax. 
---
d_hidden = 48
"""

class TwoStepRLEnvironment:
    def __init__(self, reward_switch_prob=0.025):
        """
        Two-step RL task environment.
        
        Args:
            reward_switch_prob: Probability of switching reward allocation each trial (default 2.5%)
        """
        self.reward_switch_prob = reward_switch_prob
        
        # Action definitions
        self.FIXATION_ACTION = 0
        self.ACTION_1 = 1
        self.ACTION_2 = 2
        
        # State definitions
        self.FIXATION_STATE = 0
        self.STATE_1 = 1
        self.STATE_2 = 2
        
        # Phase definitions
        self.DELAY_1 = 0  # Show fixation cue, no go signal
        self.GO = 1       # Show fixation cue with go signal
        self.DELAY_2 = 2  # Show resultant state
        
        # Transition probabilities
        self.p_high = 0.8  # High probability transition
        self.p_low = 0.2   # Low probability transition
        
        # Reward probabilities
        self.reward_probs = {
            self.STATE_1: 0.9,
            self.STATE_2: 0.1
        }
        
        self.reset_trial(reset_r=True)
        
    def reset_trial(self, reset_r=False):
        """Reset for a new trial"""
        self.phase = self.DELAY_1
        self.current_state = self.FIXATION_STATE
        self.last_action = self.FIXATION_ACTION
        self.trial_complete = False
        if reset_r:
            self.last_reward = 0
        
        # Possibly switch reward allocation (2.5% chance)
        if random.random() < self.reward_switch_prob:
            self.reward_probs[self.STATE_1], self.reward_probs[self.STATE_2] = \
                self.reward_probs[self.STATE_2], self.reward_probs[self.STATE_1]
    
    def get_observation(self):
        """
        Get current observation as concatenated one-hot vectors plus scalar reward.
        
        Returns:
            dict with:
            - observation: one-hot encoded current state (3-dim)
            - last_action: one-hot encoded last action (3-dim) 
            - reward: scalar reward from last trial
            - go_signal: whether go signal is active
            - phase: current phase of trial
        """
        # One-hot encode current state
        obs = [0, 0, 0]
        obs[self.current_state] = 1
        
        # One-hot encode last action
        action_vec = [0, 0, 0]
        action_vec[self.last_action] = 1
        
        # Go signal (only active in GO phase)
        go_signal = 1 if self.phase == self.GO else 0
        
        return {
            'observation': obs,
            'last_action': action_vec,
            'reward': self.last_reward,
            'go_signal': go_signal,
            'phase': self.phase
        }
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0=fixation, 1=action1, 2=action2)
            
        Returns:
            observation, reward, done, info
        """
        reward = 0
        
        if self.phase == self.DELAY_1:
            # Should choose fixation action
            if action == self.FIXATION_ACTION:
                reward = 0
            else:
                reward = -0.1  # Incorrect choice penalty
            self.phase = self.GO
                
        elif self.phase == self.GO:
            # Should choose action 1 or 2
            if action == self.FIXATION_ACTION:
                reward = -0.1  # Incorrect choice penalty
                self.current_state = self.FIXATION_STATE
            elif action in [self.ACTION_1, self.ACTION_2]:
                # Determine next state based on transition probabilities
                if action == self.ACTION_1:
                    # p(S1|a1) = 0.8, p(S2|a1) = 0.2
                    if random.random() < self.p_high:
                        self.current_state = self.STATE_1
                    else:
                        self.current_state = self.STATE_2
                else:  # action == ACTION_2
                    # p(S2|a2) = 0.8, p(S1|a2) = 0.2
                    if random.random() < self.p_high:
                        self.current_state = self.STATE_2
                    else:
                        self.current_state = self.STATE_1
                    
            self.phase = self.DELAY_2
                
        elif self.phase == self.DELAY_2:
            # Get reward based on resulting state
            # Should choose fixation action
            if action != self.FIXATION_ACTION or not self.current_state:
                reward = -0.1  # Incorrect choice penalty
            elif random.random() < self.reward_probs[self.current_state]:
                reward = 1.0  # High reward
            else:
                reward = 0.0  # Low reward
            
        # Update last action and reward for next step
        self.last_action = action
        self.last_reward = reward  # This will be used in next trial
            
        observation = self.get_observation()
        
        return observation
    
    def get_correct_action(self):
        """Get the correct action for current phase"""
        if self.phase == self.DELAY_1:
            return self.FIXATION_ACTION
        elif self.phase == self.GO:
            # In practice, this would depend on the learned policy
            # Return None to indicate agent should choose
            return None
        elif self.phase == self.DELAY_2:
            return self.FIXATION_ACTION
    
    def get_action_mask(self):
        """Get mask for valid actions (for use with neural network output)"""
        # All actions are technically possible, but some are incorrect
        return [1, 1, 1]  # [fixation, action1, action2]
    
    def get_input_vector(self):
        """
        Get flattened input vector for neural network.
        
        Returns:
            Concatenated vector: [obs(3) + last_action(3) + reward(1) + go_signal(1)]
        """
        obs_data = self.get_observation()
        return (obs_data['observation'] + 
                obs_data['last_action'] + 
                [obs_data['reward']] +
                [obs_data['go_signal']])
    
    def __str__(self):
        phase_names = {0: "DELAY_1", 1: "GO", 2: "DELAY_2"}
        state_names = {0: "FIXATION", 1: "STATE_1", 2: "STATE_2"}
        return f"Phase: {phase_names[self.phase]}, State: {state_names[self.current_state]}"