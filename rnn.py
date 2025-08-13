import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(VanillaRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden transformation
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=bias)
        # Hidden-to-hidden transformation
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        h_t = h_0
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current input: (batch_size, input_size)
            
            # RNN computation: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
            h_t = self.activation(
                self.input_to_hidden(x_t) + self.hidden_to_hidden(h_t)
            )
            outputs.append(h_t.unsqueeze(1))  # Add time dimension back
        
        # Stack all outputs along time dimension
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return outputs, h_t

class GRULayer(nn.Module):
    """Custom GRU layer built from scratch using PyTorch primitives"""
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate parameters
        self.input_to_reset = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_reset = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Update gate parameters
        self.input_to_update = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_update = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # New gate parameters
        self.input_to_new = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_new = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, h_0=None, record_gates=False):
        """
        Forward pass through GRU layer
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state of shape (batch_size, hidden_size)
        Returns:
            outputs: All hidden states of shape (batch_size, seq_len, hidden_size)
            h_n: Final hidden state of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        h_t = h_0

        r_record, z_record, h_new_record, h_record = [], [], [], []
        
        # Process each time step
        for t in range(seq_len):
            if record_gates:
                h_record.append(h_t.detach().cpu())

            x_t = x[:, t, :]  # Current input: (batch_size, input_size)
            
            # Reset gate: r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
            r_t = torch.sigmoid(
                self.input_to_reset(x_t) + self.hidden_to_reset(h_t)
            )
            
            # Update gate: z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
            z_t = torch.sigmoid(
                self.input_to_update(x_t) + self.hidden_to_update(h_t)
            )
            
            # New gate: n_t = tanh(W_in @ x_t + W_hn @ (r_t * h_{t-1}) + b_n)
            n_t = torch.tanh(
                self.input_to_new(x_t) + self.hidden_to_new(r_t * h_t)
            )
            
            # Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
            h_t = (1 - z_t) * h_t + z_t * n_t
            
            outputs.append(h_t.unsqueeze(1))  # Add time dimension back

            if record_gates:
                r_record.append(r_t.detach().cpu())
                z_record.append(z_t.detach().cpu())
                h_new_record.append(n_t.detach().cpu())
        
        # Stack all outputs along time dimension
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        if record_gates: return outputs, h_t, torch.stack(r_record, dim=1), torch.stack(z_record, dim=1), torch.stack(h_new_record, dim=1), torch.stack(h_record, dim=1)
        return outputs, h_t


class RNN(nn.Module):
    """Base RNN class that can use either vanilla RNN cells or GRU cells"""
    
    def __init__(self, input_size, hidden_size, 
                 out_size=0, out_act=nn.ReLU, num_layers=1, 
                 use_gru=False, hidden_bias=True, out_bias=True, learn_init=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.learn_init = learn_init
        
        # Create layers based on cell type
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            
            if use_gru:
                self.layers.append(GRULayer(layer_input_size, hidden_size, hidden_bias))
            else:
                self.layers.append(VanillaRNNLayer(layer_input_size, hidden_size, hidden_bias))
        if out_size: 
            self.layers.append(nn.Linear(hidden_size, out_size, bias=out_bias))
            self.out_act = out_act
        if self.learn_init:
            self.initial_states = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_size)) 
                for _ in range(num_layers)
            ])
        print(self.layers)
    
    def forward(self, x, h_0=None, record_gates=False):
        """
        Forward pass through multi-layer RNN
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden states for all layers, list of tensors or None
        Returns:
            outputs: All hidden states from final layer (batch_size, seq_len, hidden_size)
            final_hiddens: Final hidden states from all layers, list of tensors
        """
        batch_size = x.size(0)
        
        # Initialize hidden states for all layers if not provided
        if h_0 is None and self.learn_init:
            h_0 = [
                state.unsqueeze(0).expand(batch_size, -1)
                for state in self.initial_states
            ]
        elif h_0 is None:
            h_0 = [None] * self.num_layers
        elif not isinstance(h_0, list):
            h_0 = [h_0] + [None] * (self.num_layers - 1)
        elif len(h_0) < self.num_layers:
            h_0 = h_0 + [None] * (self.num_layers - len(h_0))
        
        outputs = x
        final_hiddens = [] # per layer --for last timestep
        r_records, z_records, h_new_records, h_records = [], [], [], []
        
        # Pass through each layer
        for i in range(len(self.layers) - (self.out_size != 0)):
            layer = self.layers[i]
            # outputs - the output of the hidden layer -- the h_t forall t
            if record_gates:
                outputs, h_n, r_record, z_record, h_new_record, h_record = layer(outputs, h_0[i], record_gates=record_gates)
                r_records.append(r_record)
                z_records.append(z_record)
                h_new_records.append(h_new_record)
                h_records.append(h_record)
            else:
                outputs, h_n = layer(outputs, h_0[i], record_gates=record_gates)
            final_hiddens.append(h_n)
        if self.out_size:
            outputs = self.out_act(self.layers[-1](outputs))
        if record_gates: return outputs, final_hiddens, torch.stack(r_records), torch.stack(z_records), torch.stack(h_new_records), torch.stack(h_records)
        return outputs, final_hiddens


# Example usage and testing
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 4
    input_size = 16
    hidden_size = 32
    num_layers = 2
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test single-layer vanilla RNN
    print("=== Single Layer Vanilla RNN ===")
    vanilla_rnn = RNN(input_size, hidden_size, num_layers=1, use_gru=False)
    vanilla_outputs, vanilla_hiddens = vanilla_rnn(x)
    print(f"Vanilla RNN output shape: {vanilla_outputs.shape}")  # (32, 10, 128)
    print(f"Vanilla RNN final hiddens: {len(vanilla_hiddens)} layers")
    print(f"Each hidden shape: {vanilla_hiddens[0].shape}")  # (32, 128)
    
    # Test multi-layer vanilla RNN
    print("\n=== Multi-Layer Vanilla RNN ===")
    multilayer_vanilla_rnn = RNN(input_size, hidden_size, num_layers=num_layers, use_gru=False)
    multilayer_vanilla_outputs, multilayer_vanilla_hiddens = multilayer_vanilla_rnn(x)
    print(f"Multi-layer Vanilla RNN output shape: {multilayer_vanilla_outputs.shape}")  # (32, 10, 128)
    print(f"Multi-layer Vanilla RNN final hiddens: {len(multilayer_vanilla_hiddens)} layers")
    
    # Test single-layer GRU
    print("\n=== Single Layer GRU ===")
    gru_rnn = RNN(input_size, hidden_size, num_layers=1, use_gru=True)
    gru_outputs, gru_hiddens = gru_rnn(x)
    print(f"GRU RNN output shape: {gru_outputs.shape}")  # (32, 10, 128)
    print(f"GRU RNN final hiddens: {len(gru_hiddens)} layers")
    print(f"Each hidden shape: {gru_hiddens[0].shape}")  # (32, 128)
    
    # Test multi-layer GRU
    print("\n=== Multi-Layer GRU ===")
    multilayer_gru_rnn = RNN(input_size, hidden_size, num_layers=num_layers, use_gru=True)
    multilayer_gru_outputs, multilayer_gru_hiddens = multilayer_gru_rnn(x)
    print(f"Multi-layer GRU RNN output shape: {multilayer_gru_outputs.shape}")  # (32, 10, 128)
    print(f"Multi-layer GRU RNN final hiddens: {len(multilayer_gru_hiddens)} layers")
    
    # Test gradient flow (basic check)
    print("\n=== Gradient Flow Test ===")
    rnn_test = RNN(input_size, hidden_size, num_layers=2, use_gru=True)
    outputs, _ = rnn_test(x)
    loss = outputs.sum()
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(param.grad is not None for param in rnn_test.parameters())
    print(f"Gradients computed successfully: {has_gradients}")
    
    # Count parameters
    total_params = sum(p.numel() for p in rnn_test.parameters())
    print(f"Total parameters in 2-layer GRU: {total_params}")
    
    vanilla_test = RNN(input_size, hidden_size, num_layers=2, use_gru=False)
    vanilla_params = sum(p.numel() for p in vanilla_test.parameters())
    print(f"Total parameters in 2-layer Vanilla RNN: {vanilla_params}")