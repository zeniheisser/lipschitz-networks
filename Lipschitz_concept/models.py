import torch
import torch.nn as nn
import monotonicnetworks as lmn

class UnconstrainedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 3, output_bias_init=None):
        super(UnconstrainedModel, self).__init__()

        if depth < 2: # Check for valid depth
            raise ValueError("Depth must be at least 2 for the UnconstrainedModel.")

        layers = [ # Adding input layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(depth - 2): # Add hidden layers according to depth
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim)) # Output layer

        if output_bias_init is not None: # Start the output at a given offset (typically the mean of the training targets)
            nn.init.constant_(layers[-1].bias, output_bias_init)

        self.layers = nn.Sequential(*layers) # Create the sequential model with the defined layers
        
    def forward(self, x):
        return self.layers(x)


class LipschitzModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_groups=2,
                 depth = 3, 
                 lipschitz_bound=1.0,
                 output_bias_init=None,
                 hidden_init="default",
                 input_norm="l1"):
        super(LipschitzModel, self).__init__()

        ######## CHECKING PARAMETERS #######
        if hidden_dim % n_groups != 0: # Check for divisibility for the GroupSort activation
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_groups ({n_groups})")
        
        if depth < 2: # Check for valid depth
            raise ValueError("Depth must be at least 2 for the LipschitzModel.")

        if hidden_init not in ("default", "identity"):
            raise ValueError(f"hidden_init ({hidden_init}) must be 'default' or 'identity'")

        # The norm of the INPUT the bound is measured in, |f(x) - f(y)| <= L ||x - y||, is set by the first layer.
        # All later layers are "inf" normalised (inf -> inf), so the first layer decides:
        #   "l1":   "one-inf", every weight |w_ij| <= 1, so every partial derivative is at most L (the default)
        #   "l2":   "two-inf", every row has Euclidean length <= 1, the same bound in every direction
        #   "linf": "inf",     every row has L1 length <= 1, so |df/dx_1| + ... + |df/dx_n| <= L
        # In 1D the three are the same.
        first_layer_kinds = {"l1": "one-inf", "l2": "two-inf", "linf": "inf"}
        if input_norm not in first_layer_kinds:
            raise ValueError(f"input_norm ({input_norm}) must be 'l1', 'l2' or 'linf'")

        ####### ADDING LAYERS #######
        layers = [ # Adding input layer with direct normalization
            lmn.direct_norm(torch.nn.Linear(input_dim, hidden_dim), kind=first_layer_kinds[input_norm]),
            lmn.GroupSort(n_groups)
        ]

        for _ in range(depth - 2): # Add hidden layers according to depth
            hidden = nn.Linear(hidden_dim, hidden_dim)
            if hidden_init == "identity":
                # The default init gives rows with an inf-norm of ~sqrt(hidden_dim)/2, so after normalization every layer
                # shrinks the signal by ~sqrt(hidden_dim). For deep networks the signal (and the gradient) then vanishes.
                # A noisy identity has unit row norms and passes the signal through unchanged.
                with torch.no_grad():
                    hidden.weight.copy_(torch.eye(hidden_dim) + 0.01*torch.randn(hidden_dim, hidden_dim))
            layers.append(lmn.direct_norm(hidden, kind="inf"))
            layers.append(lmn.GroupSort(n_groups))

        layers.append(lmn.direct_norm(nn.Linear(hidden_dim, output_dim), kind="inf")) # Output layer

        if output_bias_init is not None:
            # Start the output at a given offset (typically the mean of the training targets). The offset does not affect
            # the Lipschitz constant, but a constrained network can only learn it slowly through the biases.
            # Divided by the bound since the forward pass multiplies the output by it.
            nn.init.constant_(layers[-1].bias, output_bias_init / lipschitz_bound)

        self.layers = nn.Sequential(*layers) # Create the sequential model with the defined layers
        self.lipschitz_bound = lipschitz_bound  # Store the Lipschitz bound for later use
        self.input_norm = input_norm  # Not visible in the layers or weights, so needed to tell stored results apart
        
    def forward(self, x):
        return self.layers(x)*self.lipschitz_bound  
