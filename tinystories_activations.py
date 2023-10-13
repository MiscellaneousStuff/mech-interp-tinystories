import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange, repeat
from tqdm.notebook import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache


class SparseAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SparseAutoencoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        x = self.fc2(h)
        return h, x


def batch_process_list(model, lst):
    tinystories_tokens = [model.to_tokens(txt, prepend_bos=False)
                          for txt in lst]
    tinystories_tokens = torch.tensor(tinystories_tokens).unsqueeze(1)
    tinystories_logits, tinystories_cache = \
        model.run_with_cache(
            tinystories_tokens,
            remove_batch_dim=False)
    return tinystories_logits, tinystories_cache["ln_final.hook_normalized"]

def train(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAX_LR = 1e-3
    EPOCHS = 500

    feature_scaling = 1.0

    # in := d_model, hidden := d_model * feature_scaling, out := d_model
    autoencoder = SparseAutoencoder(
        in_dim=model.cfg.d_model,
        hidden_dim=int(model.cfg.d_model * feature_scaling),
        out_dim=model.cfg.d_model).to(device)

    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=MAX_LR)

    criterion = nn.MSELoss()

    # L1 regularization coefficient
    l1_coeff = 1e-5

    activations = batch_cache_s.to(device)

    for e in range(EPOCHS):
        # for i, activations in enumerate(batch_cache_s):
        
        # Compute forward pass, capturing hidden layer activations
        hidden_activations, reconstructions = autoencoder(activations)
        
        # Compute the MSE loss
        reconstruction_loss = criterion(reconstructions, activations)
        
        # Compute the L1 loss of the hidden layer activations
        l1_loss = hidden_activations.abs().sum()
        
        # Combine the losses
        loss = reconstruction_loss + (l1_coeff * l1_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 10 == 0:
            print("LOSS:", e, loss.item())

if __name__ == "__main__":
    current_model = "roneneldan/TinyStories-1Layer-21M"

    model     = AutoModelForCausalLM.from_pretrained(current_model)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    model = HookedTransformer.from_pretrained(
        current_model,
        hf_model=model,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer)
    
    input_list = [chr(i) for i in range(ord('a'), ord('z')+1)] # letters a-z (will eventually be replaced with tiny stories entire dataset)
    batch_logits_s, batch_cache_s = batch_process_list(model, input_list)

    print(batch_logits_s.shape, batch_cache_s.shape)

    print(torch.cuda.mem_get_info(device=torch.device("cuda:0")))