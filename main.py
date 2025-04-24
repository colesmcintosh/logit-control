# 1. Imports
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress common warnings during model loading
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`resume_download` is deprecated.*")


# 2. Model Loading
model_name = "Qwen/Qwen2.5-0.5B" # Switch to Qwen2.5-0.5B
max_seq_length = 4096 # Keep a reasonable context length for demo

print(f"Loading model: {model_name} using standard Transformers")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = None

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False, # Explicitly False
        quantization_config=None, # Explicitly None
        device_map="auto", # Let transformers handle device mapping (CPU or MPS on Mac)
        torch_dtype="auto", # Let transformers pick best dtype for device
        # trust_remote_code=True # Usually not required for Qwen, uncomment if loading fails
    )
    print("Model loaded successfully using standard Transformers.")
except Exception as e_hf:
     print(f"Error loading model with standard Transformers: {e_hf}")
     print("Please ensure you have necessary libraries (transformers, bitsandbytes, accelerate, torch)")
     print("and that your environment supports 4-bit loading.")
     exit() # Can't proceed without a model

# Ensure the model and tokenizer were loaded
if model is None or tokenizer is None:
    print("Failed to load the model or tokenizer. Exiting.")
    exit()

# Add pad token if missing (common requirement)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        print("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure the model's config also reflects this
        if hasattr(model.config, 'pad_token_id'):
             model.config.pad_token_id = tokenizer.eos_token_id
    else:
        # Add a new pad token if EOS is also missing (less common)
        print("Adding a new pad token '<|pad|>'.")
        added_tokens = tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        # Ensure the config reflects the new pad token ID
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Resized model embeddings. Added {added_tokens} token(s).")


# 3. Define Constraint & LogitsProcessor
class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that forces the next token to be one of the allowed tokens
    after a specific sequence length is reached. Captures logits for visualization.
    """
    def __init__(self, allowed_token_ids, tokenizer, trigger_sequence_len):
        if not isinstance(allowed_token_ids, list):
            allowed_token_ids = [allowed_token_ids]
        self.allowed_token_ids = allowed_token_ids
        self.tokenizer = tokenizer
        self.trigger_sequence_len = trigger_sequence_len
        self.original_logits = None
        self.modified_logits = None
        self.applied_constraint = False # Flag to check if constraint was applied
        self.vocab_size = tokenizer.vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_length = input_ids.shape[1]

        # Apply constraint only at the exact trigger length for this demo
        # We check applied_constraint to ensure it only happens once per generation
        if current_length == self.trigger_sequence_len and not self.applied_constraint:
            self.applied_constraint = True
            print(f"\n--- Applying constraint at step {current_length} ---")

            # Store original logits for visualization (Batch size 1 assumed for demo)
            self.original_logits = scores[0].clone().cpu().numpy()

            # Create a mask: -inf for disallowed tokens, 0 for allowed ones
            # Initialize mask on the same device as scores
            mask = torch.full_like(scores, -float("Inf"))
            valid_allowed_ids = []
            for token_id in self.allowed_token_ids:
                # Check if token_id is valid before indexing
                if 0 <= token_id < self.vocab_size:
                    mask[:, token_id] = 0
                    valid_allowed_ids.append(token_id)
                else:
                     print(f"Warning: Allowed token ID {token_id} is out of vocab bounds ({self.vocab_size}). Skipping.")

            if not valid_allowed_ids:
                print("Warning: No valid allowed token IDs found. Constraint will not be applied effectively.")
                return scores # Return original scores if no valid tokens

            # Apply the mask by adding it to the scores
            scores = scores + mask

            # Store modified logits (Batch size 1 assumed for demo)
            self.modified_logits = scores[0].clone().cpu().numpy()

            allowed_tokens_str = ", ".join([f"'{self.tokenizer.decode([t])}' ({t})" for t in valid_allowed_ids])
            print(f"Constraining next token to: {allowed_tokens_str}")

        return scores


# Get token IDs for " Yes", " No", and " Maybe" (Llama3 often uses leading spaces)
# Using a helper function for robustness
def get_single_token_id(text, tokenizer):
    """Encodes text and returns the first token ID if it's a single token."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 1:
            return tokens[0]
        else:
            print(f"Warning: '{text}' is tokenized into multiple tokens: {tokens}. This constraint processor expects single tokens.")
            return None
    except Exception as e:
        print(f"Error encoding '{text}': {e}")
        return None

yes_id = get_single_token_id(" Yes", tokenizer)
no_id = get_single_token_id(" No", tokenizer)
maybe_id = get_single_token_id(" Maybe", tokenizer)

# Fallback if leading space versions fail
if yes_id is None or no_id is None or maybe_id is None:
    print("Trying without leading space...")
    if yes_id is None: yes_id = get_single_token_id("Yes", tokenizer)
    if no_id is None: no_id = get_single_token_id("No", tokenizer)
    if maybe_id is None: maybe_id = get_single_token_id("Maybe", tokenizer)

# Check if we got valid IDs
if yes_id is None or no_id is None or maybe_id is None:
    print("Error: Could not reliably determine single token IDs for 'Yes', 'No', or 'Maybe'.")
    print("Visualization might be inaccurate or script might fail.")
    exit()
else:
    print(f"Using Token ID for ' Yes'/'Yes': {yes_id}")
    print(f"Using Token ID for ' No'/'No': {no_id}")
    print(f"Using Token ID for ' Maybe'/'Maybe': {maybe_id}")
    allowed_ids = [yes_id, no_id, maybe_id]


# 4. Prompt and Generation Setup
prompt = "Is Liverpool FC the best team in the world? Please answer Yes, No, or Maybe:"
# Ensure prompt ends in a way that naturally leads to Yes/No/Maybe
# Using device from the loaded model
device = model.device
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
prompt_len = input_ids.shape[1] # Length of the tokenized prompt

# Instantiate the processor - apply constraint immediately after the prompt
constraint_processor = ConstrainedLogitsProcessor(allowed_ids, tokenizer, trigger_sequence_len=prompt_len)
logits_processor_list = LogitsProcessorList([constraint_processor])


# 5. Generate Text
print("\n--- Generating with Constraint ---")
# Use sampling with low temperature to strongly favor the higher probability allowed token
outputs = model.generate(
    input_ids,
    max_new_tokens=10,       # Generate a few tokens after the constraint
    logits_processor=logits_processor_list,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,          # Enable sampling
    temperature=0.1,         # Low temp makes it less random, more deterministic
    top_k=10,                # Consider only top 10 tokens during sampling
    repetition_penalty=1.1,  # Discourage repeating tokens
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated Text:\n{generated_text}")


# 6. Visualization and Terminal Output

def plot_probs(logits, title, ax, allowed_ids=None, vocab_size=None):
    """Helper function to plot token probabilities for the graph."""
    if vocab_size is None:
        print("Warning: vocab_size not provided to plot_probs. Cannot proceed.")
        return
    # Ensure logits is a 1D numpy array
    if logits.ndim > 1:
        logits = logits.squeeze()

    # Handle potential NaN/Inf values before softmax
    logits[np.isneginf(logits)] = -1e9
    logits = np.nan_to_num(logits, nan=-1e9)

    # Softmax calculation
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)

    top_k_plot = 15 # Use a different variable name for clarity
    top_k_plot = min(top_k_plot, vocab_size)

    # Get top k indices for plotting
    partitioned_indices = np.argpartition(probs, -top_k_plot)[-top_k_plot:]
    valid_indices = [idx for idx in partitioned_indices if 0 <= idx < vocab_size]
    if not valid_indices:
        print(f"Error: No valid top indices found for plotting in '{title}'.")
        ax.set_title(f"{title}\n(Error: No valid tokens)", fontsize=10)
        return
    valid_indices = np.array(valid_indices)
    top_indices = valid_indices[np.argsort(probs[valid_indices])]

    top_probs = probs[top_indices]

    # Decode tokens safely for plotting
    top_tokens_plot = []
    for idx in top_indices:
        try:
            token_str = tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            token_str = ''.join(c if c.isprintable() else f'[{ord(c):x}]' for c in token_str)
            if not token_str.strip(): token_str = f'[WS {idx}]'
            top_tokens_plot.append(f"{token_str} ({idx})")
        except Exception:
            top_tokens_plot.append(f'[DecodeErr {idx}]')

    # Determine colors for plotting
    colors = []
    for idx in top_indices:
        if allowed_ids and idx in allowed_ids:
            colors.append('forestgreen')
        elif allowed_ids and probs[idx] > 1e-9: # Color suppressed only if prob > 0
             colors.append('salmon')
        else:
            colors.append('cornflowerblue')

    # Plotting
    y_pos = np.arange(len(top_tokens_plot))
    bars = ax.barh(y_pos, top_probs, color=colors, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens_plot, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=12)

    # Add probability values on bars
    max_prob_display = max(top_probs) if top_probs.size > 0 else 0.001
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width - (max_prob_display * 0.05) if width > (max_prob_display * 0.2) else width + (max_prob_display * 0.01)
        ha = 'right' if width > (max_prob_display * 0.2) else 'left'
        color = 'white' if width > (max_prob_display * 0.2) else 'black'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2., f'{width:.4f}',
                va='center', ha=ha, fontsize=8, color=color, fontweight='medium')

    # Adjust x-limit
    ax.set_xlim(left=0, right=max(top_probs) * 1.15 if top_probs.size > 0 else 0.1)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)


def print_terminal_visualization(original_logits, modified_logits, tokenizer, allowed_ids, vocab_size, top_k=10):
    """Prints a side-by-side comparison of top token probabilities in the terminal."""
    print("\n--- Terminal Logit Visualization (Top {}) ---".format(top_k))

    def get_top_tokens(logits, k):
        # Handle potential NaN/Inf values
        logits_copy = logits.copy()
        logits_copy[np.isneginf(logits_copy)] = -1e9
        logits_copy = np.nan_to_num(logits_copy, nan=-1e9)

        # Softmax
        logits_max = np.max(logits_copy)
        exp_logits = np.exp(logits_copy - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Get top k
        k = min(k, len(probs))
        # Ensure indices are within valid range [0, vocab_size-1]
        partitioned_indices = np.argpartition(probs, -k)[-k:]
        valid_indices = [idx for idx in partitioned_indices if 0 <= idx < vocab_size]
        if not valid_indices:
             return [], [], [] # Return empty if no valid indices found

        # Sort the valid top-k indices by probability
        valid_indices = np.array(valid_indices)
        top_indices = valid_indices[np.argsort(probs[valid_indices])][::-1] # Descending order

        top_probs = probs[top_indices]

        # Decode tokens
        top_tokens_str = []
        for idx in top_indices:
            try:
                token_str = tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                token_str = ''.join(c if c.isprintable() else f'[{ord(c):x}]' for c in token_str)
                if not token_str.strip(): token_str = f'[WS {idx}]'
                top_tokens_str.append(f"{token_str} ({idx})")
            except Exception:
                top_tokens_str.append(f'[DecodeErr {idx}]')

        return top_indices, top_probs, top_tokens_str

    # Get top tokens for both original and modified logits
    orig_indices, orig_probs, orig_tokens = get_top_tokens(original_logits, top_k)
    mod_indices, mod_probs, mod_tokens = get_top_tokens(modified_logits, top_k)

    # Determine max length for formatting
    max_len_orig = max(len(t) for t in orig_tokens) if orig_tokens else 0
    max_len_mod = max(len(t) for t in mod_tokens) if mod_tokens else 0
    col_width = max(max_len_orig, max_len_mod, 12) + 2 # Ensure minimum width + padding

    # Print header
    print(f"{'Original Probabilities':<{col_width+10}} | {'Modified Probabilities':<{col_width+10}}")
    print(f"{'Token (ID)':<{col_width}} {'Prob.':<8} | {'Token (ID)':<{col_width}} {'Prob.':<8} | {'Constraint'}")
    print("-" * (col_width + 9) + "-|-" + "-" * (col_width + 9) + "-|-" + "-" * 12)

    # Print rows
    num_rows = max(len(orig_tokens), len(mod_tokens))
    for i in range(num_rows):
        orig_part = f"{'':<{col_width}} {'':<8}" # Placeholder for alignment
        if i < len(orig_tokens):
            orig_token_str = orig_tokens[i]
            orig_prob_str = f"{orig_probs[i]:.4f}"
            orig_part = f"{orig_token_str:<{col_width}} {orig_prob_str:<8}"

        mod_part = f"{'':<{col_width}} {'':<8}" # Placeholder for alignment
        constraint_info = ""
        if i < len(mod_tokens):
            mod_token_str = mod_tokens[i]
            mod_prob_str = f"{mod_probs[i]:.4f}"
            is_allowed = mod_indices[i] in allowed_ids

            # Simplified constraint marker
            if mod_probs[i] > 1e-9: # Only mark if probability is non-negligible
                constraint_info = "Allowed" if is_allowed else "Suppressed"
            else:
                constraint_info = "" # Don't mark tiny probabilities

            mod_part = f"{mod_token_str:<{col_width}} {mod_prob_str:<8}"

        print(f"{orig_part} | {mod_part} | {constraint_info}")


# Check if the constraint was applied and logits were captured
if constraint_processor.applied_constraint and constraint_processor.original_logits is not None and constraint_processor.modified_logits is not None:
    print("\n--- Preparing Visualization ---")

    # Ensure vocab_size is correctly retrieved from the model config
    current_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else tokenizer.vocab_size

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Logit Modification Comparison at Step {prompt_len}\nPrompt: \"{prompt}\"", fontsize=14, y=0.99)

    # Plotting functions call
    plot_probs(constraint_processor.original_logits, "Original Probabilities (Top 15)", ax1, vocab_size=current_vocab_size)
    plot_probs(constraint_processor.modified_logits, f"Modified Probabilities (Constrained to Yes/No/Maybe)", ax2, allowed_ids=allowed_ids, vocab_size=current_vocab_size)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the plot instead of showing
    save_path = "logit_comparison.png"
    try:
        plt.savefig(save_path)
        print(f"\nGraph saved to {save_path}")
    except Exception as e:
        print(f"\nError saving graph: {e}")
    plt.close(fig) # Close the figure to free memory

    # Call the terminal visualization function
    print_terminal_visualization(
        constraint_processor.original_logits,
        constraint_processor.modified_logits,
        tokenizer,
        allowed_ids,
        current_vocab_size,
        top_k=10 # Show top 10 in terminal
    )

else:
    print("\nConstraint was not applied as expected or visualization data is missing.")
    print(f"Constraint applied flag: {constraint_processor.applied_constraint}")
    print(f"Original logits captured: {constraint_processor.original_logits is not None}")
    print(f"Modified logits captured: {constraint_processor.modified_logits is not None}")
