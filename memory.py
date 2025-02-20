def compute_memory(n_i_n_o, P, batch_size=1, embedding_size=768):
    """
    Computes the model memory, input memory, and attention memory for a transformer model.
    
    Args:
    n_i_n_o (int): Combined value of n_i + n_o (input size + number of generated tokens)
    P (int): Number of parameters in the model
    batch_size (int): Batch size for the computation
    embedding_size (int): Size of the embeddings (typically 768 for GPT-2-like models)
    
    Returns:
    None
    """
    # Constants
    precision_bytes = 4  # 4 bytes for each parameter (32-bit precision)
    
    # 1. Model Memory (based on number of parameters)
    model_memory = P * precision_bytes  # Memory taken by model parameters

    # 2. Input Memory (based on input size n_i and batch size)
    input_memory = n_i_n_o * batch_size * embedding_size * precision_bytes

    # 3. Attention Memory (scales quadratically with the sequence length)
    attention_memory = (n_i_n_o ** 2) * batch_size * precision_bytes

    # Total Memory
    total_memory = model_memory + input_memory + attention_memory

    # Print Results
    print(f"Model Memory: {model_memory / (1024 ** 3):.2f} GB")
    print(f"Input Memory: {input_memory / (1024 ** 3):.2f} GB")
    print(f"Attention Memory: {attention_memory / (1024 ** 3):.2f} GB")
    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")

# Example Usage
compute_memory(512, 124000000, batch_size=1, embedding_size=768)  # Adjust n_i+n_o and P as needed.
