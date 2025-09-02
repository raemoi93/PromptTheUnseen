from llava.train.train import train

if __name__ == "__main__":
    # Use SDPA attention which is optimized in PyTorch 2.0+
    # attn_implementation = "flash_attention_2"
    attn_implementation = 'sdpa'
    train(attn_implementation=attn_implementation)
