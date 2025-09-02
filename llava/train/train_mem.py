from llava.train.train import train
# I want to check visible devices
# # set visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# time.sleep(100)

if __name__ == "__main__":
    # Use SDPA attention which is optimized in PyTorch 2.0+
    attn_implementation = "flash_attention_2"
    # attn_implementation = 'sdpa'
    train(attn_implementation=attn_implementation)
