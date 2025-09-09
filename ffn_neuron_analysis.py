import copy
import random
import os
import argparse
import torch
import json
import numpy as np
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
import transformers
import inspect  
from collections import defaultdict, OrderedDict
torch.set_grad_enabled(False)


class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the newly generated part
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return '\n' in decoded_text


# Set HF_HOME if not already set
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = "/local/raehyuk/.cache/huggingface"
    print(f"Set HF_HOME to {os.environ['HF_HOME']}")
else:
    print(f"Using existing HF_HOME: {os.environ['HF_HOME']}")

from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Inference")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--vision_tower", type=str, required=True,
                        help="Vision tower model path or identifier")
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu",
                        help="MLP projector type: linear, mlp2x_gelu, etc.")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2,
                        help="Vision feature selection layer")
    parser.add_argument("--mm_use_im_start_end", type=bool, default=False,
                        help="Use image start/end tokens")
    parser.add_argument("--mm_use_im_patch_token", type=bool, default=False,
                        help="Use image patch tokens")
    parser.add_argument("--tune_mm_mlp_adapter", type=bool, default=False,
                        help="Tune MM MLP adapter")
    parser.add_argument("--version", type=str, default="plain",
                        help="Conversation template version")
    parser.add_argument("--model_max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--conv_mode", type=str, default="plain",
                        help="Conversation mode: plain, llava_v0, llava_v1, etc.")
    
    # Device and precision settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for accessing gated models (can also be set via HF_TOKEN env var)")
    parser.add_argument("--test_path", type=str)
    # generate_mode (store true default false)
    parser.add_argument("--generate_mode", action="store_true", help="Generate mode")
    parser.add_argument("--weight_path",type=str,default=None,
                        help="Path to weight file")
    parser.add_argument("--output_path",type=str,default=None,
                        help="Path to output file")
    parser.add_argument("--sample_prop",type=float,default=None,
                        help="Proportion of data to sample")
    parser.add_argument("--sample_num",type=int,default=None,
                        help="Number of data to sample")
    parser.add_argument("--image_dir",type=str,default=None,
                        help="Path to image directory")
    return parser.parse_args()

def load_model(args):
    """Load LLaVA model with the specified arguments"""
    
    # Get HuggingFace token from env if not provided in args
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token and ('meta-llama' in args.model_name_or_path.lower() or 
                         'llama' in args.model_name_or_path.lower()):
        print("Warning: Loading a Meta Llama model without an access token. This may fail.")
    elif hf_token:
        print("Using HuggingFace token for model access")
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        token=hf_token,
    )
    
    # Ensure pad token is set for tokenizer
    if tokenizer.pad_token is None:
        if 'llama' in args.model_name_or_path.lower():
            # Special handling for Llama models
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Added [PAD] token for Llama model. New vocab size: {len(tokenizer)}")
        else:
            # Fall back to using unk token as pad token
            tokenizer.pad_token = tokenizer.unk_token
            print(f"Using {tokenizer.pad_token} as pad token")
    
    # Determine precision
    compute_dtype = (torch.bfloat16 if args.bf16 else 
                    (torch.float16 if args.fp16 else torch.float32))
    
    # Load model
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        token=hf_token,
    )
    
    # Initialize vision modules
    model_args = argparse.Namespace(
        vision_tower=args.vision_tower,
        mm_vision_select_layer=args.mm_vision_select_layer,
        mm_projector_type=args.mm_projector_type,
        mm_use_im_start_end=args.mm_use_im_start_end,
        mm_use_im_patch_token=args.mm_use_im_patch_token,
        mm_vision_select_feature="patch",
        pretrain_mm_mlp_adapter=None,
        mm_patch_merge_type="flat",
        tune_mm_mlp_adapter=args.tune_mm_mlp_adapter,
        version=args.conv_mode,
        freeze_backbone=False,
    )
    
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=None
    )
    
    # Move vision tower to appropriate device and dtype
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=args.device)
    
    # Set model configuration
    model.config.image_aspect_ratio = 'square'
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.image_token_index = IMAGE_TOKEN_INDEX
    
    # Set additional model configuration based on args
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
    
    # Initialize vision tokenizer
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    # Move model to device
    model.to(args.device)
 
    
    return model, tokenizer, vision_tower.image_processor


def print_model_dtypes(model):
    """Print the data types of the model's parameters"""
    for name, param in model.named_parameters():
        if 'mm_projector' in name or name.startswith('vision_tower') or name.endswith('.weight'):
            print(f"{name}: {param.dtype}")

def decoding(_input_toks, tokenizer):
    with torch.no_grad():
        decoded = ''
        _input_toks = _input_toks.squeeze(0).tolist()
        for _tok in _input_toks:
            # print(f'decoding: {_tok}')
            if _tok == -200:
                decoded += '<image> '
            elif _tok == -100:
                decoded += '<ignore> '
            else:
                decoded += tokenizer.decode(_tok)
    return decoded
    

def random_sample_num_samples(data, num_samples):
    random.seed(42)
    random.shuffle(data)
    return data[:num_samples]

def random_sample_by_prop(data, prop):
    random.seed(42)
    random.shuffle(data)
    return data[:int(len(data)*prop)]



    
def observe_pre_x(model):

    forward_acts = defaultdict(list)   # layer_idx -> list of pre_act tensors
    grads    = defaultdict(list)   # layer_idx -> list of grad tensors
    handles  = []
    # 2) Hook factories
    def make_fwd_hook(layer_idx):
        def fwd_hook(module, inp, out):
            # out is pre_act_x = up_proj(x)
            # detach & clone if you plan to accumulate
            ___ = out.detach().clone()
            # print(f'___ shape: {___.shape}')
            forward_acts[layer_idx].append(___)
        return fwd_hook

    def make_bwd_hook(layer_idx):
        def bwd_hook(module, grad_in, grad_out):
            # grad_out[0] is dL/d(pre_act_x)
            # print(f'len_grad_out: {len(grad_out)}')
            __ = grad_out[0].detach().clone()
            # print(f'__ shape: {__.shape}')
            grads[layer_idx].append(__)
        return bwd_hook

    for idx, layer in enumerate(model.model.layers):
        up = layer.mlp.up_proj
        fh = up.register_forward_hook( make_fwd_hook(idx) )
        bh = up.register_full_backward_hook( make_bwd_hook(idx) )
        handles.extend([fh, bh])
    return forward_acts, grads, handles


def observe_post_x(model):
    forward_acts = defaultdict(list)   # layer_idx -> list of post_act_x tensors
    grads= defaultdict(list)   # layer_idx -> list of ∂L/∂post_act_x tensors
    handles   = []

    # Forward hook factory: grabs the *input* to down_proj, i.e. post_act_x
    def make_fwd_hook(layer_idx):
        def fwd_hook(module, inp, out):
            # inp is a tuple; inp[0] is post_act_x
            pa = inp[0].detach().clone()
            forward_acts[layer_idx].append(pa)
        return fwd_hook

    # Backward hook factory: grabs gradient w.r.t. that same input
    def make_bwd_hook(layer_idx):
        def bwd_hook(module, grad_in, grad_out):
            # grad_in is a tuple; grad_in[0] is ∂L/∂post_act_x
            gpa = grad_in[0].detach().clone()
            grads[layer_idx].append(gpa)
        return bwd_hook

    # Register on every layer’s down_proj
    for idx, layer in enumerate(model.model.layers):
        down = layer.mlp.down_proj
        fh = down.register_forward_hook(    make_fwd_hook(idx) )
        bh = down.register_full_backward_hook(make_bwd_hook(idx))
        handles.extend([fh, bh])

    return forward_acts, grads, handles

def get_top_k_neurons(forward_acts, grads, model, configs, head_w):
    top_toks_per_layer = defaultdict(list)
    top_keys_per_layer = dict()
    # neuron_act_per_layer = dict()
    attr_vecs = dict()

    for idx in forward_acts:
        linear = model.model.layers[idx].mlp.down_proj
        U  = forward_acts[idx][0]    # shape [1, seq_len, hidden]
        # same shape
        U_tgt  = U[:, -1, :]
        # print(f'U.shape: {U.shape} and U_tgt.shape: {U_tgt.shape}')
        direction = -1
        if configs['use_grad']:
            dU = grads[idx][0] 
            dU_tgt = dU[:, -1, :]
            attr_vec = (U_tgt * direction*dU_tgt).sum(dim=(0,1))  # [hidden]            
        else:
            # attr_vec = (U_tgt).sum(dim=(0,1))  # [hidden]          
            attr_vec = U_tgt.squeeze(0)  
        
        # print(f'attr_vec.shape: {attr_vec.shape}')
        top_k = configs['num_neurons']
        top_vals, top_idxs = torch.topk(attr_vec, top_k, largest=True, sorted=True)
        top_keys_per_layer[idx] = top_idxs.tolist()
        attr_vecs[idx] = attr_vec


        for ii, _idx in enumerate(top_idxs):
            value_vector = linear.weight[:, _idx]
            token_logits = head_w @ value_vector
            token_vals, token_ids = torch.topk(token_logits, 3, largest=True, sorted=True)
            top_toks_per_layer[idx].extend(token_ids.tolist())
    return top_toks_per_layer, attr_vecs, top_keys_per_layer

def get_top_n_per_layer(attr_vecs: dict[int, torch.Tensor],
                        top_n: int) -> dict[int, list[int]]:
    """
    For each layer, pick the top_n neurons by attribution score.

    Args:
        attr_vecs: dict mapping layer_idx -> attribution vector (Tensor[hidden])
        top_n:     number of neurons to select per layer

    Returns:
        ablate_map: dict mapping layer_idx -> list of top_n neuron indices
    """
    ablate_map: dict[int, list[int]] = {}
    for layer_idx, vec in attr_vecs.items():
        # vec: Tensor of shape [hidden]
        # topk returns (values, indices)
        _, top_idxs = torch.topk(vec, top_n, largest=True, sorted=True)
        ablate_map[layer_idx] = top_idxs.tolist()
    return ablate_map

def get_random_n_per_layer(attr_vecs: dict[int, torch.Tensor],
                           top_n: int,
                           generator: torch.Generator = torch.Generator()
                          ) -> dict[int, list[int]]:
    """
    For each layer, pick `top_n` random neurons to ablate.

    Args:
        attr_vecs: dict mapping layer_idx -> attribution vector (Tensor[hidden])
                   We only use this to infer the hidden dimension for each layer.
        top_n:     number of neurons to select per layer
        generator: optional torch.Generator for reproducible randomness

    Returns:
        ablate_map: dict mapping layer_idx -> list of `top_n` randomly chosen neuron indices
    """
    ablate_map: dict[int, list[int]] = {}
    for layer_idx, vec in attr_vecs.items():
        hidden_size = vec.shape[0]
        if top_n > hidden_size:
            raise ValueError(f"top_n ({top_n}) exceeds hidden size ({hidden_size}) at layer {layer_idx}")
        # sample without replacement
        perm = torch.randperm(hidden_size, generator=generator)
        random_idxs = perm[:top_n].tolist()
        ablate_map[layer_idx] = random_idxs
    return ablate_map

def get_top_n_neurons_all_layers(attr_vecs: dict[int, torch.Tensor],
                                 top_n: int):
    """
    attr_vecs: { layer_idx: Tensor[hidden] }
    top_n: how many total neurons to pick across all layers
    returns: List[(layer_idx, neuron_idx)] of length top_n, sorted by descending score
    """
    # Sort layer keys so we can reconstruct mapping later
    layers = sorted(attr_vecs.keys())
    # Stack into a single tensor: shape [num_layers, hidden]
    mat = torch.stack([attr_vecs[l] for l in layers], dim=0)
    num_layers, hidden = mat.shape

    # Flatten to shape [num_layers * hidden]
    flat = mat.view(-1)
    # Top-N in the flattened vector
    top_vals, top_idxs = torch.topk(flat, top_n, largest=True, sorted=True)
    # print(f'top_vals: {top_vals} top_idxs: {top_idxs}')
    # Convert flat idx back to (layer, neuron)
    results = []
    for idx in top_idxs.tolist():
        layer_idx = idx // hidden
        neuron_idx = idx % hidden
        results.append((layers[layer_idx], neuron_idx))

    return results


def get_random_neurons(forward_acts, grads, model,  configs, head_w):
    idx_per_layer = defaultdict(list)

    # iterate over those layers you actually have activations for
    for idx in forward_acts.keys():              
        # grab the same linear you use for top_k
        linear = model.model.layers[idx].mlp.down_proj
        
        U  = forward_acts[idx][0]   # [1, seq_len, hidden]
        # pick a random subset of the hidden neurons
        hidden_dim     = U.shape[2]
        perm           = torch.randperm(hidden_dim)
        random_neurons = perm[: configs['num_neurons']]

        # now unembed each of those random neurons
        for neuron_idx in random_neurons.tolist():
            value_vector = linear.weight[:, neuron_idx]       # [hidden]
            token_logits = head_w @ value_vector              # [vocab_size]
            token_vals, token_ids = torch.topk(token_logits, 
                                               3, largest=True, sorted=True)
            idx_per_layer[idx].extend(token_ids.tolist())

    return idx_per_layer

def decode(neurons_per_layer, tokenizer):
        toks_per_layer = {}
        for layer_idx in neurons_per_layer:
            toks = neurons_per_layer[layer_idx]
            toks = [tokenizer.decode([tid]) for tid in toks]
            toks_per_layer[layer_idx] = toks
        return toks_per_layer


def make_ablation_pre_hook(neuron_idxs: list[int]):
    """
    Returns a forward‐pre‐hook that zeroes out the specified neuron channels
    in the module’s *input* tensor (i.e. the post‐activation vector).
    """
    def pre_hook(module, inputs):
        # inputs is a tuple; inputs[0] is the tensor of shape [..., hidden_dim]
        x = inputs[0]
        # clone so we don’t mutate upstream state
        x = x.clone()
        x[..., neuron_idxs] = 0.0
        # return a new inputs-tuple
        return (x,)
    return pre_hook


def register_ablation_pre_hooks(model, ablate_map: dict[int, list[int]]):
    """
    Register one forward‐pre‐hook on each layer’s down_proj so that
    before down_proj runs, certain hidden‐dims are zeroed.

    Returns the list of handles so you can remove them later.
    """
    handles = []
    for layer_idx, neuron_idxs in ablate_map.items():
        down = model.model.layers[layer_idx].mlp.down_proj
        h = down.register_forward_pre_hook(make_ablation_pre_hook(neuron_idxs))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()
        

def main():
    # Parse arguments
    args = parse_args()
    
    # Set conversation mode if not specified
    if args.conv_mode == "plain":
        args.conv_mode = args.version  # Use the version as conv_mode if not specified
    
    # Load model and tokenizer
    model, tokenizer, image_processor = load_model(args)
    
    # I want to load /local/raehyuk/LLaVA/checkpoints/llama3b-pretrain/checkpoint-8721/mm_projector.bin
    # and print the data type of the parameters
    # mm_projector = torch.load('/local/raehyuk/LLaVA/checkpoints/llama3b-pretrain/checkpoint-8721/mm_projector.bin')
    mm_projector = torch.load(args.weight_path)
    # load weights of mm_projector to model
    match_result = model.load_state_dict(mm_projector, strict=False)
    
    # mm_projector has very little part of what the model has. It is okay that missing keys are found. But the weights in mm_projector must be loaded in to the model
    # write assertion: that there must be no unexpected keys in mm_projector
    assert len(match_result.unexpected_keys) == 0, "Unexpected keys found in mm_projector"
    
    # Convert entire model to bfloat16
    print("Converting model to bfloat16")
    model.to(torch.bfloat16)
    model.eval()
    head_w = model.lm_head.weight.detach()
    
    input_data_name = args.test_path.split('/')[-1].split('.')[0]
    print(f'input_data_name: {input_data_name}')
    # configs_1 = {'observe_target':'post_x', 'use_grad':True, 'output_dir': f'neuron_analysis_results2/{input_data_name}_post_x_use_grad', 'num_neurons':3}
    configs = {'observe_target':'post_x', 'use_grad':False, 'num_neurons':3}

    
    forward_acts, grads, handles = observe_post_x(model)
    

    input_data_file = args.test_path
    with open(input_data_file, 'r') as fp:
        data = json.load(fp)
    if 'data' in data:
        data = data['data']

    args.sample_num 
    args.sample_prop
    os.makedirs(args.output_path, exist_ok=True)
    # only one must be provided
    assert int(args.sample_num is None)+int(args.sample_prop is None) == 1, "Only one of sample_num or sample_prop must be provided"

    if args.sample_num is not None:
        sampled_data = random_sample_num_samples(data, args.sample_num)
    else:
        sampled_data = random_sample_by_prop(data, args.sample_prop)

    
    print(f'Before sampling: {len(data)} After sampling: {len(sampled_data)}')
   
    output_dict = {}

    num_layers = len(model.model.layers)
    average_attr_vecs = OrderedDict()
    for i in range(num_layers):
        average_attr_vecs[i] = np.zeros(8192, dtype=np.float32)

    for data_i, elem in enumerate(sampled_data):    
        print(f'{data_i} / {len(sampled_data)}')
        _synset = elem['synset']
        prompt = DEFAULT_IMAGE_TOKEN+' '+elem['conversations'][1]['value'] +'\n'
        
        classname = elem['conversations'][1]['value'].split('] ')[-1]

        image_fp = args.image_dir+'/'+elem['image']
        print(f'image_fp: {image_fp}')
        image = Image.open(image_fp)
        image_tensor = process_images([image], image_processor, model.config)[0]
    
        # Convert image tensor to match model dtype
        image_tensor = image_tensor.to(dtype=model.dtype, device=args.device)
        new_prompt = prompt.split(']')[0]+']'
        
        mask_input_ids = tokenizer_image_token(new_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
        
        # labels = copy.deepcopy(input_ids)
        # labels[0, :mask_input_ids.shape[1]] = -100
        
        # with torch.no_grad():
            # print(f'labels: {labels.shape}')
            # non_ignores = labels[0, mask_input_ids.shape[1]:]
            
        
        # with torch.no_grad():
        #     num_target_tokens = labels.shape[1]-mask_input_ids.shape[1]
        #     if num_target_tokens > 2:
        #         print(f'num_target_tokens > 1, skipping non_ignores: {non_ignores} and {tokenizer.decode(non_ignores, skip_special_tokens=True)}')
        #         continue

        # print(f'num_target_tokens: {num_target_tokens}')        
        # print('\n')
        forward_acts.clear()
        grads.clear()  
        print(f'mask_input_ids: {mask_input_ids.shape}')
        output = model(input_ids=mask_input_ids, images=image_tensor.unsqueeze(0))

        top_k_neurons, attr_vecs, top_keys_per_layer = get_top_k_neurons(forward_acts, grads, model, configs, head_w)
        random_neurons = get_random_neurons(forward_acts, grads, model, configs, head_w)
        
        top_toks_per_layer = decode(top_k_neurons, tokenizer)
        random_toks_per_layer = decode(random_neurons, tokenizer)
        output_dict['image_fp'] = image_fp
        output_dict['gt_label'] = classname
        # output_dict['loss'] = output.loss.item()
        output_dict['top_toks_per_layer'] = top_toks_per_layer
        output_dict['random_toks_per_layer'] = random_toks_per_layer
        output_dict['top_keys_per_layer'] = top_keys_per_layer
        output_dict['gt_synset'] = _synset
        output_dict['prompt'] = elem['conversations'][1]['value']

        print(f'saving to {os.path.join(args.output_path, f"{data_i}.json")}')
        with open(os.path.join(args.output_path, f'{data_i}.json'), 'w') as fp:
            json.dump(output_dict, fp, indent=4)
        
        os.path.join(args.output_path, f'{data_i}.npz')
        
        for i in range(num_layers):
            average_attr_vecs[i] += attr_vecs[i].to(torch.float32).cpu().detach().numpy()

        # save average_attr_vecs to a file
        # stacked_attr_vecs = np.stack([attr_vecs[i].to(torch.float32).cpu().detach().numpy() for i in range(num_layers)], axis=0)
        # print(f'stacked_attr_vecs.shape: {stacked_attr_vecs.shape} and attr_vecs.keys(): {attr_vecs.keys()}')
        # np.savez_compressed(os.path.join(args.output_path, f'{data_i}.npz'), stacked_attr_vecs=stacked_attr_vecs)

        output_dict = {}
        model.zero_grad()

    # After processing all data, save the average attribution vectors to a file
    average_attr_vecs_np = np.stack([average_attr_vecs[i] for i in range(num_layers)], axis=0)
    np.save(os.path.join(args.output_path, "attr_vecs.npy"), average_attr_vecs_np)
        
        
if __name__ == "__main__":
    import transformers
    transformers.logging.set_verbosity_error()
    main() 

