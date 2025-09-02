from dino_encoder import DINOVisionTower
from clip_encoder import CLIPVisionTower
from mae_encoder import MAEVisionTower
from vit_encoder import ViTVisionTower
from PIL import Image
import torch


image_fp = "/local/raehyuk/LLaVA/vg_object_centric_data_snd/train_seen_vis/1_bell_bbox.jpg"
image = Image.open(image_fp).convert("RGB")
device = torch.device(7)

test_dino = False
test_clip = False
test_mae = False
test_vit = True

if test_dino:
    model_code = "facebook/dinov2-large"
    model = DINOVisionTower(
        vision_tower=model_code,
        args=type('args', (), {'mm_vision_select_layer': -2})(),  # Dummy args object with attribute
        delay_load=False
    )
    model.load_model()
    model.to(device)
    num_param = sum(p.numel() for p in model.parameters())
    print(f'number of parameters in the model for dino: {num_param}')
    pixel_values = model.image_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=model.dtype)
    # Forward the image
    features = model(pixel_values)  # Input as list if needed

    print("DINOv2 features shape:", features.shape)

elif test_clip:
    model_code = "openai/clip-vit-large-patch14-336"
    
    model = CLIPVisionTower(
        vision_tower=model_code,
        args=type('args', (), {'mm_vision_select_layer': -2})(),  # Dummy args object with attribute
        delay_load=False
    )
    model.load_model()
    for n, p in model.named_parameters():
        print(n, p.device)
    # exit()
    # print out the number of parameters in the model
    num_param = sum(p.numel() for p in model.parameters())
    print(f'number of parameters in the model for clip: {num_param}')
 
    # Forward the image
    image = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    print(image.shape)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    print(image.shape, 'after unsqueeze')

    image_features = model(image)

    print("CLIP features shape:", image_features.shape)

elif test_mae:
    # model_code = "facebook/vit-mae-large"
    model_code = 'facebook/webssl-mae300m-full2b-224'

    model = MAEVisionTower(
        vision_tower=model_code,
        args=type('args', (), {'mm_vision_select_layer': -2})(),  # Dummy args object with attribute
        delay_load=False
    )
   
    model.to(device)
    num_param = sum(p.numel() for p in model.parameters())

    image = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    print(image.shape)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    print(image.shape, 'after unsqueeze')
    image_features = model(image)
    print(f'image_features shape: {image_features.shape}')


elif test_vit:
    model_code = 'google/vit-large-patch16-224-in21k'
    model = ViTVisionTower(
        vision_tower=model_code,
        args=type('args', (), {'mm_vision_select_layer': -2})(),  # Dummy args object with attribute
        delay_load=False
    )
    model.to(device)
    num_param = sum(p.numel() for p in model.parameters())
    print(f'number of parameters in the model for vit: {num_param}')

    image = model.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    print(image.shape)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    print(image.shape, 'after unsqueeze')
    image_features = model(image)
    print(f'image_features shape: {image_features.shape}')