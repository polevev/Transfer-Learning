from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Load/Save ---
def image_loader(img, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()
    ])
    image = loader(img).unsqueeze(0).to(device)
    return image.to(device, torch.float)

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    return image

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    content_mean, content_std = content_feat.mean([2, 3], keepdim=True), content_feat.std([2, 3], keepdim=True)
    style_mean, style_std = style_feat.mean([2, 3], keepdim=True), style_feat.std([2, 3], keepdim=True)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean

def get_encoder():
    vgg = models.vgg19(pretrained=True).features[:21].eval()  # до relu4_1
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3)
        )
    
def stylize_adain(content_img, style_img, decoder, encoder, alpha=1.0, imsize=512):
    content = image_loader(content_img, imsize)
    style = image_loader(style_img, imsize)
    with torch.no_grad():
        c_feat = encoder(content)
        s_feat = encoder(style)
        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat
        stylized = decoder(t)
    return tensor_to_pil(stylized)