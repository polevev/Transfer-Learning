import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import streamlit as st

#Для загрузки изображений
def image_loader(image, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

#Преобразование тензора в картинку
def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#COntentLoss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        n_H, n_W, n_C = input.shape[2], input.shape[3], input.shape[1]
        self.loss = (nn.functional.mse_loss(input, self.target))/(4 * n_H * n_W * n_C)
        return input
    
def gram_matrix(input):
    batch_size, feature_maps, h, w = input.size()
    features = input.view(batch_size * feature_maps, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * feature_maps * h * w)

# Класс для вычисления style loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Оборачиваем в тензоры и добавляем размерность
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def total_variation_loss(img):
    tv_h = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    tv_w = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return tv_h + tv_w

#Основная функция
def run_style_transfer(content_img, style_img, num_steps=200,
                       style_weight=1e5, content_weight=1):
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    #Нормализация
    normalization = Normalization(mean=cnn_normalization_mean, std = cnn_normalization_std).to(device)
    #Слои
    content_layers = ['block5_conv_2']

    style_layers =  ['block1_conv_1', 'block2_conv_1', 'block3_conv_1', 'block4_conv_1', 'block5_conv_1']
    
    #Сборка модели
    vgg_layers = {
        '0': 'block1_conv_1',
        '5': 'block2_conv_1',
        '10': 'block3_conv_1',
        '19': 'block4_conv_1',
        '28': 'block5_conv_1',
        '30': 'block5_conv_2'
    }

    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    for idx, layer in enumerate(cnn.children()):
        # Для всех ReLU слоёв явно задаём inplace=False
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False) 
        # Для MaxPool используем ceil_mode=False (как в оригинале VGG)
        if isinstance(layer, nn.MaxPool2d):
            layer = nn.MaxPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                ceil_mode=False 
            )
        name = vgg_layers.get(str(idx))
        model.add_module(name if name else f'layer_{idx}', layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{name}', content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{name}', style_loss)
            style_losses.append(style_loss)
    #Удаляем все слои после последнего loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1].to(device)
    #Оптимизируем изображение
    input_img = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([input_img])
    for epoch in tqdm(range(num_steps)):
        def closure():
            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight*style_score + content_weight*content_score + 1e-6*total_variation_loss(input_img)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        if epoch%10==0:
            print(f'Loss(Style+content+var): {loss.item():.4f}')
            st.image(tensor_to_pil(input_img), caption=f'Epoch: {epoch}', width=512)
    print(f'Final Loss(Style+content+var): {loss.item():.4f}')
    return input_img