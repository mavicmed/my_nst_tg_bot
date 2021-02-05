from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from gif import create_gif
from logger import logger

imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize),  # нормируем размер изображения
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # превращаем в удобный формат

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target )#to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    batch_size , h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)

    features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
    def __init__(self, target_feature, target_feature2):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.target2 = gram_matrix(target_feature2).detach()
        self.loss = F.mse_loss(self.target, self.target)# to initialize with something

    def forward(self, input):
        half = input.shape[3] // 2
        style1 = input.clone()
        style1[:,:,:,:half] = input[:,:,:,half:]
        g1 = gram_matrix(style1)
        l1 = F.mse_loss(g1, self.target)

        style2 = input.clone()
        style2[:,:,:,half:] = input[:,:,:,:half]
        g2 = gram_matrix(style2)
        l2 = F.mse_loss(g2, self.target2)

        self.loss = l1 + l2
        return input

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
content_layers_default = ['conv_3',]
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

cnn = models.vgg19(pretrained=True).features.to(device).eval()

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, style_img2, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            #Переопределим relu уровень
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            target_feature2 = model(style_img2).detach()
            style_loss = StyleLoss(target_feature, target_feature2)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    #выбрасываем все уровни после последенего styel loss или content loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(path_res, cnn, normalization_mean, normalization_std,
                        content_img, style_img, style_img2, input_img, num_steps=600,
                        style_weight=50000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, style_img2, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        if run[0] % 10 == 0:
            fig = plt.figure(frameon=False, dpi=1)
            plt.imshow(input_img.data.clamp_(0, 1).cpu().squeeze(0).numpy().transpose([1,2,0]), )
            #plt.imsave(output, 'output.png')
            # sphinx_gallery_thumbnail_number = 4
            plt.axis('off')
            plt.ioff()
            fig.set_size_inches(512, 512)
            fig.savefig(f'{path_res}/result/{run[0]}.png', bbox_inches='tight')
            plt.close('all')

        def closure():
            # correct the values
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            #взвешивание ощибки
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()


            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()


            return style_score + content_score

        optimizer.step(closure)



#             optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


unloader = transforms.ToPILImage()
def ran_nst_(user, bot, *args, **kwargs):
    path = user.image_folder / 'result'
    path.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    path = user.image_folder
    style_img2 = image_loader(f'{path}/1.jpg')
    style_img = image_loader(f'{path}/2.jpg')
    content_img = image_loader(f'{path}/0.jpg')
    # input_img = content_img.clone()
    input_img = torch.randn(content_img.data.size(), device=device)
    output = run_style_transfer(path, cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, style_img2, input_img)
    bot.send_message(user.uid, 'Стиль перенесен, сейчас покажу что получилось')
    res = unloader(output.data.cpu().squeeze(0))
    res.save(f'{user.image_folder}/result.jpg')
    i = bot.send_photo(user.uid, res)
    i.wait()
    logger.sent_result_img(user)
    bot.send_message(user.uid, 'Сейчас покажу весь процесс как я это сделал')
    ani = create_gif(user)
    with open(f'{user.image_folder}/result.gif', 'rb') as ani:
        i = bot.send_document(user.uid, ani)
        i.wait()
        logger.sent_result_gif(user)
    user.get_default()


# input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
# plt.figure()
# imshow(input_img, title='Input Image')
# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_img2, input_img)
