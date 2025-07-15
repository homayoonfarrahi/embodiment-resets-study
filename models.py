from collections import OrderedDict
import itertools
from relod.utils import random_augment
import torch
from torch import nn, rand
from relod.algo.models import ActorModel, CriticModel, EncoderModel, SpatialSoftmax, squash, gaussian_logprob, LOG_STD_MIN, LOG_STD_MAX
from torchvision import models

def count_params(model):
    for name, p in model.named_parameters():
        train = p.numel() if p.requires_grad else 0
        all = p.numel()
        print(f'{name}: {train}, {all}')

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class PNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        norm = norm.clamp(min=self.eps) # avoid div by zero
        x = torch.div(x, norm)
        return x

class ResnetLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super(ResnetLayerNorm, self).__init__()
        # self.ln = nn.LayerNorm([9, 88, 156])
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # print('x: ', x.shape)
        # return self.ln(x)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # return self.ln(x.transpose(1, -1)).transpose(1, -1)

class EncoderModelResNet18(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, layernorm):
        super().__init__()
        use_imgnet = False
        c, h, w = image_shape
        self.rad_h = round(rad_offset * h)
        self.rad_w = round(rad_offset * w)
        if False: #layernorm:
            self.model = models.resnet18(num_classes=1000, pretrained=use_imgnet, norm_layer=ResnetLayerNorm)
        else:
            self.model = models.resnet18(num_classes=1000, pretrained=use_imgnet)
        self.latent_dim = net_params['latent'] + proprioception_shape[0]

        # change conv1 to support 9 channels using pre-trained weights
        self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if use_imgnet:
            # copy the pre-trained imagenet weights to conv1
            w_c1 = self.model.conv1.weight
            w_c1_9ch = torch.cat((w_c1 / 3., w_c1 / 3., w_c1 / 3.), dim=1)
            self.model.conv1.requires_grad_(False)
            self.model.conv1.weight.copy_(nn.parameter.Parameter(w_c1_9ch, requires_grad=False))
            self.model.requires_grad_(False)

        self.model.fc = nn.Linear(512, net_params['latent'])
        self.model.fc.requires_grad_(True)

        self.model.apply(weight_init)

        print(f'using model: {self.__class__.__name__}')
        # count_params(self.model)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        images = images / 255.
        if random_rad:
            images = random_augment(images, self.rad_h, self.rad_w)
        else:
            n, c, h, w = images.shape
            images = images[:, :,
              self.rad_h : h-self.rad_h,
              self.rad_w : w-self.rad_w,
            ]

        # print('images: ', images.shape)
        hid = self.model(images)

        if detach:
            hid = hid.detach()

        hid = torch.cat([hid, proprioceptions], dim=-1)

        return hid

class EncoderClassifierModelResNet18(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, shared_layers_max=-1):
        super().__init__()
        self.shared_layers_max = shared_layers_max
        use_imgnet = False
        use_emnist = True
        c, h, w = image_shape
        self.rad_h = round(rad_offset * h)
        self.rad_w = round(rad_offset * w)
        self.model = models.resnet18(num_classes=1000, pretrained=use_imgnet)
        self.latent_dim = net_params['latent'] + proprioception_shape[0]

        # change conv1 to support 9 channels using pre-trained weights
        w_c1 = self.model.conv1.weight
        w_c1_9ch = torch.cat((w_c1 / 3., w_c1 / 3., w_c1 / 3.), dim=1)
        self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if use_imgnet:
            self.model.conv1.requires_grad_(False)
            self.model.conv1.weight.copy_(nn.parameter.Parameter(w_c1_9ch, requires_grad=False))
            self.model.requires_grad_(False)

        if use_emnist:
            self.model.fc = nn.Linear(512, 47)
            ckpt = torch.load(f'classifier/resnet18_e-mnist_aug_stacked_True_imgnet_False_fix_lower_layers_False.pt')
            self.load_state_dict(ckpt['state_dict'])
            self.model.requires_grad_(False)
            # emnist_fc = self.model.fc

        self.model.fc = nn.Linear(512, net_params['latent'])
        self.model.fc.requires_grad_(True)
        # self.ss = SpatialSoftmax(width, height, conv_params[-1][1])

        if use_imgnet or use_emnist:
            self.model.layer4.requires_grad_(True)
            self.model.layer4.apply(weight_init)
            self.model.fc.apply(weight_init)
        else:
            self.model.apply(weight_init)

        if shared_layers_max >= 0 and use_emnist:
            model_layers = list(self.model.named_children())
            backbone_layers = model_layers[:shared_layers_max + 1]
            encoder_layers = model_layers[shared_layers_max + 1:-1]
            encoder_fc_layer = model_layers[-1]
            # self.backbone = nn.Sequential(*backbone_layers)
            # self.encoder_head = nn.Sequential(*encoder_layers)
            self.backbone = nn.Sequential(OrderedDict(backbone_layers))
            self.encoder_head = nn.Sequential(OrderedDict(encoder_layers))
            self.encoder_fc = nn.Sequential(OrderedDict([encoder_fc_layer]))
            self.backbone.requires_grad_(False)
            self.encoder_head.requires_grad_(True)
            self.encoder_fc.requires_grad_(True)
            print(self.model.conv1.weight[0,0,1])
            print(self.backbone.conv1.weight[0,0,1])
            self.encoder_head.apply(weight_init)
            self.encoder_fc.apply(weight_init)
            if not (use_imgnet or use_emnist):
                self.backbone.apply(weight_init)
            del self.model


        if shared_layers_max >= 0:
            print('-------- backbone')
            count_params(self.backbone)
            print('-------- encoder_head')
            count_params(self.encoder_head)
        else:
            print('-------- model')
            count_params(self.model)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        images = images / 255.
        if random_rad:
            images = random_augment(images, self.rad_h, self.rad_w)
        else:
            n, c, h, w = images.shape
            images = images[:, :,
              self.rad_h : h-self.rad_h,
              self.rad_w : w-self.rad_w,
            ]

        if self.shared_layers_max >= 0:
            with torch.no_grad():
                backbone_out = self.backbone(images).detach()
            encoder_head_out = self.encoder_head(backbone_out)
            encoder_head_out = encoder_head_out.reshape((encoder_head_out.shape[0], -1))
            hid = self.encoder_fc(encoder_head_out)
        else:
            hid = self.model(images)
        # hid = self.ss(hid)

        if detach:
            hid = hid.detach()

        hid = torch.cat([hid, proprioceptions], dim=-1)

        if self.shared_layers_max >= 0:
            return hid, backbone_out
        else:
            return hid

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class ActorModelResNet18(ActorModel):
    """MLP actor network."""
    def __init__(
        self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, layernorm, pnorm, shared_layers_max=6):
        super().__init__(image_shape, proprioception_shape, action_dim, net_params, rad_offset)

        # self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        self.encoder = EncoderModelResNet18(image_shape, proprioception_shape, net_params, rad_offset, layernorm)
        # self.encoder = EncoderClassifierModelResNet18(image_shape, proprioception_shape, net_params, rad_offset, shared_layers_max=shared_layers_max)

        mlp_params = net_params['mlp']
        mlp_params[0][0] = self.encoder.latent_dim
        mlp_params[-1][-1] = action_dim * 2
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if pnorm and i == len(mlp_params) - 2:
                layers.append(PNorm())
            if i < len(mlp_params) - 1:
                if layernorm:
                    layers.append(nn.LayerNorm(out_dim, elementwise_affine=True))
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

        if isinstance(self.encoder, EncoderClassifierModelResNet18):
            # load the classification head
            ckpt = torch.load(f'classifier/resnet18_e-mnist_aug_stacked_True_imgnet_False_fix_lower_layers_False.pt')
            self.model = models.resnet18(num_classes=47, pretrained=False)
            self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.load_state_dict(ckpt['state_dict'], strict=False)
            model_layers = list(self.model.named_children())
            classifier_layers = model_layers[shared_layers_max + 1:-1]
            classifier_fc_layer = model_layers[-1]
            self.classifier_head = nn.Sequential(OrderedDict(classifier_layers))
            self.classifier_fc = nn.Sequential(OrderedDict([classifier_fc_layer]))
            # self.classifier_head.eval()
            # self.classifier_fc.eval()
            self.classifier_head.requires_grad_(False)
            self.classifier_fc.requires_grad_(False)
            del self.model

        self.outputs = dict()
        self.trunk.apply(weight_init)
        self.trunk[-1].weight.data.fill_(0.0)
        self.trunk[-1].bias.data.fill_(0.0)
        print('Using normal distribution initialization.')

    def forward(
        self, images, proprioceptions, random_rad=True, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        if isinstance(self.encoder, EncoderClassifierModelResNet18):
            latents, backbone_out = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        else:
            latents = self.encoder(images, proprioceptions, random_rad, detach=detach_encoder)
        mu, log_std = self.trunk(latents).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (
            LOG_STD_MAX - LOG_STD_MIN
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            # log_pi = gaussian_logprob(noise, log_std)
            mynormal = torch.distributions.Normal(mu, std)
            log_pi = mynormal.log_prob(pi).sum(-1, keepdim=True)
        else:
            log_pi = None

        unsquashed_mu, unsquashed_pi = mu.clone(), pi.clone()
        mu, pi, log_pi = squash(mu, pi, log_pi)

        if isinstance(self.encoder, EncoderClassifierModelResNet18):
            # classification forward pass
            with torch.inference_mode():
                # print(self.classifier_head.layer4[0].conv1.weight[0, 0, 0])
                # print(self.classifier_fc.fc.weight[:,0])
                classifier_head_out = self.classifier_head(backbone_out.detach())
                classifier_head_out = classifier_head_out.reshape((classifier_head_out.shape[0], -1))
                classifier_logits = self.classifier_fc(classifier_head_out)
            return mu, pi, log_pi, log_std, unsquashed_mu, unsquashed_pi, classifier_logits

        return mu, pi, log_pi, log_std, unsquashed_mu, unsquashed_pi

class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, latent_dim, action_dim, net_params, layernorm, pnorm):
        super().__init__()

        mlp_params = net_params['mlp']
        mlp_params[0][0] = latent_dim + action_dim
        mlp_params[-1][-1] = 1
        layers = []
        for i, (in_dim, out_dim) in enumerate(mlp_params):
            layers.append(nn.Linear(in_dim, out_dim))
            if pnorm and i == len(mlp_params) - 2:
                layers.append(PNorm())
            if i < len(mlp_params) - 1:
                if layernorm:
                    layers.append(nn.LayerNorm(out_dim, elementwise_affine=True))
                layers.append(nn.ReLU())
        self.trunk = nn.Sequential(
            *layers
        )

    def forward(self, latents, actions):
        latent_actions = torch.cat([latents, actions], dim=-1)
        
        return self.trunk(latent_actions)

class CriticModelResNet18(CriticModel):
    """Critic network, employes two q-functions."""
    def __init__(
        self, image_shape, proprioception_shape, action_dim, net_params, rad_offset, layernorm, pnorm, shared_layers_max=6):
        super().__init__(image_shape, proprioception_shape, action_dim, net_params, rad_offset)

        # self.encoder = EncoderModel(image_shape, proprioception_shape, net_params, rad_offset)
        self.encoder = EncoderModelResNet18(image_shape, proprioception_shape, net_params, rad_offset, layernorm)
        # self.encoder = EncoderClassifierModelResNet18(image_shape, proprioception_shape, net_params, rad_offset, shared_layers_max=shared_layers_max)

        self.Q1 = QFunction(
            self.encoder.latent_dim, action_dim, net_params, layernorm, pnorm
        )
        self.Q2 = QFunction(
            self.encoder.latent_dim, action_dim, net_params, layernorm, pnorm
        )

        self.outputs = dict()
        self.Q1.apply(weight_init)
        self.Q2.apply(weight_init)

    def forward(self, images, proprioceptions, actions, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if isinstance(self.encoder, EncoderClassifierModelResNet18):
            latents, backbone_out = self.encoder(images, proprioceptions, detach=detach_encoder)
        else:
            latents = self.encoder(images, proprioceptions, detach=detach_encoder)
        q1s = self.Q1(latents, actions)
        q2s = self.Q2(latents, actions)

        self.outputs['q1'] = q1s
        self.outputs['q2'] = q2s

        return q1s, q2s

