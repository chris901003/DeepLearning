import torch
from torchvision import transforms
from torch import nn
from PIL import Image
from functools import partial

transform_data = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

image_path = '../CustomizePicture/123.png'
image = Image.open(image_path)
image = transform_data(image)
image = image.unsqueeze(dim=0)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patch = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[1], 'Photo is not available.'
        # print(x.shape)
        x = self.proj(x)
        # print(x.shape)
        # transpose是為了norm layer做的，norm layer是對[-1]做均值會做[-2]次
        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias)
        # self.attn_bn = nn.BatchNorm2d(12)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.prof_drop = nn.Dropout(p=proj_drop)
        # self.bn_prof = nn.BatchNorm1d(197)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        # print(qkv.shape)
        qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = self.attn_bn(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(attn.shape)
        # print(v.shape)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        # x = self.bn_prof(x)
        x = self.prof_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p=drop)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.ln2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_ratio,
                              proj_drop=drop_ratio)
        self.drop1 = nn.Dropout(p=drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop_ratio)
        self.drop2 = nn.Dropout(p=drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x = x+self.drop1(self.attn(self.norm1(x)))
        # x = x+self.drop2(self.mlp(self.norm2(x)))
        x = x + self.ln1(self.attn(self.norm1(x)))
        x = x + self.ln2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_class=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_class = num_class
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                       norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.ln = nn.LayerNorm(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(in_features=embed_dim, out_features=num_class) if num_class > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_feature(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        x = self.ln(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000, img_size=224, mlp_ratio=0.2, drop_path_ratio=0.2,
                         attn_drop_ratio=0.2, patch_size=16):
    model = VisionTransformer(img_size=img_size,
                              patch_size=patch_size,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              num_class=num_classes,
                              mlp_ratio=mlp_ratio,
                              drop_path_ratio=drop_path_ratio,
                              attn_drop_ratio=attn_drop_ratio)
    return model


net = vit_base_patch16_224()
v1 = net(image)
