"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '/Users/huanghongyan/Documents/DeepLearning/mmgeneration/configs/_base_/'
    'datasets/unconditional_imgs_flip_256x256.py',
    '/Users/huanghongyan/Documents/DeepLearning/mmgeneration/configs/_base_/models/stylegan/stylegan2_base.py',
    '/Users/huanghongyan/Documents/DeepLearning/mmgeneration/configs/_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth'

# ada settings
aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

model = dict(generator=dict(out_size=256),
             discriminator=dict(
                 in_size=256,
                 type='ADAStyleGAN2Discriminator',
                 data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs, ada_kimg=100)
             )
             )

data = dict(
    samples_per_gpu=8,
    train=dict(imgs_root='/MMGeneration/mmgeneration/data/milk/Milk'),
    val=dict(imgs_root='/MMGeneration/mmgeneration/data/milk/Milk')
)
ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=500),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema',),
        interval=1,
        interp_cfg=dict(momentum=0.5 ** (32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

checkpoint_config = dict(interval=500, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

total_iters = 10000

num_sample = 500
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=num_sample,
        inception_pkl='work_dirs/inception_pkl/strawberry.pkl',
        bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=num_sample, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=num_sample))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=1000,
    metrics=dict(
        type='FID',
        num_images=num_sample,
        inception_pkl='work_dirs/inception_pkl/strawberry.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
