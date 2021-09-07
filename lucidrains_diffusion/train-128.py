from diffusion import Unet, GaussianDiffusion, Trainer


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 2, 4, 4, 8, 8,) # Consider making this less deep/wide! it seems to be taking a long time to run...
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of steps
    loss_type = 'l1+l2'    # L1 or L2
).cuda()

lr = 1e-5

trainer = Trainer(
    diffusion,
    '../../cats/',
    image_size = 128,
    train_batch_size = 32,
    train_lr = lr,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.999,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = './results_128_deep'
)

trainer.load('model-105000.pt')
# trainer.load(20000) # <step> = # in the name

trainer.train()


