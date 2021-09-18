from diffusion import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 2, 4, 4, 8, 8, 16)
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 2000,   # number of steps
    loss_type = 'l1+l2'    # L1 or L2
).cuda()

lr = 3e-5  #1e-5

trainer = Trainer(
    diffusion,
    '../../../datasets/CelebA_HQM',
    image_size = 256,   #256,
    train_batch_size = 16,
    train_lr = lr,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.998,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = 'results',
)

trainer.load(140000) # <step> = # in the name

trainer.train()


