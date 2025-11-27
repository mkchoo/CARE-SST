#############################################################################################
#                        Code taken from Pablos implementation:                             #
#    https://github.com/pabloppp/pytorch-tools/blob/master/torchtools/utils/diffusion.py    #
#############################################################################################
import torch
from tqdm import tqdm

# Exponential Moving Average (EMA) class
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    # Update model parameters with EMA
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    # Calculate EMA for parameters
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    # Perform EMA step
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    # Reset EMA model parameters to match the current model
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

# Base class for sampling methods
class SimpleSampler:
    def __init__(self, diffuzz):
        self.current_step = -1
        self.diffuzz = diffuzz

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    # Initialize sample with random noise
    def init_x(self, shape):
        return torch.randn(*shape, device=self.diffuzz.device)

    # Abstract step function to be implemented by subclasses
    def step(self, x, t, t_prev, noise):
        raise NotImplementedError("You should override the 'apply' function.")

# DDPM Sampler class
class DDPMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
        std = ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * torch.randn_like(mu)
        return mu + std * (t_prev != 0).float().view(t_prev.size(0), *[1 for _ in x.shape[1:]])

# DDIM Sampler class
class DDIMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise

# Dictionary of available samplers
sampler_dict = {
    'ddpm': DDPMSampler,
    'ddim': DDIMSampler,
}

# Custom simplified forward/backward diffusion (cosine schedule)
class Diffuzz:
    def __init__(self, s=0.008, device="cpu", cache_steps=None, scaler=1):
        self.device = device
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2
        self.scaler = scaler
        self.cached_steps = None
        if cache_steps is not None:
            self.cached_steps = self._alpha_cumprod(torch.linspace(0, 1, cache_steps, device=device))

    # Compute alpha cumulative product for given timesteps
    def _alpha_cumprod(self, t):
        if self.cached_steps is None:
            if self.scaler > 1:
                t = 1 - (1 - t) ** self.scaler
            elif self.scaler < 1:
                t = t ** self.scaler
            alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod
            return alpha_cumprod.clamp(0.0001, 0.9999)
        else:
            return self.cached_steps[t.mul(len(self.cached_steps) - 1).long()]

    # Forward diffusion process
    def diffuse(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1 - alpha_cumprod).sqrt() * noise, noise

    # Reverse diffusion process using a specified sampler
    def undiffuse(self, x, t, t_prev, noise, sampler='ddpm'):
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, noise)

    # Sampling from the diffusion model
    def sample(self, model, background, shape, mask=None, t_start=1.0, t_end=0.0, timesteps=250, x_init=None, sampler='ddpm'):
        r_range = torch.linspace(t_start, t_end, timesteps + 1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(
                    f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler = sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
        preds = []
        x = sampler.init_x(shape)

        # Iterate over timesteps and apply diffusion model
        for i in tqdm(range(0, timesteps), desc="Sampling Progress"):
            if mask is not None:
                x_renoised, _ = self.diffuse(x_init, r_range[i])
                x = x * mask + x_renoised * (1 - mask)

            pred_noise = model(torch.cat((x, background), dim=1), r_range[i])
            x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler)
            preds.append(x)
        return preds


    def sample_denoise(self, model, background, shape, mask=None, t_start=1.0, t_end=0.0, timesteps=250, x_init=None, sampler='ddpm', tau = 0.9):
        r_range = torch.linspace(t_start, t_end, timesteps + 1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(
                    f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler = sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
        preds = []
        
        x,_ = self.diffuse(background, r_range[0])
        for i in range(0, timesteps): 
            if i <= timesteps*tau:   
                if mask is not None:
                    x_renoised, _ = self.diffuse(x_init, r_range[i]) 
                    x = x * mask + x_renoised * (1 - mask)          
                    pred_noise = model(torch.cat((x,background), dim=1), r_range[i]) 

                x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler) 
                preds.append(x)
            else:                     
                if mask is not None:
                    pred_noise = model(torch.cat((x,background), dim=1), r_range[i])            
                    x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler) 
                preds.append(x)                    
                
        return preds

    def sample_blending_denoise_background_one_day(self, model, background, context_past=None,  shape=None, mask=None, past_mask=None,  t_start=1.0, t_end=0.0, timesteps=250, x_init=None, sampler='ddpm', tau = 0.9):
        r_range = torch.linspace(t_start, t_end, timesteps + 1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(
                    f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler = sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
        preds = []

        x,_ = self.diffuse(background, r_range[0])
        
        for i in tqdm(range(0, timesteps), desc="Sampling Progress"):
            if i <= timesteps*tau:
                     
                if i <= timesteps*0.7:
                    x_renoised, _ = self.diffuse(x_init, r_range[i])
                    c_renoised_past, _ = self.diffuse(context_past, r_range[i])
                    x = x * mask * (1 - past_mask) + x_renoised * (1 - mask) + c_renoised_past * (past_mask)
                    
                    pred_noise = model(torch.cat((x,background), dim=1), r_range[i])
                    x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler)
                    preds.append(x)                     
                    
                    
                else:
                    x_renoised, _ = self.diffuse(x_init, r_range[i])
                    x = x * mask + x_renoised * (1 - mask)
                    
                    pred_noise = model(torch.cat((x,background), dim=1), r_range[i])
                    x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler)
                    preds.append(x)
                         
            
            else:
                pred_noise = model(torch.cat((x,background), dim=1), r_range[i])
                x = self.undiffuse(x, r_range[i], r_range[i + 1], pred_noise, sampler=sampler)
                preds.append(x)
        return preds