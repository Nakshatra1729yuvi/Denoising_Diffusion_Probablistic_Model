import torch
import torch.nn as nn

class LinearNoiseScheduler:

  def __init__(self,beta_start,beta_end,num_timesteps):
    self.beta_start=beta_start
    self.beta_end=beta_end
    self.num_timesteps=num_timesteps

    self.betas=torch.linspace(beta_start,beta_end,num_timesteps)

    self.alphas=1-self.betas
    self.sqrt_alphas=torch.sqrt(self.alphas)
    self.alphas_cum_prod=torch.cumprod(self.alphas ,dim=0)
    self.sqrt_alphas_cum_prod=torch.sqrt(self.alphas_cum_prod)
    self.sqrt_one_minus_alphas_cum_prod=torch.sqrt(1-self.alphas_cum_prod)

  def add_noise(self,original,noise,t):

    sqrt_alphas_cum_prod=self.sqrt_alphas_cum_prod.to(original.device)[t].view(-1,1,1,1)
    sqrt_one_minus_alphas_cum_prod=self.sqrt_one_minus_alphas_cum_prod.to(original.device)[t].view(-1,1,1,1)

    return sqrt_alphas_cum_prod*original+sqrt_one_minus_alphas_cum_prod*noise

  def sample_prev_timestep(self,xt,noise_pred,t):

    x0=xt-(self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t])*noise_pred
    x0=x0/self.sqrt_alphas_cum_prod.to(xt.device)[t]
    x0=torch.clamp(x0,-1,1)


    mean=xt-(self.betas.to(xt.device)[t]/self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t])*noise_pred
    mean=mean/self.sqrt_alphas.to(xt.device)[t]

    if t==0:
      return mean, x0
    else:
      variance=(self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t-1])/(self.sqrt_one_minus_alphas_cum_prod.to(xt.device)[t])
      sigma=(self.betas.to(xt.device)[t]**0.5)*variance
      z=torch.randn(xt.shape).to(xt.device)

      return mean+sigma*z,x0


def get_time_embedding(time_steps,t_emb_dim):

  assert t_emb_dim%2==0

  factor=1000**(torch.arange(0,t_emb_dim//2,dtype=torch.float32,device=time_steps.device)/(t_emb_dim//2))

  t_emb=time_steps.view(-1,1)/factor
  t_emb=torch.concat([torch.sin(t_emb),torch.cos(t_emb)],dim=-1)

  return t_emb

class DownSample(nn.Module):
  def __init__(self,in_channels,hidden_channels,emb_dim,downsample,num_layers=1,num_heads=4):
    super().__init__()

    self.downsample=downsample
    self.num_layers=num_layers
    self.resnet_first=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,in_channels if i==0 else hidden_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers)
    )

    self.t_emb_layers=nn.ModuleList(
        nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,hidden_channels)
        ) for i in range(num_layers)
    )

    self.resnet_second=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),
        nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers)
    )

    self.attn_norms=nn.ModuleList([nn.GroupNorm(8,hidden_channels) for _ in range(num_layers)])

    self.attns=nn.ModuleList([nn.MultiheadAttention(hidden_channels,num_heads,batch_first=True) for _ in range(num_layers)])

    self.resnet_input=nn.ModuleList([nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1) for i in range(num_layers)])

    self.downsample_conv=nn.Conv2d(hidden_channels,hidden_channels,kernel_size=4,stride=2,padding=1)

  def forward(self,x,t_emb):
    out=x
    for i in range(self.num_layers):

      resnet_input=out
      out=self.resnet_first[i](resnet_input)
      out=out+self.t_emb_layers[i](t_emb)[:,:,None,None]
      out=self.resnet_second[i](out)
      out=out+self.resnet_input[i](resnet_input)

      B,C,H,W=out.shape
      attn=out.reshape(B,C,H*W)
      attn=self.attn_norms[i](attn)
      attn=attn.transpose(1,2)
      attn,_=self.attns[i](attn,attn,attn)
      attn=attn.transpose(1,2).reshape(B,C,H,W)
      out=out+attn



    return self.downsample_conv(out) if self.downsample else out




class MidBlock(nn.Module):
  def __init__(self,in_channels,hidden_channels,emb_dim,num_layers=1,num_heads=4):
    super().__init__()


    self.num_layers=num_layers
    self.resnet_first=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,in_channels if i==0 else hidden_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers+1)
    )

    self.t_emb_layers=nn.ModuleList(
        nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,hidden_channels)
        ) for i in range(num_layers+1)
    )

    self.resnet_second=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),
        nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers+1)
    )

    self.attn_norms=nn.ModuleList([nn.GroupNorm(8,hidden_channels) for _ in range(num_layers)])

    self.attns=nn.ModuleList([nn.MultiheadAttention(hidden_channels,num_heads,batch_first=True) for _ in range(num_layers)])

    self.resnet_input=nn.ModuleList([nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1) for i in range(num_layers+1)])


  def forward(self,x,t_emb):
    out=x

    resnet_input=out

    out=self.resnet_first[0](resnet_input)
    out=out+self.t_emb_layers[0](t_emb)[:,:,None,None]
    out=self.resnet_second[0](out)
    out=out+self.resnet_input[0](resnet_input)

    for i in range(self.num_layers):

      B,C,H,W=out.shape
      attn=out.reshape(B,C,H*W)
      attn=self.attn_norms[i](attn)
      attn=attn.transpose(1,2)
      attn,_=self.attns[i](attn,attn,attn)
      attn=attn.transpose(1,2).reshape(B,C,H,W)
      out=out+attn

      resnet_input=out
      out=self.resnet_first[i+1](resnet_input)
      out=out+self.t_emb_layers[i+1](t_emb)[:,:,None,None]
      out=self.resnet_second[i+1](out)
      out=out+self.resnet_input[i+1](resnet_input)


    return out




class UpSample(nn.Module):
  def __init__(self,in_channels,hidden_channels,emb_dim,upsample,num_layers=1,num_heads=4):
    super().__init__()

    self.upsample=upsample
    self.upsample_conv=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=4,stride=2,padding=1)

    self.num_layers=num_layers
    self.resnet_first=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,in_channels if i==0 else hidden_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers)
    )

    self.t_emb_layers=nn.ModuleList(
        nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,hidden_channels)
        ) for i in range(num_layers)
    )

    self.resnet_second=nn.ModuleList(nn.Sequential(
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),
        nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1)
    ) for i in range(num_layers)
    )

    self.attn_norms=nn.ModuleList([nn.GroupNorm(8,hidden_channels) for _ in range(num_layers)])

    self.attns=nn.ModuleList([nn.MultiheadAttention(hidden_channels,num_heads,batch_first=True) for _ in range(num_layers)])

    self.resnet_input=nn.ModuleList([nn.Conv2d(in_channels if i==0 else hidden_channels,hidden_channels,kernel_size=3,stride=1,padding=1) for i in range(num_layers)])



  def forward(self,x,out_down,t_emb):
    x=self.upsample_conv(x) if self.upsample else x
    x=torch.cat([x,out_down],dim=1)
    out=x
    for i in range(self.num_layers):

      resnet_input=out
      out=self.resnet_first[i](resnet_input)
      out=out+self.t_emb_layers[i](t_emb)[:,:,None,None]
      out=self.resnet_second[i](out)
      out=out+self.resnet_input[i](resnet_input)

      B,C,H,W=out.shape
      attn=out.reshape(B,C,H*W)
      attn=self.attn_norms[i](attn)
      attn=attn.transpose(1,2)
      attn,_=self.attns[i](attn,attn,attn)
      attn=attn.transpose(1,2).reshape(B,C,H,W)
      out=out+attn



    return out




class Unet(nn.Module):
  def __init__(self,in_channels):
    super().__init__()
    self.down_channels=[32,64,128,256]
    self.mid_channels=[256,256,128]
    self.up_channels=[256,128,64,32]
    self.down_bool=[True,True,False]
    self.up_bool=[False,True,True]
    self.t_emb_dim=128

    self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.down_channels[0],kernel_size=3,stride=1,padding=1)

    self.downs=nn.ModuleList([])
    self.mids=nn.ModuleList([])
    self.ups=nn.ModuleList([])

    for i in range(len(self.down_channels)-1):
      self.downs.append(DownSample(self.down_channels[i],self.down_channels[i+1],self.t_emb_dim,self.down_bool[i]))

    for i in range(len(self.mid_channels)-1):
      self.mids.append(MidBlock(self.mid_channels[i],self.mid_channels[i+1],self.t_emb_dim))

    for i in range(len(self.up_channels)-1):
      self.ups.append(UpSample(self.up_channels[i],self.up_channels[i+1],self.t_emb_dim,self.up_bool[i]))

    self.t_proj=nn.Sequential(
        nn.Linear(self.t_emb_dim,self.t_emb_dim),
        nn.SiLU(),
        nn.Linear(self.t_emb_dim,self.t_emb_dim)
    )
    self.norm_out=nn.GroupNorm(8,self.up_channels[-1])
    self.conv_out=nn.Conv2d(self.up_channels[-1],in_channels,kernel_size=3,stride=1,padding=1)

  def forward(self,x,t):

    out=self.in_conv(x)
    t_emb=get_time_embedding(t,self.t_emb_dim)
    t_emb=self.t_proj(t_emb)

    down_outs=[]

    for i in range(len(self.downs)):
      down_outs.append(out)
      out=self.downs[i](out,t_emb)

    for i in range(len(self.mids)):
      out=self.mids[i](out,t_emb)

    for i in range(len(self.ups)):
      skip=down_outs.pop()
      out=self.ups[i](out,skip,t_emb)

    out=self.norm_out(out)
    return self.conv_out(out)




from tqdm import tqdm

device=torch.device('cpu')


time_steps=500
time_scheduler=LinearNoiseScheduler(beta_start=1e-4,beta_end=0.02,num_timesteps=time_steps)

model=Unet(1).to(device)

model.load_state_dict(torch.load('D:\Diffusion(DDPM)\ddpm.pth',map_location=device))


# inference


# inference

img_stack=[]
x=torch.randn(1,1,28,28).to(device)
for t in tqdm(range(time_steps)):
  t_tensor = torch.full((x.shape[0],), 499 - t, device=x.device)
  noise_pred=model(x,t_tensor)
  x,_=time_scheduler.sample_prev_timestep(x,noise_pred,t_tensor)
  img_stack.append(x.detach().cpu())

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

frames = []

for img_tensor in img_stack:
    img = img_tensor[0,0]              # [28,28]
    img = (img + 1) / 2                # [-1,1] → [0,1]
    img = (img * 255).clamp(0,255)     # [0,255]
    img = img.numpy().astype(np.uint8)

    frames.append(Image.fromarray(img, mode="L"))  # grayscale
frames[0].save('ddpm_mnist.gif', save_all=True, append_images=frames[1:], loop=0, duration=30)

# Show images every 100 steps
for i in range(len(img_stack)):  # 99, 199, 299 ... for 100th, 200th, etc.
    img = (img_stack[-1][0, 0] + 1) / 2  # normalize to [0,1]
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(f"Step {i+1}")
    plt.axis('off')
    plt.show()
    break