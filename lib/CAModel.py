import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, scale_growth=1):
        super(CAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.scale_growth = scale_growth

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def normalize_grads(self):
        ''' 
        from(github.com/Mayukhdeb/differentiable-morphogenesis)
        gradient normalization for constant step size and to avoid spikes 
        '''
        for p in self.parameters():
            p.grad.data = p.grad.data/(p.grad.data.norm()+1e-8) 

    def alive(self, x):
        return F.max_pool2d(x[:, 1:2, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, scale_growth_synthesis, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        dx = dx * self.scale_growth * scale_growth_synthesis
        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, scale_growth_synthesis=1, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, scale_growth_synthesis, angle)
        return x
