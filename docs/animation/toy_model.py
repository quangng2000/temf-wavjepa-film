"""Small numerical illustration, not the proposed restoration architecture."""
import torch
from torch import nn


def pnorm(value):
    return value / (value.square().mean(-1, keepdim=True) + 1e-8).sqrt()


def frame_group_norm(value):
    b, c, f, t = value.shape
    groups = value.reshape(b, 2, c // 2, f, t)
    centered = groups - groups.mean((2, 3), keepdim=True)
    return (centered / (centered.square().mean((2, 3), keepdim=True) + 1e-5).sqrt()).reshape_as(value)


class ToyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(128, 8), nn.SiLU(), nn.Linear(8, 8))
        self.film = nn.Linear(4, 8)
        self.head = nn.Conv2d(4, 2, 1)
        self.skip = nn.Parameter(torch.tensor([0.9, -0.65]))
        self.register_buffer('frequencies', 10000 ** (-torch.arange(32) / 31))
        with torch.no_grad():
            for p in self.time_mlp.parameters():
                p.mul_(0.25)
            self.time_mlp[-1].bias[:4].add_(1)
            self.film.weight.mul_(0.25)
            self.film.bias.copy_(torch.tensor([.35, -.35, .1, -.1, .1, -.1, .04, -.02]))

    def inspect(self, state, observed, t, r, context):
        phases = torch.stack([t, r])[:, None] * self.frequencies
        embedding = torch.cat([phases.sin(), phases.cos()], -1).flatten()[None]
        time_raw = self.time_mlp(embedding)
        scale, bias = time_raw.chunk(2, -1)
        scale, bias = pnorm(scale), pnorm(bias)
        features = self.conv(torch.cat([state, observed], 1))
        normalized = frame_group_norm(features)
        timed = normalized * scale[:, :, None, None] + bias[:, :, None, None]
        gamma, beta = self.film(context.mean(-1)).chunk(2, -1)
        gamma = 1 + gamma
        modulated = timed * gamma[:, :, None, None] + beta[:, :, None, None]
        velocity = .08 * self.head(torch.tanh(modulated)) + self.skip[0] * state + self.skip[1] * observed
        return dict(embedding=embedding, time_raw=time_raw, scale=scale, bias=bias,
                    features=features, normalized=normalized, timed=timed,
                    gamma=gamma, beta=beta, modulated=modulated, velocity=velocity)

    def forward(self, state, observed, t, r, context):
        return self.inspect(state, observed, t, r, context)['velocity']


@torch.no_grad()
def sample(model, observed, context, prior):
    """No clean target is accepted or read by this sampler."""
    states, steps = [prior.clone()], []
    for current, destination in zip([1., .75, .5, .25], [.75, .5, .25, 0.]):
        t, r = prior.new_tensor(current), prior.new_tensor(destination)
        velocity = model(states[-1], observed, t, r, context)
        delta = -(t - r) * velocity
        next_state = states[-1] + delta
        steps.append(dict(t=t, r=r, velocity=velocity, delta=delta, next=next_state))
        states.append(next_state)
    return states, steps
