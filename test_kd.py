"""Real PyTorch GPU checks; does not claim an RNNT training test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'icefall/egs/librispeech/ASR/zipformer'))
import torch
from kd_utils import match_time, encoder_kd, logit_kd

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t = torch.tensor([[[1.], [3.], [99.], [99.]], [[2.], [4.], [6.], [8.]]], device=device)
lens = torch.tensor([3, 5], device=device)
a = match_time(t, torch.tensor([2,4], device=device), lens, 5)
assert torch.allclose(a[0,:,0], torch.tensor([1.,2.,3.,0.,0.], device=device))
s = torch.randn(2,5,1,device=device,requires_grad=True)
for kind in ['cosine','mse','l1']:
    value = encoder_kd(s,a,lens,kind)
    altered = a.clone(); altered[0,3:] = 1000
    assert torch.allclose(value, encoder_kd(s,altered,lens,kind))
l = torch.randn(2,5,3,7,device=device,requires_grad=True)
assert abs(logit_kd(l,l.detach(),lens,2.).item()) < 1e-5
teacher = torch.randn_like(l)
assert torch.allclose(logit_kd(l,teacher,lens,2.), logit_kd(l.repeat(1,1,2,1),teacher.repeat(1,1,2,1),lens,2.),atol=1e-5)
opt = torch.optim.Adam([s,l],lr=.03)
losses=[]
for step in range(20):
    opt.zero_grad()
    loss=encoder_kd(s,a,lens,'mse')+logit_kd(l,teacher,lens,2.)
    loss.backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(l.grad).all()
    assert (s.grad[0,3:]==0).all() and (l.grad[0,3:]==0).all()
    opt.step(); losses.append(loss.item())
assert losses[-1]<losses[0]
print(f'{device}: padding, alignment, KL identity, range normalization, gradients passed; 20 synthetic KD steps: {losses[0]:.4f} -> {losses[-1]:.4f}')
