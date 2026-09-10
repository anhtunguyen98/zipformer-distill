"""Small real Zipformer/RNNT smoke test with random teacher and synthetic features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'icefall/egs/librispeech/ASR/zipformer'))
import train_distill as d
import torch, k2
p=d.base.get_parser(); d.LibriSpeechAsrDataModule.add_arguments(p); d.add_distillation_arguments(p)
a=p.parse_args([])
params=d.base.get_params(); params.update(vars(a))
params.update(dict(vocab_size=16,blank_id=0,num_encoder_layers='1,1',downsampling_factor='1,2',encoder_dim='32,48',encoder_unmasked_dim='32,32',feedforward_dim='64,96',num_heads='2',query_head_dim='8',value_head_dim='4',pos_head_dim='4',cnn_module_kernel='7,7',decoder_dim=32,joiner_dim=32, teacher_encoder_dim='48,64'))
tp=params.copy(); tp.update(dict(encoder_dim='48,64'))
tp=d.base.AttributeDict(tp)
teacher=d._ORIG_GET_MODEL(tp).cuda().eval().requires_grad_(False)
student=d.get_model_with_distill(params).cuda().train()
object.__setattr__(student,'_teacher',teacher)
assert not any('_teacher' in k for k in student.state_dict())
opt=torch.optim.Adam(student.parameters(),lr=1e-3)
x=torch.randn(2,80,80,device='cuda'); lengths=torch.tensor([80,64],device='cuda'); y=k2.RaggedTensor([[1,2,3],[2,4]]).to('cuda')
for i in range(3):
    opt.zero_grad()
    losses=student(x,lengths,y,prune_range=3)
    loss=sum(losses); assert torch.isfinite(loss)
    loss.backward()
    assert student.kd_proj.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())
    opt.step()
    print(i, [round(v.item(),4) for v in losses],flush=True)
Path('smoke_exp').mkdir(exist_ok=True)
torch.save({'model':student.state_dict(),'optimizer':opt.state_dict(),'synthetic':True},'smoke_exp/checkpoint.pt')
print('PASS: real Zipformer + k2 RNNT + both KD losses, 3 GPU updates; synthetic random teacher only')
