"""Exercise real icefall epoch/validation/optimizer/checkpoints on synthetic batches."""
import copy
import logging
import sys
from pathlib import Path
import torch
import sentencepiece as spm
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'icefall/egs/librispeech/ASR/zipformer'))
import train_distill as d
from icefall.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(1)

def run(mode="encoder", fp16=False):
    torch.manual_seed(10)
    parser=d.base.get_parser()
    d.LibriSpeechAsrDataModule.add_arguments(parser)
    d.add_distillation_arguments(parser)
    p=d.base.get_params(); p.update(vars(parser.parse_args([])))
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/'assets/tokenizer/bpe.model'))
    arch=dict(num_encoder_layers='1,1',downsampling_factor='1,2',encoder_dim='32,48',
        encoder_unmasked_dim='32,32',feedforward_dim='64,96',num_heads='2',
        query_head_dim='8',value_head_dim='4',pos_head_dim='4',cnn_module_kernel='7,7',
        decoder_dim=32,joiner_dim=32,causal=True,chunk_size='16',left_context_frames='32')
    p.update(arch)
    for k in list(p):
        if k.startswith('teacher_') and k!='teacher_checkpoint':
            p[k]=p[k[len('teacher_'):]]
    p.teacher_encoder_dim='48,64'
    p.teacher_causal=False
    p.teacher_chunk_size='-1'
    p.update(dict(vocab_size=len(sp),blank_id=0,kd_type=mode,use_autocast=fp16,
        dtype=torch.float16 if fp16 else torch.float32,cur_epoch=1,
        average_period=2,valid_interval=2,log_interval=1,save_every_n=100,
        batch_idx_train=0,prune_range=3,kd_warmup_steps=2))
    p.exp_dir=ROOT/'loss_decrease_exp'
    p.exp_dir.mkdir(parents=True,exist_ok=True)
    tp=copy.deepcopy(p);tp.encoder_dim=p.teacher_encoder_dim;tp.causal=False;tp.chunk_size='-1'
    teacher=d._ORIG_GET_MODEL(tp)
    p.teacher_checkpoint=str(p.exp_dir/'teacher.pt')
    torch.save({'model':teacher.state_dict()},p.teacher_checkpoint)
    del teacher
    # Patch as run() does, ensuring lazy teacher construction uses original factory.
    d.base.get_model=d.get_model_with_distill
    d.base.compute_loss=d.compute_loss_distill
    model=d.get_model_with_distill(p).cuda()
    avg=copy.deepcopy(model).to(dtype=torch.float64)
    optimizer=d.base.ScaledAdam(d.base.get_parameter_groups_with_lrs(model,lr=.01,include_names=True),lr=.01,clipping_scale=2.)
    scheduler=d.base.Eden(optimizer,p.lr_batches,p.lr_epochs,warmup_start=1.)
    scaler=d.base.create_grad_scaler(enabled=fp16,init_scale=1.)
    batch={'inputs':torch.randn(2,100,80),'supervisions':{
        'text':['XIN CHÀO','VIỆT NAM'],'num_frames':torch.tensor([100,80])}}
    import json
    # Keep objective weights fixed for meaningful before/after comparisons.
    p.warm_step=0
    p.kd_warmup_steps=0
    p.batch_idx_train=1
    p.valid_interval=10000
    p.save_every_n=10000
    p.log_interval=10
    history=[]
    def measure(step):
        model.eval()
        loss,info=d.compute_loss_distill(p,model,sp,batch,False)
        row={"step":step, **{key:float(info[key]) for key in
             ["loss","simple_loss","pruned_loss","kd_enc_loss"]}}
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        history.append(row)
        (p.exp_dir/'loss_history.json').write_text(json.dumps(history,indent=2))
        print("MEASURE " + json.dumps(row),flush=True)
    measure(0)
    teacher_before={k:v.detach().cpu().clone() for k,v in model._teacher.state_dict().items()}
    for step in range(10,101,10):
        d.base.train_one_epoch(params=p,model=model,optimizer=optimizer,scheduler=scheduler,
            sp=sp,train_dl=[batch]*10,valid_dl=[batch],scaler=scaler,model_avg=avg)
        measure(step)
    assert all(torch.equal(v.cpu(),teacher_before[k]) for k,v in model._teacher.state_dict().items())
    assert all(t.grad is None for t in model._teacher.parameters())
    for key in ["loss","simple_loss","pruned_loss","kd_enc_loss"]:
        assert history[-1][key] < history[0][key], (key,history[0],history[-1])
    save_checkpoint(p.exp_dir/'checkpoint.pt',model,model_avg=avg,optimizer=optimizer,
                    scheduler=scheduler,scaler=scaler,params=p)
    print("PASS: all four measured losses decreased; teacher unchanged",flush=True)

if __name__=="__main__":
    run()
