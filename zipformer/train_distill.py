#!/usr/bin/env python3
# RNNT loss implementation derived from icefall model.py.
# Copyright 2021-2023 Xiaomi Corp. (Amir Hussein, Fangjun Kuang, Wei Kang,
# Mingshuang Luo, Zengwei Yao, Daniel Povey). Licensed under Apache-2.0.
"""Zipformer RNNT distillation; place beside the upstream train.py.

The RNNT implementation below derives from icefall model.py (Apache-2.0).
Only load checkpoints from trusted sources (torch.load uses pickle).
"""
import argparse
import copy
import logging
from pathlib import Path
from typing import Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from asr_datamodule import LibriSpeechAsrDataModule
from model import AsrModel
import train as base
from icefall.utils import MetricsTracker, add_sos, torch_autocast
from kd_utils import match_time, encoder_kd, logit_kd

_ORIG_GET_MODEL = base.get_model


def add_distillation_arguments(parser):
    g = parser.add_argument_group("distillation")
    g.add_argument("--teacher-checkpoint")
    g.add_argument("--student-init-ckpt")
    g.add_argument("--kd-type", choices=["encoder", "logit", "both", "none"], default="both")
    g.add_argument("--kd-encoder-loss", choices=["cosine", "mse", "l1"], default="cosine")
    for name, default in [("encoder-scale", 1.), ("logit-scale", 1.), ("temperature", 2.)]:
        g.add_argument("--kd-" + name, type=float, default=default)
    g.add_argument("--kd-warmup-steps", type=int, default=0)
    # Mirror upstream architecture flags, with large-teacher defaults.
    overrides = dict(num_encoder_layers="2,2,4,5,4,2",
                     feedforward_dim="512,768,1536,2048,1536,768",
                     encoder_dim="192,256,512,768,512,256")
    names = "num_encoder_layers downsampling_factor feedforward_dim num_heads encoder_dim query_head_dim value_head_dim pos_head_dim pos_dim encoder_unmasked_dim cnn_module_kernel decoder_dim joiner_dim causal chunk_size left_context_frames context_size".split()
    actions = {a.dest: a for a in parser._actions}
    for name in names:
        a = actions[name]
        g.add_argument("--teacher-" + name.replace("_", "-"), type=a.type,
                       default=overrides.get(name, a.default))


def load_weights(model, path, allowed_missing=(), allowed_unexpected=()):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(allowed_missing)]
    unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected)]
    if missing or unexpected:
        raise ValueError(f"Incompatible checkpoint {path}: missing={missing}, unexpected={unexpected}")


def build_teacher(params, device):
    tp = copy.deepcopy(params)
    for name in list(params):
        if name.startswith("teacher_") and name != "teacher_checkpoint":
            tp[name[len("teacher_"):]] = params[name]
    tp.use_transducer, tp.use_ctc = True, False
    tp.use_attention_decoder = False
    teacher = _ORIG_GET_MODEL(tp)
    load_weights(teacher, params.teacher_checkpoint,
                 allowed_unexpected=("ctc_output.", "attention_decoder."))
    teacher.to(device).eval().requires_grad_(False)
    return teacher


class DistillAsrModel(AsrModel):
    _teacher = None

    def forward(self, x, x_lens, y, prune_range=5, am_scale=0., lm_scale=0.):
        cfg = self.kd_cfg
        enc, lengths = self.forward_encoder(x, x_lens)
        splits = y.shape.row_splits(1)
        ylens = splits[1:] - splits[:-1]
        simple, pruned, ranges, logits = self.transducer_with_logits(
            enc, lengths, y, ylens, prune_range, am_scale, lm_scale)
        zero = enc.new_zeros(())
        ctc = self.forward_ctc(enc, lengths, y.values, ylens) if self.use_ctc else zero
        ke, kl = zero, zero
        if cfg.kd_type != "none":
            with torch.no_grad():
                te, tlens = self._teacher.forward_encoder(x, x_lens)
                te = match_time(te, tlens, lengths, enc.size(1))
            if cfg.kd_type in ("encoder", "both"):
                ke = encoder_kd(self.kd_proj(enc), te, lengths, cfg.kd_encoder_loss)
            if cfg.kd_type in ("logit", "both"):
                with torch.no_grad():
                    teacher = self._teacher
                    sos = add_sos(y, sos_id=teacher.decoder.blank_id)
                    dec = teacher.decoder(sos.pad(mode="constant", padding_value=teacher.decoder.blank_id))
                    am, lm = k2.do_rnnt_pruning(teacher.joiner.encoder_proj(te),
                                               teacher.joiner.decoder_proj(dec), ranges)
                    teacher_logits = teacher.joiner(am, lm, project_input=False)
                kl = logit_kd(logits, teacher_logits, lengths, cfg.kd_temperature)
        return simple, pruned, ctc, ke, kl

    def transducer_with_logits(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch_autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch_autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss, ranges, logits


def get_model_with_distill(params):
    model = _ORIG_GET_MODEL(params)
    model.__class__ = DistillAsrModel
    model.kd_cfg = params
    if params.kd_type in ("encoder", "both"):
        model.kd_proj = nn.Linear(max(base._to_int_tuple(params.encoder_dim)),
                                  max(base._to_int_tuple(params.teacher_encoder_dim)))
    if params.student_init_ckpt and params.start_epoch == 1 and params.start_batch == 0:
        load_weights(model, params.student_init_ckpt, allowed_missing=("kd_proj.",))
    return model


def compute_loss_distill(params, model, sp, batch, is_training, spec_augment=None):
    device = next(model.parameters()).device
    x = batch["inputs"].to(device)
    lengths = batch["supervisions"]["num_frames"].to(device)
    y = k2.RaggedTensor(sp.encode(batch["supervisions"]["text"], out_type=int)).to(device)
    inner = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if params.kd_type != "none" and inner._teacher is None:
        # Outside the module tree: exclude frozen teacher from DDP, averaging and checkpoints.
        object.__setattr__(inner, "_teacher", build_teacher(params, device))
    with torch.set_grad_enabled(is_training):
        simple, pruned, ctc, ke, kl = model(x=x, x_lens=lengths, y=y,
            prune_range=params.prune_range, am_scale=params.am_scale, lm_scale=params.lm_scale)
        progress = min(1., params.batch_idx_train / max(1, params.warm_step))
        ramp = min(1., params.batch_idx_train / params.kd_warmup_steps) if params.kd_warmup_steps else 1.
        loss = (1. - progress * (1. - params.simple_loss_scale)) * simple
        loss = loss + (0.1 + 0.9 * progress) * pruned
        if params.use_ctc:
            loss = loss + params.ctc_loss_scale * ctc
        loss = loss + ramp * (params.kd_encoder_scale * ke + params.kd_logit_scale * kl)
    info = MetricsTracker()
    info["frames"] = (lengths // params.subsampling_factor).sum().item()
    info["utterances"] = x.size(0)
    for name, value in dict(loss=loss, simple_loss=simple, pruned_loss=pruned,
                            ctc_loss=ctc, kd_enc_loss=ke, kd_logit_loss=kl).items():
        info[name] = value.detach().item()
    return loss, info


def run(rank, world_size, args):
    # Spawn imports a fresh interpreter: patch inside every worker.
    base.get_model = get_model_with_distill
    base.compute_loss = compute_loss_distill
    base.run(rank, world_size, args)


def main():
    parser = base.get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    add_distillation_arguments(parser)
    args = parser.parse_args()
    if not args.use_transducer or args.use_attention_decoder or args.use_cr_ctc:
        parser.error("Requires transducer training; attention decoder and CR-CTC are unsupported")
    if args.kd_temperature <= 0 or args.kd_warmup_steps < 0 or min(args.kd_encoder_scale, args.kd_logit_scale) < 0:
        parser.error("KD temperature must be positive; weights and warmup must be nonnegative")
    if args.kd_type != "none" and (not args.teacher_checkpoint or not Path(args.teacher_checkpoint).is_file()):
        parser.error("A valid --teacher-checkpoint is required for KD")
    if args.world_size < 1:
        parser.error("world-size must be positive")
    args.exp_dir = Path(args.exp_dir)
    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(0, 1, args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
