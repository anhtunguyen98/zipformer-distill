"""Padding-safe KD reductions; sums over frames, means over pruned states."""
import torch
import torch.nn.functional as F


def match_time(teacher, teacher_lengths, student_lengths, padded_length):
    aligned = teacher.new_zeros((teacher.size(0), padded_length, teacher.size(-1)))
    for i, (nt, ns) in enumerate(zip(teacher_lengths.tolist(), student_lengths.tolist())):
        if not (0 < nt <= teacher.size(1) and 0 < ns <= padded_length):
            raise ValueError("Invalid encoder lengths")
        valid = teacher[i, :nt]
        if nt != ns:
            valid = F.interpolate(valid.T[None].float(), size=ns, mode="linear",
                                  align_corners=False)[0].T.to(teacher.dtype)
        aligned[i, :ns] = valid
    return aligned


def mask_for(x, lengths):
    return torch.arange(x.size(1), device=x.device)[None] < lengths[:, None]


def encoder_kd(student, teacher, lengths, kind="cosine"):
    s, t = student.float(), teacher.float()
    if kind == "cosine":
        loss = 1 - F.cosine_similarity(s, t, dim=-1)
    elif kind == "mse":
        loss = (s-t).square().mean(-1)
    elif kind == "l1":
        loss = (s-t).abs().mean(-1)
    else:
        raise ValueError(kind)
    return loss.masked_fill(~mask_for(student, lengths), 0).sum()


def logit_kd(student, teacher, lengths, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = F.log_softmax(student.float() / temperature, dim=-1)
    t = F.log_softmax(teacher.float() / temperature, dim=-1)
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1).mean(-1)
    return loss.masked_fill(~mask_for(student, lengths), 0).sum() * temperature**2
