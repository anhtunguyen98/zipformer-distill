# Zipformer RNNT Knowledge Distillation

Huấn luyện student Zipformer với RNNT loss và kiến thức từ một teacher đóng băng. Repo cung cấp phần mở rộng cho recipe `egs/librispeech/ASR/zipformer` của [icefall](https://github.com/k2-fsa/icefall), tái sử dụng optimizer ScaledAdam, scheduler Eden, validation, model averaging và checkpoint.

**Đã chạy trên RTX 3090:** encoder KD FP32/FP16, cả hai KD FP16, logit KD FP32 và không KD. Test học trên một batch giả lập trong 100 bước giảm tổng loss từ **542.01 xuống 23.43**. Đây là kiểm tra khả năng tối ưu, không phải kết quả WER trên tiếng nói thật.

## 1. Cài đặt

Yêu cầu Python 3.12, Git, GPU NVIDIA và bộ PyTorch/k2 tương thích. Cấu hình đã kiểm thử: PyTorch `2.11.0+cu130`, k2 `1.24.4.dev20260423+cuda13.0.torch2.11.0`.

```bash
git clone https://github.com/anhtunguyen98/zipformer-distill.git
cd zipformer-distill
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-deps 'https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20260423+cuda13.0.torch2.11.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
bash scripts/setup_icefall.sh
python -m pip install -r icefall/requirements.txt
python -m pip install lhotse huggingface_hub graphviz
export PYTHONPATH="$PWD/.deps:$PWD/icefall${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch,k2,lhotse; print(torch.__version__,k2.__version__,torch.cuda.is_available())'
bash scripts/train.sh --help
```

Script pin icefall tại `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` rồi copy hai file trong `zipformer/` sang recipe. Sau khi sửa code trong `zipformer/`, chạy lại `setup_icefall.sh` để cập nhật bản chạy.

Nếu dùng Python/PyTorch/CUDA khác, chọn wheel phù hợp theo [hướng dẫn k2](https://k2-fsa.github.io/k2/installation/from_wheels.html). Không dùng một wheel ngẫu nhiên từ `pip install k2`: binary phải tương thích với PyTorch. Ví dụ trên yêu cầu driver hỗ trợ CUDA 13.0; chưa kiểm thử toàn bộ quy trình cài mới trên máy sạch. Các dependency ngoài PyTorch/k2/icefall chưa được khóa phiên bản.

## 2. Dữ liệu đầu vào

Giống training Zipformer thông thường:

- Fbank **80 chiều**, tensor `[batch, time, 80]`, cùng số frame hợp lệ của từng utterance.
- Transcript trong `supervisions.text`, được SentencePiece chuyển thành token ID.
- Teacher và student nhận cùng features, bao gồm augmentation do dataloader áp dụng. Không cần lưu sẵn teacher outputs.

Sử dụng Lhotse CutSet có đường dẫn features hợp lệ, transcript và thông tin số frame. Recipe hiện vẫn dùng tên manifest của LibriSpeech. Với `--full-libri 0`, thư mục `--manifest-dir` cần:

```text
librispeech_cuts_train-clean-100.jsonl.gz
librispeech_cuts_dev-clean.jsonl.gz
librispeech_cuts_dev-other.jsonl.gz
```

Có thể chứa dữ liệu tiếng Việt: tên file là quy ước của datamodule, không phải kiểm tra ngôn ngữ. Hai manifest validation phải là các phần validation tách biệt. Với `--full-libri 1`, train dùng `librispeech_cuts_train-all-shuf.jsonl.gz`. Đổi datamodule nếu muốn tên/split riêng.

Chuẩn bị dữ liệu theo [recipe LibriSpeech](https://github.com/k2-fsa/icefall/tree/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR). Ví dụ cấu hình tắt MUSAN (`--enable-musan 0`); bật lại cần manifest/features MUSAN tương ứng. Code không tự chuyển thư mục WAV + TXT thành manifests. Recipe lọc utterance ngoài khoảng 1–20 giây và các utterance quá nhiều token so với số frame.

Chuẩn hóa transcript phù hợp tokenizer **trước khi train**. NghiASR tokenizer dùng chủ yếu chữ hoa; code không tự `.upper()` transcript.

## 3. Chọn checkpoint và kiến trúc

`--teacher-checkpoint` và `--student-init-ckpt` nhận checkpoint PyTorch chứa `model` state dict hoặc một state dict trực tiếp. Teacher phải có các weights khớp kiến trúc; thiếu/sai keys sẽ báo lỗi. Các head CTC/attention không dùng của teacher được phép là keys thừa. Chỉ load checkpoint tin cậy vì loader dùng pickle.

- Student architecture: `--encoder-dim`, `--num-encoder-layers`, `--feedforward-dim`, `--causal`, v.v.
- Teacher architecture: cùng tên nhưng thêm `--teacher-`, kể cả `--teacher-context-size`.
- Tất cả kích thước phải khớp checkpoint; không tự suy ra architecture từ file.
- Student init là tùy chọn. Bỏ nó để train student từ đầu.
- TorchScript/JIT và ONNX exports không load trực tiếp bằng loader này.

Với encoder KD, teacher không dùng transcript/decoder để tạo KD target. Tuy nhiên factory hiện vẫn dựng cả teacher RNNT với `vocab_size` của student: nếu kích thước vocab khác nhau cần chỉnh loader/factory. Với logit KD, teacher và student phải dùng **cùng token semantics, ID mapping và tokenizer**, không chỉ cùng kích thước vocab.

Các tokenizer NghiASR, `zzasdf/viet_iter3_pseudo_label` và `hynt/Zipformer-30M-RNNT-Streaming-6000h` đã kiểm tra có ID mapping khác nhau; student hynt là TorchScript export. Chúng chưa phải bộ checkpoint tương thích sẵn với ví dụ bên dưới. Repo không tự remap weights hoặc chuyển JIT.

## 4. Train encoder-only KD

Dùng đường dẫn tuyệt đối vì launcher chuyển thư mục làm việc vào recipe icefall:

```bash
export BPE_MODEL=/absolute/path/bpe.model
export TEACHER_CHECKPOINT=/absolute/path/teacher.pt
export STUDENT_CHECKPOINT=/absolute/path/student.pt  # tùy chọn
export MANIFEST_DIR=/absolute/path/manifests
export EXP_DIR=/absolute/path/exp/distill
bash configs/encoder_kd.sh
```

**Sửa kiến trúc trong `configs/encoder_kd.sh` cho đúng checkpoint của bạn.** File mẫu là student streaming 6 stack nhỏ và teacher non-streaming; không phải architecture được tự nhận dạng từ các pretrained links. Có thể ghi đè tham số bằng cách thêm ở cuối:

```bash
bash configs/encoder_kd.sh --max-duration 60 --base-lr 0.001 --num-epochs 10
bash configs/encoder_kd.sh --kd-type both --kd-logit-scale 0.1
bash configs/encoder_kd.sh --kd-type logit --kd-temperature 2.0
```

`--max-duration` là tổng thời lượng audio tính bằng giây mỗi batch trên mỗi GPU, không phải số utterance. Ví dụ dùng 100 giây làm điểm bắt đầu; chưa benchmark bộ model đầy đủ trên 3090. Giảm nếu OOM. Learning rate upstream mặc định `0.045`; khi fine-tune pretrained student nên thử LR nhỏ hơn và theo dõi validation, không mặc định dùng LR train từ đầu.

## 5. Loại KD và trọng số

| `--kd-type` | Tín hiệu distillation | Projection encoder |
|---|---|---|
| `encoder` | Encoder representations | Có, linear student → teacher dim |
| `logit` | KL trên joiner output distributions | Không |
| `both` | Encoder + logit KD | Có |
| `none` | RNNT thông thường, teacher không được load | Không |

CLI mặc định là `both`; **config mẫu chọn `encoder`**. Không có loss riêng so sánh decoder hidden states. Logit KD truyền gradient qua student encoder, decoder và joiner; encoder KD truyền gradient qua student encoder và projection. Teacher luôn `eval()`, không nhận gradient và không nằm trong optimizer/checkpoint student.

| Tham số | Mặc định CLI | Ý nghĩa |
|---|---:|---|
| `--kd-encoder-scale` | 1.0 | Trọng số encoder KD |
| `--kd-logit-scale` | 1.0 | Trọng số logit KD |
| `--kd-encoder-loss` | cosine | `cosine`, `mse`, hoặc `l1` |
| `--kd-temperature` | 2.0 | Nhiệt độ logit softmax; phải > 0 |
| `--kd-warmup-steps` | 0 | Ramp trọng số KD; config mẫu dùng 2000 |
| `--simple-loss-scale` | 0.5 | RNNT simple weight sau warmup |
| `--prune-range` | 5 | Số label positions giữ lại mỗi frame |
| `--use-ctc` | false | Thêm CTC loss cho student nếu bật |
| `--ctc-loss-scale` | 0.2 | Trọng số CTC khi bật |

Công thức:

```text
L = a(step) × L_simple_RNNT + b(step) × L_pruned_RNNT
    + ctc_loss_scale × L_CTC                        # khi use_ctc=true
    + ramp(step) × (kd_encoder_scale × L_encoder
                   + kd_logit_scale × L_logit)

ramp(step) = min(1, step / kd_warmup_steps), hoặc 1 nếu warmup=0
```

`a` giảm từ 1 xuống `simple_loss_scale`; `b` tăng từ 0.1 lên 1 trong `warm_step=2000` bước mặc định của upstream. Vì trọng số thay đổi trong warmup, không nên kết luận học tốt/xấu chỉ từ tổng loss giữa các thời điểm khác trọng số.

Encoder loss là tổng trên **valid frames**: cosine dùng `1 - cosine_similarity`, MSE/L1 lấy trung bình chiều feature. Nếu frame rate khác, teacher được nội suy từng utterance theo độ dài thật, bỏ padding.

Logit KD là `T² × KL(teacher || student)` trên **cùng student pruned ranges**, lấy trung bình qua các pruned positions rồi tổng qua valid frames. Scale vì vậy không bị nhân trực tiếp theo prune range. Encoder interpolation chỉ là xấp xỉ alignment khi frame rates khác.

Bắt đầu với encoder cosine scale `1.0`, warmup `2000` như config; đây là điểm thử, không phải tham số tối ưu đã benchmark. Theo dõi `kd_enc_loss`, `kd_logit_loss`, RNNT losses và WER. Nếu KD lấn át RNNT, giảm scale; nếu chuyển cosine → MSE, cần cân chỉnh lại vì độ lớn loss khác nhau. Đặt scale bằng 0 vẫn tính nhánh KD đã chọn; dùng `--kd-type` để bỏ hẳn nhánh không cần.

## 6. Resume, nhiều GPU và inference

Giữ nguyên kiến trúc/KD type/vocab khi resume:

```bash
# Đã có epoch-5.pt trong EXP_DIR:
bash configs/encoder_kd.sh --start-epoch 6 --num-epochs 30
# Resume checkpoint-10000.pt:
bash configs/encoder_kd.sh --start-batch 10000 --num-epochs 30
# Máy có 2 GPU:
CUDA_VISIBLE_DEVICES=0,1 bash configs/encoder_kd.sh --world-size 2
```

Student init được bỏ qua khi resume; teacher checkpoint vẫn cần tồn tại vì teacher không lưu trong checkpoint student. Multi-GPU được hỗ trợ bằng spawn/DDP nhưng chưa kiểm thử multi-process ở repo này.

Validation loss gồm cả KD, nên `best-valid-loss.pt` không đồng nghĩa best WER. Đánh giá WER riêng trên validation/test tiếng nói thật.

Inference chỉ cần student. Khi dùng decoder/export upstream, bỏ các keys `kd_proj.*` trong student state dict (và `model_avg` nếu dùng), hoặc cho phép đúng các keys thừa đó khi load. Repo chưa cung cấp quy trình export/decode tự động.

Không hỗ trợ CTC-only, attention decoder loss hoặc CR-CTC; CLI từ chối các cấu hình này.

## 7. Chạy tests

```bash
python scripts/download_test_tokenizer.py
export PYTHONPATH="$PWD/.deps:$PWD/icefall${PYTHONPATH:+:$PYTHONPATH}"
python test_kd.py
python smoke_train.py
python test_training_integration.py
python test_loss_decrease.py
```

Integration tests dùng GPU CUDA và model nhỏ khởi tạo ngẫu nhiên; test helpers có thể chạy CPU. Tokenizer chỉ dùng để tạo labels giả lập, không download pretrained models.

| Kiểm tra | Kết quả |
|---|---|
| Padding, nội suy từng utterance, KL identity, gradient masking | Pass |
| Encoder KD FP32 và FP16 | Pass |
| Both KD FP16; logit-only FP32; none FP32 | Pass |
| ScaledAdam/Eden, validation, model averaging, strict model reload | Pass |
| Teacher frozen, weights không thay đổi, không lưu trong student | Pass |
| Full dataset pipeline, multi-process DDP, optimizer resume | Chưa kiểm thử |

Test 100 bước dùng batch cố định, weight loss cố định, đo eval loss trên chính batch đó:

| Loss | Bước 0 | Bước 100 |
|---|---:|---:|
| Combined | 542.0110 | 23.4347 |
| Simple RNNT | 324.3256 | 17.1679 |
| Pruned RNNT | 328.0008 | 14.6027 |
| Encoder cosine KD | 51.8474 | 0.2481 |

Số liệu từng 10 bước: [`results/loss_history.json`](results/loss_history.json). Test lưu checkpoint giả lập trong `loss_decrease_exp/`; không dùng nó làm ASR pretrained model.

## License

Apache-2.0, xem [LICENSE](LICENSE). Phần RNNT trong `train_distill.py` dựa trên icefall `model.py` (Xiaomi Corp. và các tác giả upstream). License repo này áp dụng cho code, không thay thế license của checkpoint/tokenizer/dataset bên ngoài.
