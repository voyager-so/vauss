#!/bin/bash
# Python environment for `vauss fill`'s fixer step on a Linux NVIDIA machine (no root needed).
#   bash setup_fixer.sh [fixer|difix|all]        (default fixer)
# Everything goes under $PREFIX (default ~/.local/share/splat-fixer):
#   venv/            python 3.12 venv: torch 2.8 cu128, lpips, and for `fixer` cosmos-predict2 1.0.9 (--no-deps) plus the
#                    modules its text2image path imports. Transformer Engine has no torch-2.8/Blackwell wheel and the
#                    Fixer DiT only needs RMSNorm, SDPA attention and RoPE, so shims/ stand in for transformer_engine and
#                    megatron.core (single GPU).
#   Fixer/           github.com/nv-tlabs/Fixer at a pinned commit (Apache-2.0); its model paths read $FIXER_MODELS.
#   fixer-models/    huggingface.co/nvidia/Fixer weights (5.5 GB).
#   Difix3D/         (difix/all only) github.com/nv-tlabs/Difix3D: the difix_ref weights are NON-COMMERCIAL.
# At the end it prints the environment `vauss fill` needs (FIXER_SRC, FIXER_MODELS, DIFIX_SRC, --fixer-python).
set -euo pipefail
WHAT=${1:-fixer}
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${PREFIX:-$HOME/.local/share/splat-fixer}
FIXER_COMMIT=b39dfcaf4eeec90dc943b057ff368c16252c6c6e
DIFIX_COMMIT=c76edc595586e16732c91ddee82f3a6d83a8a9cc
mkdir -p "$PREFIX"
export HF_HOME=${HF_HOME:-$PREFIX/hf} PIP_DISABLE_PIP_VERSION_CHECK=1 UV_CACHE_DIR=${UV_CACHE_DIR:-$PREFIX/uv-cache}
python3 -m uv --version > /dev/null 2>&1 || python3 -m pip install -q --user --break-system-packages uv
UV="python3 -m uv"
PY=$PREFIX/venv/bin/python
SP() { $PY -c 'import site; print(site.getsitepackages()[0])'; }
[ -x "$PY" ] || $UV venv -q -p python3.12 "$PREFIX/venv"
echo 'torch==2.8.0+cu128' > "$PREFIX/cons.txt"
$UV pip install -q -p "$PY" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q -p "$PY" -c "$PREFIX/cons.txt" 'numpy<2' Pillow lpips
if [ "$WHAT" = fixer ] || [ "$WHAT" = all ]; then
  $UV pip install -q -p "$PY" --no-deps cosmos-predict2==1.0.9
  $UV pip install -q -p "$PY" -c "$PREFIX/cons.txt" scipy einops attrs termcolor loguru omegaconf hydra-core fvcore iopath \
    'imageio[ffmpeg]' 'diffusers==0.33.1' 'transformers==4.51.3' 'huggingface-hub==0.32.4' safetensors sentencepiece ftfy \
    natsort pandas opencv-python-headless tqdm boto3 'peft==0.14.0' accelerate ninja jaxtyping rich packaging \
    typing_extensions setuptools wheel mediapy webdataset pyyaml protobuf nvidia-ml-py qwen-vl-utils
  # The Qwen text encoder module asserts flash-attn at import; Fixer never builds it (use_text_encoder=False).
  sed -i 's/^assert is_flash_attn_2_available().*$/pass  # vauss fill: Qwen text encoder unused/' "$(SP)/imaginaire/networks/qwen2_5_vl.py"
  cp -r "$HERE/shims/transformer_engine" "$HERE/shims/megatron" "$(SP)/"
  if [ ! -d "$PREFIX/Fixer" ]; then
    git clone -q https://github.com/nv-tlabs/Fixer "$PREFIX/Fixer"
    git -C "$PREFIX/Fixer" checkout -q $FIXER_COMMIT
    sed -i "s#'/work/models/base/#os.environ.get('FIXER_MODELS', '/work/models') + '/base/#" \
      "$PREFIX/Fixer/src/pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py"
  fi
  patch -N "$(SP)/cosmos_predict2/tokenizers/tokenizer.py" "$PREFIX/Fixer/tokenizer.patch" || true
  # Always resolve the snapshot (fast when complete; finishes a partial download) and record what was fetched.
  $PY -c "
import json
from huggingface_hub import HfApi, snapshot_download
info = HfApi().model_info('nvidia/Fixer')
snapshot_download('nvidia/Fixer', revision=info.sha, local_dir='$PREFIX/fixer-models', max_workers=8)
json.dump(dict(repo='nvidia/Fixer', revision=info.sha, licence=(info.card_data or {}).get('license'),
               licence_name=(info.card_data or {}).get('license_name'), note='NVIDIA Open Model License; Built on NVIDIA Cosmos'),
          open('$PREFIX/fixer-models/MANIFEST.json', 'w'), indent=1)"
fi
if [ "$WHAT" = difix ] || [ "$WHAT" = all ]; then
  echo "NOTE: Difix3D+ weights (nvidia/difix_ref) are licensed for NON-COMMERCIAL use only."
  [ -d "$PREFIX/Difix3D" ] || { git clone -q https://github.com/nv-tlabs/Difix3D "$PREFIX/Difix3D" && git -C "$PREFIX/Difix3D" checkout -q $DIFIX_COMMIT; }
  if [ "$WHAT" = difix ]; then
    $UV pip install -q -p "$PY" -c "$PREFIX/cons.txt" einops 'diffusers==0.25.1' 'transformers==4.38.0' 'huggingface-hub==0.25.1' \
      'peft==0.9.0' 'accelerate==0.27.2'
  fi
fi
cat <<EOF
SETUP_OK
export FIXER_SRC=$PREFIX/Fixer/src FIXER_MODELS=$PREFIX/fixer-models DIFIX_SRC=$PREFIX/Difix3D/src HF_HOME=$HF_HOME
vauss fill ... --fixer-python $PY
EOF
