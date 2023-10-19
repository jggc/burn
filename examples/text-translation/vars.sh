source ~/devel/burn/venv/bin/activate
torch_dir="$(pip3 show torch | grep -i -F Location | awk '{print $2}')"
export LD_LIBRARY_PATH="$torch_dir/torch/lib"
export TORCH_CUDA_VERSION=cu113
export LIBTORCH_USE_PYTORCH=1

