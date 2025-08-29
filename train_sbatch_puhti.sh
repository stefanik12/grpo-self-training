#!/bin/bash
#SBATCH --job-name=train_loras_nllb_langs
#SBATCH --account=project_2001194
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=gpu
##SBATCH --out=gpu.%J.log
##SBATCH --err=gpu.%J.log
##SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --gres=gpu:v100:1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/mstefani/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    echo "Initializing conda"
    eval "$__conda_setup"
else
    if [ -f "/users/mstefani/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/users/mstefani/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/mstefani/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate /scratch/project_2001194/mstefani/conda_env_grpo-self-training/conda_env
# from here on, we assume a set and activated python env

pip install -y uv
export UV_PROJECT_ENVIRONMENT=/scratch/project_2001194/mstefani/conda_env_grpo-self-training/conda_env
export UV_CACHE_DIR=/scratch/project_2001194/mstefani/conda_env_grpo-self-training/uv_cache
export HF_HOME=/scratch/project_2001194/mstefani/conda_env_grpo-self-training/hf_home

uv sync --cache-dir /scratch/project_2001194/mstefani/conda_env_grpo-self-training/uv_cache

python train.py --config config_bt.yaml

srun python train.py --config config_bt.yaml

