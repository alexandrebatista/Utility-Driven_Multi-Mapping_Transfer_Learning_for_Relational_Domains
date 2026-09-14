#!/bin/bash
SOURCE=$1
TARGET=$2
GAMMA=$3
SIZE=$4
IDS=$5


# 1. Criar uma pasta para o venv e descompactar dentro dela
mkdir -p my_venv
tar -xzf venv.tar.gz -C my_venv

# 2. Ativar o ambiente (agora os caminhos relativos funcionam!)
source my_venv/bin/activate

# 3. Garantir que o Python encontre seus módulos locais
export PYTHONPATH=$PYTHONPATH:.

# 4. Executar o job
python3 multisource_job.py --source "$SOURCE" --target "$TARGET" --gamma "$GAMMA" --size "$SIZE" --ids "$IDS"