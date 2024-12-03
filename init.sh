#!/bin/bash

set -e  # Faz o script parar se algum comando falhar

# 1. Verificar se o Python 3 está instalado
if ! command -v python3 &>/dev/null; then
    echo "Erro: Python 3 não está instalado. Por favor, instale o Python 3 e tente novamente."
    exit 1
fi

# 2. Criar o ambiente virtual
echo "Criando o ambiente virtual..."
if [ -d "venv" ]; then
    echo "O ambiente virtual já existe. Usando o existente."
else
    python3 -m venv venv || { echo "Erro ao criar o ambiente virtual."; exit 1; }
    echo "Ambiente virtual criado com sucesso."
fi

# 3. Ativar o ambiente virtual
echo "Ativando o ambiente virtual..."
source ./venv/bin/activate || { echo "Erro ao ativar o ambiente virtual."; exit 1; }

# 4. Instalar dependências
echo "Instalando dependências..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt || { echo "Erro ao instalar dependências."; deactivate; exit 1; }
    echo "Dependências instaladas com sucesso."
else
    echo "Aviso: O arquivo requirements.txt não foi encontrado. Nenhuma dependência instalada."
fi

# 5. Confirmação final
echo "Ambiente configurado e pronto para uso!"

