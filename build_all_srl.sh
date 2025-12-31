#!/bin/bash
# filepath: build_all.sh

set -e

# 1. Executa o build.sh na pasta UtilityMultiDomainRDNBoost
cd SRLBoost
./build.sh

# 2. Renomeia o .jar gerado para BoostSRL.jar na pasta target
cd build/libs 
JAR_ORIGINAL=$(ls *.jar | head -n 1)
cp "$JAR_ORIGINAL" BoostSRL.jar

# 3. Copia BoostSRL.jar para srlearn/srlearn
cp BoostSRL.jar ../../../srlearn/srlearn/

# 4. Volta para a raiz do projeto e executa o build_and_install.sh na pasta srlearn
cd ../../../srlearn
./build_and_install.sh
