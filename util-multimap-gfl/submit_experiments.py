import htcondor
import json
import os
from itertools import combinations

# 1. Garantir estrutura de logs local
os.makedirs('logs', exist_ok=True)

# 2. Carrega a lista de mappings
with open('mappings.json', 'r') as f:
    mappings_data = json.load(f)

# Agrupar os índices da lista por par único (Source, Target)
pair_groups = {}
for index, m in enumerate(mappings_data):
    source = m['source']
    target = m['target']
    
    # Trava de segurança para o Twitter
    if target.lower() == "twitter":
        continue 
    
    key = (source, target)
    if key not in pair_groups:
        pair_groups[key] = []
    pair_groups[key].append(str(index))

gammas = [0.2, 1, 5]
group_sizes = [5, 4, 3, 2, 1]
job_items = []

# 3. Gerar a grade de experimentos (78 jobs por par se houverem 5 mapeamentos)
for (source, target), ids in pair_groups.items():
    # Garante que vai pegar no máximo os 5 primeiros mapeamentos deste par
    current_ids = ids[:5] 
    
    for size in group_sizes:
        if len(current_ids) < size:
            continue
            
        for combo in combinations(current_ids, size):
            mapping_ids_str = ",".join(combo)
            for g in gammas:
                job_items.append({
                    "source": source,
                    "target": target,
                    "gamma": str(g),
                    "size": str(size),
                    "ids": mapping_ids_str
                })

if not job_items:
    print("⚠️ Nenhum job válido para submissão. Verifique se os destinos são válidos.")
    exit()

# 4. Configuração do HTCondor
sub = htcondor.Submit({
    "executable": "multisource_launcher.sh",
    "arguments": "$(source) $(target) $(gamma) $(size) $(ids)",
    "transfer_input_files": "multisource_job.py, mappings.json, models, mapping, preprocessed, database, dependencies, venv.tar.gz",
    "output": "logs/$(source)_$(target)_sz$(size)_g$(gamma)_proc$(Process).out",
    "error": "logs/$(source)_$(target)_sz$(size)_g$(gamma)_proc$(Process).err",
    "log": "logs/cluster.log",
    "max_jobs_running": "20",
    "request_cpus": "1",
    "request_memory": "1500",
    "should_transfer_files": "YES",
    "when_to_transfer_output": "ON_EXIT",
    "transfer_output_files": "results"
})

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue_with_itemdata(txn, 1, iter(job_items))

print(f"🚀 Cluster {cluster_id} iniciado com {len(job_items)} jobs.")