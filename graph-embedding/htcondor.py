import htcondor
import json
import os

# Garantir estrutura de logs local
os.makedirs('logs', exist_ok=True)

with open('mappings.json', 'r') as f:
    data = json.load(f)

sub = htcondor.Submit({
    "executable": "launcher.sh",
    "arguments": "$(Process)",
    "transfer_input_files": "train_job.py, mappings.json, models/, mapping/",
    "output": "logs/job_$(Process).out",
    "error": "logs/job_$(Process).err",
    "log": "logs/cluster.log",

    # Gerenciamento de Vagas (Fair Share)
    "max_jobs_running": "5",

    # Recursos
    "request_cpus": "1",
    "request_memory": "8GB",
    "request_gpus": "1",

    # Transferência de arquivos
    "should_transfer_files": "YES",
    "when_to_transfer_output": "ON_EXIT",
    # Garante que a pasta 'results' criada no node volte para você
    "transfer_output_files": "results"
})

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn, count=len(data))

print(f"🚀 Cluster {cluster_id} iniciado. Cada job usará um base_path único.")