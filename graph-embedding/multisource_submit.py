import htcondor
import json
import os

os.makedirs('logs', exist_ok=True)

with open('mappings.json', 'r') as f:
    mappings_data = json.load(f)

unique_pairs = set()

# CORREÇÃO: Verifica se é lista ou dicionário para evitar o erro de .values()
if isinstance(mappings_data, list):
    for details in mappings_data:
        unique_pairs.add((details['source'], details['target']))
elif isinstance(mappings_data, dict):
    for details in mappings_data.values():
        unique_pairs.add((details['source'], details['target']))

job_items = [{"source": s, "target": t} for s, t in unique_pairs]

sub = htcondor.Submit({
    "executable": "multisource_launcher.sh",
    "arguments": "$(source) $(target)",
    "transfer_input_files": "multisource_job.py, mappings.json, models, mapping, preprocessed, database, dependencies, venv.tar.gz",
    "output": "logs/job_$(source)_$(target).out",
    "error": "logs/job_$(source)_$(target).err",
    "log": "logs/cluster.log",
    "max_jobs_running": "5",
    "request_cpus": "1",
    "request_memory": "1300",
    "request_gpus": "0",
    "should_transfer_files": "YES",
    "when_to_transfer_output": "ON_EXIT",
    "transfer_output_files": "results"
})

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue_with_itemdata(txn, 1, iter(job_items))

print(f"🚀 Cluster {cluster_id} iniciado. {len(job_items)} jobs criados.")