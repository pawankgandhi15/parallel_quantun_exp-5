import json

nb_path = r'F:\Pawan_Nitj\parallel_quantum\QC-CNN-Parallel1\QC_CNN_Parallel_Experiments.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

token = 'YOUR_GITHUB_PAT_HERE'

for cell in nb['cells']:
    if cell.get('id') == 'config-cell':
        new_src = []
        for line in cell['source']:
            if 'GITHUB_TOKEN = ' in line and 'ghp_YOUR_TOKEN_HERE' in line:
                line = 'GITHUB_TOKEN = "' + token + '"   # <- your token\n'
                print(f'Token updated: {line.strip()}')
            new_src.append(line)
        cell['source'] = new_src
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved.')
