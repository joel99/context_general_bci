
# Defunct.
root_folder = './data/'
root_folder = './data/preprocessed/'
import shutil
from pathlib import Path
import os
source = 'pitt_co'
source = 'pitt_grasp'
def rename_folder(root):
    for p in Path(root).glob("*.pth"):
        print(p)
        # move to pitt_co_alter
        snippets = p.name.split('_')
        name_index = -1
        for i, x in enumerate(snippets):
            if any([y in x for y in ['Home', 'Lab']]):
                name_index = i
        if name_index == -1:
            raise ValueError(f"Could not find Home or Lab in {p.name}")
        nametag = snippets[name_index]
        loc = 'Home' if 'Home' in nametag else 'Lab'
        subj = nametag.replace(loc, '')
        rename_subj = REDACT_MAP[subj] # True code to redacted code
        new_nametag = f"{rename_subj}{loc}"
        snippets[name_index] = new_nametag
        new_name = '_'.join(snippets)
        shutil.move(p, f'{root}/{new_name}')
        
for folder in ['eval', 'calib']:
    if not os.path.exists(f'{root_folder}{folder}/{source}_alter'):
        os.mkdir(f'{root_folder}{folder}/{source}_alter')
    # note we shouldn't recurse as the filenames registered in meta.csv are old
    # for p in Path(f'{root_folder}{folder}/{source}').glob("*"):
        # if p.is_dir():
            # rename_folder(p)
    # then do top level rename
    rename_folder(f'{root_folder}{folder}/{source}')
exit(0)