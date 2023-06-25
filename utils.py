from pathlib import Path
from typing import List
import os


def rename_files(files: List[str], find: str, replace: str):
    for f in files:
        p = Path(f)
        new_name = p.name.replace(find, replace)
        print(f"{p.name} => {new_name}")
        print(Path.joinpath(p.parent, new_name))
        os.rename(p, Path.joinpath(p.parent, new_name))


rename_files(files=[f for f in Path('data/rp_release_notes_all/1').iterdir()], find='RPVersion', replace='Revolution Performance v')
rename_files(files=[f for f in Path('data/rp_release_notes_all/1').iterdir()], find='ReleaseNotes', replace=' Release Notes - Revolution Support')
