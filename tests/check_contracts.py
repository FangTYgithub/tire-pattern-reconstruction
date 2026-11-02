"""Check output isolation, dependency imports and result compatibility without rendering."""
from pathlib import Path
import argparse
import json
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True,help='An existing partial CH4-VALID result database.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.database.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if not source.is_file():raise FileNotFoundError(source)
    marker=out/'import_fixture';marker.mkdir(exist_ok=True)
    (marker/'sibling_dependency.py').write_text('VALUE = 7\n','utf-8')
    code="""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
sys.path.insert(0,sys.argv[2])
from runtime import setup,ENGINE
setup(Path(sys.argv[3]))
import sibling_dependency
assert sibling_dependency.VALUE==7
for p in (ENGINE/'forbidden.txt',Path(sys.argv[2])/'forbidden.txt'):
    try:open(p,'wb')
    except PermissionError:continue
    raise AssertionError('Write guard failed: '+str(p))
print('ISOLATION PASSED')
"""
    subprocess.run([sys.executable,'-B','-c',code,str(ROOT),str(marker),str(out/'guard')],check=True)
    refused=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'select-ch4','CH4-VALID','--database',str(source),'--output',str(out/'partial')],cwd=ROOT,text=True,capture_output=True)
    if refused.returncode==0 or 'Whole registered matrix required' not in refused.stderr:
        raise AssertionError(refused.stdout+refused.stderr)
    bad=out/'mismatch';bad.mkdir(exist_ok=True)
    with sqlite3.connect('file:'+source.as_posix()+'?mode=ro',uri=True) as src,sqlite3.connect(bad/'results.sqlite') as dst:
        src.backup(dst);dst.execute("UPDATE cases SET binding='deliberately-invalid'");dst.commit()
        key=dst.execute("SELECT k FROM cases WHERE study='CH4-VALID'").fetchone()[0]
    refused=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'run','CH4-VALID','--where','case_key='+key,'--limit','1','--output',str(bad)],cwd=ROOT,text=True,capture_output=True)
    if refused.returncode==0 or 'different code or inputs' not in refused.stderr:
        raise AssertionError(refused.stdout+refused.stderr)
    result=dict(status='passed',sibling_import=True,input_write_guard=True,incomplete_selection_rejected=True,incompatible_resume_rejected=True,new_renders=0)
    (out/'check.json').write_text(json.dumps(result,indent=2),'utf-8');print(result)


if __name__=='__main__':main()
