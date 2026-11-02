# Compare representative Chapter 3 results with their stored numerical references.

from pathlib import Path
import json
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run


def main():
    cases=json.loads((ROOT/'tests/chapter3_cases.json').read_text('utf-8'))
    output=ROOT/'outputs/study_check'
    output.mkdir(parents=True,exist_ok=True)
    comparisons=0
    for case in cases:
        study,key=case['study'],case['key']
        key_column=run.STUDIES[study][1]
        folder=output/study/key
        subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'run',study,'--where',key_column+'='+key,'--limit','1','--output',str(folder)],check=True,cwd=ROOT)
        con=sqlite3.connect(folder/'results.sqlite')
        items=json.loads(con.execute('SELECT result FROM cases WHERE study=? AND k=?',(study,key)).fetchone()[0]);con.close()
        actual={(x['method'],x.get('arm')):x['result'] for x in items}
        for expected in case['expected']:
            row=actual[(expected['method'],expected.get('arm'))]
            for metric,value in expected['metrics'].items():
                got=row[metric]
                equal=got is None if value is None else got is not None and abs(float(got)-float(value))<=1e-10
                if not equal:raise AssertionError(f'{study} {key} {metric}: {got} != {value}')
                comparisons+=1
    summary=dict(status='passed',cases=len(cases),scalar_comparisons=comparisons,historical_images_used=0)
    (output/'check.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(summary)


if __name__=='__main__':main()
