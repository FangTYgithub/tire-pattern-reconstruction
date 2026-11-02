"""Supplementary checks. --run: compute; default: compare saved results."""
from pathlib import Path
import argparse
import json
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/supplement_check')
    args = parser.parse_args()
    cases = json.loads((ROOT/'tests/supplement_cases.json').read_text('utf-8'))
    comparisons = 0
    for case in cases:
        study, key = case['study'], case['key']
        folder = args.output.resolve()/case['name']
        if args.run:
            subprocess.run([sys.executable, '-B', str(ROOT/'run.py'), 'run', study,
                            '--where', run.STUDIES[study][1]+'='+key, '--limit', '1', '--output', str(folder)],
                           check=True, cwd=ROOT)
        database = folder/'results.sqlite'
        with sqlite3.connect('file:'+database.as_posix()+'?mode=ro', uri=True) as db:
            record = db.execute('SELECT result FROM cases WHERE study=? AND k=?', (study, key)).fetchone()
        if record is None:
            raise AssertionError(f'Missing case: {study} {key}')
        items = {x['method']: x for x in json.loads(record[0])}
        actual = {k: x['result'] for k,x in items.items()}
        for expected in case['expected']:
            row = actual[expected['method']]
            for metric, value in expected['result'].items():
                got = row[metric]
                equal = got == value if value is None or isinstance(value, str) else got is not None and abs(float(got)-float(value)) <= 1e-10
                if not equal:
                    raise AssertionError(f'{study} {key} {metric}: {got} != {value}')
                comparisons += 1
        for expected in case.get('expected_layers', []):
            layers = items[expected['method']]['sampling_layers']
            row = next(x for x in layers if x['layer']==expected['layer'] and x['domain']==expected['domain'])
            for metric, value in expected['metrics'].items():
                if abs(float(row[metric])-float(value)) > 1e-10:
                    raise AssertionError(f'{study} {key} {expected["layer"]} {metric}')
                comparisons += 1
        print('PASS', case['name'])
    print(json.dumps(dict(cases=len(cases), comparisons=comparisons, rendered_this_call=bool(args.run))))


if __name__ == '__main__':
    main()
