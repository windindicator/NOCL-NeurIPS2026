"""Compare measured NOCL results with the identified paper or rebuttal record."""
import argparse
import json
import math
from pathlib import Path


def compare_run(directory, targets):
    data = json.loads((directory / 'metrics.json').read_text())
    if data.get('smoke') or data['status'] != 'completed' or data['method'] != 'nocl':
        raise ValueError(f'{directory}: use a completed full NOCL run')
    problem = data['problem']
    reference = targets[problem]
    if reference.get('rd_protocol') != data['config'].get('rd_protocol'):
        raise ValueError('RD protocol differs from the reference record')
    rows = []
    for target in reference['values']:
        matching = [row for row in data['results'] if
                    len(row['coefficients']) == len(target['coefficients']) and
                    all(math.isclose(a, b, abs_tol=2e-6) for a, b in
                        zip(row['coefficients'], target['coefficients']))]
        if not matching:
            rows.append({**target, 'status': 'missing'})
            continue
        measured = matching[-1]['relative_l2']
        expected = target['relative_l2']
        if reference['comparison'] == 'three_significant_digits':
            match = float(f'{measured:.3g}') == expected
        else:
            match = math.isclose(measured, expected, rel_tol=2e-5, abs_tol=2e-7)
        rows.append({'coefficients': target['coefficients'],
                     'reference_relative_l2': expected, 'measured_relative_l2': measured,
                     'relative_deviation': (measured - expected) / expected,
                     'status': 'match' if match else 'mismatch'})
    return {'run': str(directory), 'problem': problem, 'method': 'nocl',
            'reference_source': reference['source'], 'comparison': reference['comparison'],
            'all_values_match': all(row['status'] == 'match' for row in rows),
            'comparisons': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--strict', action='store_true', help='Nonzero exit if any reference value differs')
    args = parser.parse_args()
    targets = json.loads((Path(__file__).parent / 'records/reference_targets.json').read_text())['targets']
    try:
        records = [compare_run(run, targets) for run in args.runs]
    except ValueError as error:
        parser.error(str(error))
    content = json.dumps(records, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)
    if args.strict and not all(record['all_values_match'] for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
