"""Optional pipeline gate: exit 0 for pass, 1 for blocked, 2 for request failure."""
import argparse
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description='Evaluate an agent and return a gate exit code.')
    parser.add_argument('--agent', choices=['demo_safe', 'demo_unsafe', 'llm', 'autonomous'], default='demo_safe')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    try:
        response = httpx.post(f'{args.url.rstrip("/")}/api/runs', json={'agent': args.agent}, timeout=300)
        response.raise_for_status()
        report = response.json()
    except (httpx.HTTPError, ValueError):
        print('Evaluation request failed. Check the API and configuration.', file=sys.stderr)
        return 2
    print(f'{report["passed"]}/{report["total"]} checks passed. '
          f'Gate: {"PASS" if report["deployment_allowed"] else "BLOCKED"}.')
    return 0 if report['deployment_allowed'] else 1


if __name__ == '__main__':
    sys.exit(main())
