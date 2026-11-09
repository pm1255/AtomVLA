"""Adapter entry point for simulator and robot integrations.

The evaluation package deliberately does not contain a fake simulator.  A
dataset-specific adapter can be supplied with ``ATOMVLA_ADAPTER_MODULE``;
that module must expose ``evaluate(episode, variant)`` and return measured
evidence.  Without it, this program records an unavailable trial explicitly.
"""
import importlib
import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m adapters.<name> REQUEST.json')
    request_path = sys.argv[1]
    request = json.load(open(request_path, encoding='utf-8'))
    module_name = os.environ.get('ATOMVLA_ADAPTER_MODULE')
    if not module_name:
        result = {'success': False, 'steps': 0, 'retries': 0,
                  'contract_ok': False, 'status': 'unavailable',
                  'error': 'no_environment_adapter_configured'}
    else:
        module = importlib.import_module(module_name)
        fn = getattr(module, 'evaluate', None)
        if fn is None:
            raise SystemExit(f'{module_name} has no evaluate(episode, variant)')
        result = dict(fn(request['episode'], request['variant']))
        if 'success' not in result:
            raise SystemExit('adapter response omitted measured success')
    with open(request['output'], 'w', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

