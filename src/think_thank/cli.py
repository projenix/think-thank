"""Command-line interface. Credentials are read only from environment variables."""
import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import ExitStack

from pydantic import ValidationError

from .engine import ResearchEngine
from .llm import ModelClient, ModelError
from .models import Limits
from .reporting import render_report
from .store import Store
from .web import WebClient, WebError


def _limits(parser, defaults=False):
    values = Limits()
    for flag, kind, field in [
        ('max-iterations', int, 'max_iterations'), ('max-minutes', float, 'max_minutes'),
        ('max-tokens', int, 'max_tokens'), ('max-cost', float, 'max_cost_usd'),
        ('max-sources', int, 'max_sources'), ('reads-per-iteration', int, 'max_reads_per_iteration'),
        ('input-price', float, 'input_usd_per_million'), ('output-price', float, 'output_usd_per_million')]:
        parser.add_argument('--' + flag, dest=field, type=kind,
                            default=getattr(values, field) if defaults else None,
                            help=field.replace('_', ' ') + (' (cumulative session limit)' if flag.startswith('max-') else ''))


def _providers(parser):
    parser.add_argument('--model', default=None, help='OpenAI-compatible model name')
    parser.add_argument('--base-url', default=None, help='OpenAI-compatible API base URL')
    parser.add_argument('--search', choices=['duckduckgo', 'tavily'], default=None)
    parser.add_argument('--quiet', action='store_true')


def parser():
    root = argparse.ArgumentParser(prog='think-thank', description='Autonomous, evidence-driven web research with resumable sessions.')
    root.add_argument('--data-dir', type=Path, default=Path('.think-thank'), help='Local research storage (default: .think-thank)')
    commands = root.add_subparsers(dest='command', required=True)
    serve = commands.add_parser('serve', help='Launch the local research web app')
    serve.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost', '::1'])
    serve.add_argument('--port', type=int, default=8000)
    run = commands.add_parser('research', help='Investigate a topic using live web and model APIs')
    run.add_argument('topic')
    _limits(run, defaults=True)
    _providers(run)
    resume = commands.add_parser('resume', help='Resume a saved session; budget overrides are cumulative totals')
    resume.add_argument('session_id')
    _limits(resume)
    _providers(resume)
    commands.add_parser('list', help='List saved sessions')
    show = commands.add_parser('show', help='Print the current Markdown report or JSON state')
    show.add_argument('session_id')
    show.add_argument('--json', action='store_true', dest='as_json')
    demo = commands.add_parser('demo', help='Run a fictional, entirely offline end-to-end demonstration')
    demo.add_argument('--quiet', action='store_true')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'serve':
            from .server import serve
            serve(data_dir=args.data_dir, host=args.host, port=args.port)
            return 0
        with ExitStack() as stack:
            store = Store(args.data_dir)
            stack.callback(store.close)
            if args.command == 'list':
                rows = store.list_sessions()
                if not rows:
                    print('No research sessions yet.')
                for row in rows:
                    print('{id}  {status:10}  iteration={iteration}  {topic}'.format(**row))
                return 0
            if args.command == 'show':
                state = store.load(args.session_id)
                print(json.dumps(state.model_dump(), indent=2, ensure_ascii=False) if args.as_json else render_report(state))
                return 0
            state = store.load(args.session_id) if args.command == 'resume' else None
            if state and state.status == 'completed':
                print('Session is already completed. Start a new research session to investigate further.')
                print(store.root / 'sessions' / state.id / 'report.md')
                return 0
            config = dict(state.config) if state else {}
            is_demo = args.command == 'demo' or config.get('mode') == 'demo'
            if is_demo:
                from .demo import DEMO_TOPIC, DemoModel, DemoWeb
                model, web = DemoModel(), DemoWeb()
                config = {'mode': 'demo'}
            else:
                config = {
                    'mode': 'live',
                    'model': args.model or config.get('model') or os.getenv('THINK_THANK_MODEL', 'gpt-4.1-mini'),
                    'base_url': args.base_url or config.get('base_url') or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                    'search': args.search or config.get('search') or os.getenv('THINK_THANK_SEARCH', 'duckduckgo'),
                }
                model = ModelClient(model=config['model'], base_url=config['base_url'], api_key=os.getenv('OPENAI_API_KEY'))
                stack.callback(model.close)
                web = WebClient(search_provider=config['search'], tavily_api_key=os.getenv('TAVILY_API_KEY'))
                print('Live research sends your topic and retrieved excerpts to the configured model provider.', file=sys.stderr)
                print('Cost limits are estimates at configured prices, not a billing guarantee.', file=sys.stderr)
            if is_demo:
                stack.callback(model.close)
            stack.callback(web.close)
            progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
            engine = ResearchEngine(model, web, store, progress=progress)
            if state:
                updates = {name: getattr(args, name) for name in Limits.model_fields
                           if getattr(args, name, None) is not None}
                state.limits = Limits.model_validate({**state.limits.model_dump(), **updates})
                state.config = config
            else:
                limits = Limits() if is_demo else Limits(**{name: getattr(args, name) for name in Limits.model_fields})
                state = engine.create(DEMO_TOPIC if is_demo else args.topic, limits=limits, config=config)
            print('Session: ' + state.id, file=sys.stderr, flush=True)
            state = engine.run(state)
            print('Status: ' + state.status)
            print('Reason: ' + state.stop_reason)
            print('Sources: %d | Evidence: %d | Iterations: %d' % (len(state.sources), len(state.evidence), state.iteration))
            print('Estimated model cost: $%.4f | Tokens: %d' % (state.estimated_cost_usd, state.tokens_used))
            if state.report:
                print('Report: ' + str(store.root / 'sessions' / state.id / 'report.md'))
                print('JSON:   ' + str(store.root / 'sessions' / state.id / 'report.json'))
            return 130 if state.status == 'paused' else (1 if state.status == 'failed' else 0)
    except (ModelError, WebError, ValueError, ValidationError, OSError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
