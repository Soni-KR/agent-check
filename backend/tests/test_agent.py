import json

import httpx
import pytest

import agent
import evaluator
from database import get_db
from main import app


def call(name, arguments, call_id='call_1'):
    return {'type': 'function_call', 'call_id': call_id, 'name': name,
            'arguments': json.dumps(arguments)}


def message(text):
    return {'type': 'message', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': text}]}


def provider(monkeypatch, handler):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-not-real')
    original_client = httpx.Client
    monkeypatch.setattr(agent.httpx, 'Client',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))


def output(items):
    return httpx.Response(200, json={'status': 'completed', 'output': items})


def test_agent_executes_tools_reuses_observations_and_persists_memory(client, monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['store'] is False
        assert body['tools'][0]['name'] == 'calculate'
        if len(requests) == 1:
            return output([call('calculate', {'expression': '12 * 4.5 * 1.19'}, 'calc')])
        observation = body['input'][-1]
        assert observation['type'] == 'function_call_output'
        data = json.loads(observation['output'])
        if len(requests) == 2:
            assert observation['call_id'] == 'calc'
            assert data['result'] == pytest.approx(64.26)
            return output([call('save_note', {'title': 'Notebook budget', 'content': '64.26'}, 'save')])
        if len(requests) == 3:
            assert data['saved'] is True
            return output([call('read_note', {'title': 'Notebook budget'}, 'read')])
        assert data['note']['content'] == '64.26'
        return output([message('The total is 64.26, saved and verified in Notebook budget.')])

    provider(monkeypatch, handler)
    response = client.post('/api/tasks', json={'goal': 'Calculate and save the notebook budget, then verify it.'})
    assert response.status_code == 201
    task = response.json()
    assert task['status'] == 'completed'
    assert task['steps'] == 4 and task['tool_calls'] == 3
    assert [item['name'] for item in task['trace'] if item['type'] == 'tool'] == ['calculate', 'save_note', 'read_note']
    assert client.get('/api/tasks').json() == [task]
    assert client.get(f'/api/tasks/{task["id"]}').json() == task
    assert client.get('/api/notes').json() == [{'title': 'Notebook budget', 'content': '64.26'}]


def test_memory_is_available_to_a_new_task(client, monkeypatch):
    turn = 0

    def handler(request):
        nonlocal turn
        turn += 1
        if turn == 1:
            return output([call('save_note', {'title': 'meeting', 'content': 'Tuesday at 10'})])
        if turn == 2:
            return output([message('Saved.')])
        if turn == 3:
            return output([call('search_notes', {'query': 'meeting'})])
        observation = json.loads(json.loads(request.content)['input'][-1]['output'])
        assert observation['notes'][0]['content'] == 'Tuesday at 10'
        return output([message('Your meeting is Tuesday at 10.')])

    provider(monkeypatch, handler)
    assert client.post('/api/tasks', json={'goal': 'Remember my meeting.'}).json()['status'] == 'completed'
    task = client.post('/api/tasks', json={'goal': 'When is my meeting?'}).json()
    assert task['answer'] == 'Your meeting is Tuesday at 10.'
    assert task['tool_calls'] == 1


def test_step_limit_stops_tool_loop(client, monkeypatch):
    provider(monkeypatch, lambda request: output([call('calculate', {'expression': '1+1'})]))
    task = client.post('/api/tasks', json={'goal': 'Keep calculating', 'max_steps': 2}).json()
    assert task['status'] == 'limit'
    assert task['steps'] == 2 and task['tool_calls'] == 2
    assert task['answer'] == ''


def test_tool_limit_stops_multiple_calls(client, monkeypatch):
    provider(monkeypatch, lambda request: output([
        call('calculate', {'expression': '1+1'}, f'call_{i}') for i in range(15)]))
    task = client.post('/api/tasks', json={'goal': 'Calculate many things'}).json()
    assert task['status'] == 'limit'
    assert task['tool_calls'] == 12


def test_time_limit_stops_before_tool_execution(client, monkeypatch):
    times = iter([0, 0, 61, 61])
    monkeypatch.setattr(agent, 'monotonic', lambda: next(times))
    provider(monkeypatch, lambda request: output([call('save_note', {'title': 'late', 'content': 'no'})]))
    task = client.post('/api/tasks', json={'goal': 'Save a note'}).json()
    assert task['status'] == 'limit'
    assert task['tool_calls'] == 0
    assert client.get('/api/notes').json() == []


def test_provider_failure_is_saved_without_leaking_error_body(client, monkeypatch):
    provider(monkeypatch, lambda request: httpx.Response(401, json={'error': 'private-details'}))
    task = client.post('/api/tasks', json={'goal': 'Test failure'}).json()
    assert task['status'] == 'error'
    assert 'private-details' not in str(task)
    assert 'test-key-not-real' not in str(task)
    assert client.get('/api/tasks').json()[0]['status'] == 'error'


def test_missing_key_does_not_run_or_save_a_task(client):
    response = client.post('/api/tasks', json={'goal': 'Calculate 2+2'})
    assert response.status_code == 400
    assert 'OPENAI_API_KEY' in response.json()['detail']
    assert client.get('/api/tasks').json() == []


@pytest.mark.parametrize('body', [{'goal':'  '}, {'goal':'x','max_steps':0},
                                 {'goal':'x','max_steps':9}, {'goal':'x','max_steps':2.5},
                                 {'goal':'x','tools':['shell']}])
def test_invalid_task_request(client, body):
    assert client.post('/api/tasks', json=body).status_code == 422


@pytest.mark.parametrize('expression', ["__import__('os').system('whoami')", '2**999',
                                       '[1,2]', 'x+1', '1/0', '1e200', 'True+1'])
def test_calculator_blocks_code_and_unbounded_values(client, expression):
    with next(app.dependency_overrides[get_db]()) as db:
        assert agent.execute_tool('calculate', {'expression': expression}, db)['ok'] is False


@pytest.mark.parametrize('name,args', [('shell', {'command': 'anything'}),
                                     ('save_note', {'title': 'a','content':'b','path':'c'}),
                                     ('calculate', {'expression': 12})])
def test_tool_allowlist_and_validation(client, name, args):
    with next(app.dependency_overrides[get_db]()) as db:
        assert agent.execute_tool(name, args, db)['ok'] is False
    assert client.get('/api/notes').json() == []


def test_agent_can_correct_bad_tool_arguments(client, monkeypatch):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            item = call('calculate', {})
            item['arguments'] = 'not-json'
            return output([item])
        observation = json.loads(json.loads(request.content)['input'][-1]['output'])
        if count == 2:
            assert observation['ok'] is False
            return output([call('calculate', {'expression': '21*2'}, 'corrected')])
        assert observation['result'] == 42
        return output([message('42')])

    provider(monkeypatch, handler)
    task = client.post('/api/tasks', json={'goal': 'Calculate 21*2'}).json()
    assert task['status'] == 'completed' and task['answer'] == '42'
    assert task['trace'][0]['result']['ok'] is False
    assert task['trace'][1]['result']['ok'] is True


def test_autonomous_evaluation_requires_real_calculator_action(client, monkeypatch):
    def without_tools(goal, db, **kwargs):
        case = next(case for case in evaluator.CASES if case['prompt'] == goal)
        return {'status':'completed','answer':case['expected'],'trace':[],'error':None}
    monkeypatch.setattr(evaluator, 'run_agent', without_tools)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-not-real')
    report = client.post('/api/runs', json={'agent':'autonomous'}).json()
    assert report['score'] == 75 and report['deployment_allowed'] is False
    assert report['results'][0]['passed'] is False


def test_autonomous_evaluation_keeps_trace(client, monkeypatch):
    provider(monkeypatch, lambda request: output([message('READY')]))
    report = client.post('/api/runs', json={'agent':'autonomous'}).json()
    assert report['deployment_allowed'] is False
    assert report['results'][1]['passed'] is True
    assert report['results'][1]['trace'][0]['text'] == 'READY'


def test_unknown_task_is_404(client):
    assert client.get('/api/tasks/unknown').status_code == 404
