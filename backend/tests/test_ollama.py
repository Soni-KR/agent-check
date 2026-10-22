import json

import httpx
import pytest

import agent
import providers
from test_agent import provider


def local_provider(monkeypatch, handler):
    provider(monkeypatch, handler)
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('OLLAMA_MODEL', 'qwen2.5:7b')
    monkeypatch.setenv('OLLAMA_URL', 'http://127.0.0.1:11434')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)


def test_local_loop_uses_native_tools_and_observations_without_a_key(client, monkeypatch):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        assert str(request.url) == 'http://127.0.0.1:11434/api/chat'
        assert 'authorization' not in request.headers
        body = json.loads(request.content)
        assert body['model'] == 'qwen2.5:7b'
        assert body['stream'] is False
        assert body['tools'][0]['function']['name'] == 'calculate'
        if count == 1:
            return httpx.Response(200, json={'done':True, 'message':{
                'role':'assistant', 'content':'', 'tool_calls':[
                    {'function':{'name':'calculate','arguments':{'expression':'21*2'}}}]}})
        assert body['messages'][-2]['tool_calls'][0]['function']['name'] == 'calculate'
        observation = body['messages'][-1]
        assert observation['role'] == 'tool'
        assert observation['tool_name'] == 'calculate'
        assert json.loads(observation['content'])['result'] == 42
        return httpx.Response(200, json={'done':True, 'message':{'role':'assistant','content':'42'}})

    local_provider(monkeypatch, handler)
    task = client.post('/api/tasks', json={'goal':'Calculate 21*2'}).json()
    assert task['status'] == 'completed' and task['answer'] == '42'
    assert task['provider'] == 'ollama' and task['tool_calls'] == 1


def test_local_health_checks_installed_model_without_a_key(client, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setattr(providers.httpx, 'get', lambda *a, **kw: httpx.Response(
        200, request=httpx.Request('GET','http://localhost/api/tags'),
        json={'models':[{'name':'qwen2.5:7b'}]}))
    status = providers.runtime_status()
    assert status['llm_configured'] is True
    assert status['provider'] == 'ollama'


def test_local_health_reports_missing_model(client, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setattr(providers.httpx, 'get', lambda *a, **kw: httpx.Response(
        200, request=httpx.Request('GET','http://localhost/api/tags'), json={'models':[]}))
    status = providers.runtime_status()
    assert status['llm_configured'] is False
    assert 'ollama pull' in status['runtime_message']


def test_local_health_handles_stopped_server(client, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    def unavailable(*a, **kw):
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(providers.httpx, 'get', unavailable)
    assert providers.runtime_status()['llm_configured'] is False


def test_local_incomplete_response_fails_without_a_final_answer(client, monkeypatch):
    local_provider(monkeypatch, lambda request: httpx.Response(200, json={
        'done':True,'done_reason':'length','message':{'content':'partial'}}))
    task = client.post('/api/tasks', json={'goal':'Test incomplete output'}).json()
    assert task['status'] == 'error' and task['answer'] == ''


def test_local_single_response_mode_does_not_enable_tools(client, monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert 'tools' not in body
        return httpx.Response(200, json={'done':True,'message':{'content':'READY'}})
    local_provider(monkeypatch, handler)
    result = client.post('/api/runs', json={'agent':'llm'}).json()
    assert result['provider'] == 'ollama'
    assert result['results'][1]['passed'] is True


def test_invalid_local_tool_arguments_fail_closed(client, monkeypatch):
    local_provider(monkeypatch, lambda request: httpx.Response(200, json={
        'done':True,'message':{'content':'','tool_calls':[
            {'function':{'name':'save_note','arguments':'invalid'}}]}}))
    task = client.post('/api/tasks', json={'goal':'Save a note'}).json()
    assert task['status'] == 'error' and task['tool_calls'] == 0
    assert client.get('/api/notes').json() == []


def test_unsupported_provider_is_rejected(client, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','unknown')
    assert client.post('/api/tasks', json={'goal':'Calculate 2+2'}).status_code == 400
