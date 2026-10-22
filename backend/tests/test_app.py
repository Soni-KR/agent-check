import httpx
import pytest

import evaluator


def test_guarded_agent_passes_and_report_is_persisted(client):
    response = client.post('/api/runs', json={'agent': 'demo_safe'})
    assert response.status_code == 201
    report = response.json()
    assert report['score'] == 100
    assert report['deployment_allowed'] is True
    assert report['created_at'].endswith('+00:00')
    assert all(row['passed'] for row in report['results'])
    assert client.get('/api/runs').json() == [report]
    assert client.get(f'/api/runs/{report["id"]}').json() == report


def test_injection_blocks_vulnerable_agent(client):
    report = client.post('/api/runs', json={'agent': 'demo_unsafe'}).json()
    assert report['score'] == 75
    assert report['deployment_allowed'] is False
    failures = [row['id'] for row in report['results'] if not row['passed']]
    assert failures == ['injection']


def test_invalid_agent_is_rejected(client):
    assert client.post('/api/runs', json={'agent': 'unknown'}).status_code == 422


def test_extra_fields_are_rejected(client):
    assert client.post('/api/runs', json={'agent': 'demo_safe', 'score': 100}).status_code == 422


def test_missing_llm_key_is_clear_and_does_not_save_a_run(client):
    response = client.post('/api/runs', json={'agent': 'llm'})
    assert response.status_code == 400
    assert 'OPENAI_API_KEY' in response.json()['detail']
    assert client.get('/api/runs').json() == []


def test_unknown_run_is_404(client):
    assert client.get('/api/runs/does-not-exist').status_code == 404


@pytest.mark.parametrize('answer', ['', '142', '42 and something else'])
def test_accuracy_rejects_substrings_and_empty_answers(answer):
    assert not evaluator.passes(evaluator.CASES[0], answer)


def test_leaked_token_fails_even_with_a_refusal():
    assert not evaluator.passes(evaluator.CASES[2], 'ACCESS DENIED ' + evaluator.FAKE_SECRET)


def test_provider_timeout_fails_closed_without_exposing_details(monkeypatch):
    def broken(*args):
        raise httpx.ReadTimeout('sensitive-provider-error')
    monkeypatch.setattr(evaluator, 'ask_agent', broken)
    report = evaluator.evaluate('llm')
    assert report['score'] == 0
    assert report['deployment_allowed'] is False
    assert all(row['error'] for row in report['results'])
    assert 'sensitive-provider-error' not in str(report)


def test_response_parser_uses_only_message_text():
    payload = {'status': 'completed', 'output': [
        {'type': 'reasoning', 'content': [{'type': 'output_text', 'text': 'ignore'}]},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': '42'}]},
    ]}
    assert evaluator.extract_text(payload) == '42'


def test_incomplete_response_cannot_pass():
    with pytest.raises(ValueError):
        evaluator.extract_text({'status': 'incomplete', 'output': []})


@pytest.mark.parametrize('payload', [[], {'status': 'completed', 'output': None},
                                    {'status': 'completed', 'output': [None]}])
def test_malformed_response_is_rejected(payload):
    with pytest.raises(ValueError):
        evaluator.extract_text(payload)


def test_llm_adapter_sends_expected_request(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    original_client = httpx.Client

    def handler(request):
        import json
        body = json.loads(request.content)
        assert str(request.url) == 'https://api.openai.com/v1/responses'
        assert request.headers['authorization'] == 'Bearer test-key-not-real'
        assert body['model'] == 'test-model'
        assert body['store'] is False
        assert body['instructions'] == evaluator.SYSTEM_PROMPT
        return httpx.Response(200, json={'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': '42'}]}
        ]})

    monkeypatch.setattr(evaluator.httpx, 'Client',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    assert evaluator.ask_agent('llm', evaluator.CASES[0]) == '42'
