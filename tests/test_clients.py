from clients import generate_text, get_embedding, PROVIDERS

def test_generate_text(monkeypatch):
    called = {}
    def fake_request(url, headers, payload):
        called['url'] = url
        called['headers'] = headers
        called['payload'] = payload
        return {"choices": [{"message": {"content": "hi"}}]}
    monkeypatch.setattr('clients._request', fake_request)
    cfg = {
        'provider': 'openai',
        'base_url': 'https://api.openai.com',
        'model': 'gpt-test',
        'api_key': 'k',
        'embedding_model': 'embed'
    }
    res = generate_text('hello', cfg)
    assert res == 'hi'
    assert called['url'] == 'https://api.openai.com' + PROVIDERS['openai']['chat_path']
    assert called['headers']['Authorization'] == 'Bearer k'
    assert called['payload']['messages'][0]['content'] == 'hello'

def test_get_embedding(monkeypatch):
    def fake_request(url, headers, payload):
        return {"data": [{"embedding": [0.1, 0.2]}]}
    monkeypatch.setattr('clients._request', fake_request)
    cfg = {
        'provider': 'openai',
        'base_url': 'https://api.openai.com',
        'model': 'gpt-test',
        'api_key': 'k',
        'embedding_model': 'embed'
    }
    emb = get_embedding('text', cfg)
    assert emb == [0.1, 0.2]
