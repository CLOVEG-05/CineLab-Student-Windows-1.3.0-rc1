import socket, urllib.request, json, os

base = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
print("LLM_BASE_URL =", base)
try:
    print("resolve host.docker.internal ->", socket.gethostbyname("host.docker.internal"))
except Exception as e:
    print("resolve FAIL", repr(e))

try:
    r = urllib.request.urlopen(base.rstrip("/") + "/models", timeout=8)
    j = json.loads(r.read())
    print("models OK:", [m["id"] for m in j.get("data", [])][:3])
except Exception as e:
    print("models FAIL", repr(e))

body = json.dumps({
    "model": os.environ.get("LLM_MODEL", "qwen2.5-7b-instruct-1m"),
    "messages": [{"role": "user", "content": "用一句话介绍电影《肖申克的救赎》。"}],
    "temperature": 0.3, "max_tokens": 60}).encode("utf-8")
try:
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "lm-studio")})
    with urllib.request.urlopen(req, timeout=60) as resp:
        j = json.loads(resp.read())
    print("chat OK:", j["choices"][0]["message"]["content"][:80])
except Exception as e:
    print("chat FAIL", repr(e))
