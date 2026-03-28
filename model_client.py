"""
Client for the persistent model server.
Sends JSON commands and receives responses.
"""

import socket
import json


class ModelClient:
    def __init__(self, host="localhost", port=9999):
        self.host = host
        self.port = port

    def _send(self, cmd):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall((json.dumps(cmd) + "\n").encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk: break
            data += chunk
            if b"\n" in data: break
        sock.close()
        return json.loads(data.decode().strip())

    def ping(self):
        return self._send({"action": "ping"})

    def learn(self, prompt, answer):
        return self._send({"action": "learn", "prompt": prompt, "answer": answer})

    def learn_batch(self, facts):
        return self._send({"action": "learn_batch", "facts": facts})

    def generate(self, prompt, max_tokens=30):
        r = self._send({"action": "generate", "prompt": prompt, "max_tokens": max_tokens})
        return r.get("response", "")

    def stats(self):
        return self._send({"action": "stats"})

    def clear(self):
        return self._send({"action": "clear"})

    def set_boost(self, value):
        return self._send({"action": "set_boost", "value": value})

    def set_threshold(self, value):
        return self._send({"action": "set_threshold", "value": value})


if __name__ == "__main__":
    c = ModelClient()
    print(c.ping())
