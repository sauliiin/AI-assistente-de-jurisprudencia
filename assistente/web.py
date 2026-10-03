"""Página local do assistente (só 127.0.0.1), com a resposta chegando aos poucos (SSE)."""

from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import PORTA_WEB
from .llm import ErroLLM
from .respondedor import Assistente, Conversa

PAGINA = Path(__file__).with_name("pagina.html")


def servir(assistente: Assistente, porta: int | None = None, abrir: bool = True) -> None:
    porta = porta or PORTA_WEB
    conversas: dict[str, Conversa] = {}
    vez = threading.Lock()  # um único modelo: uma resposta por vez

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def _json(self, dados: object, status: int = 200) -> None:
            corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                corpo = PAGINA.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
            elif self.path == "/api/status":
                self._json({
                    "modelo": assistente.modelo.nome,
                    "decisoes": len(assistente.acervo.decisoes),
                    "pareceres": len({p.file_id for p in assistente.acervo.pareceres}),
                    "ocupado": vez.locked(),
                })
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/perguntar":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                pedido = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                self._json({"erro": "JSON inválido"}, 400)
                return
            pergunta = (pedido.get("pergunta") or "").strip()[:2000]
            if not pergunta:
                self._json({"erro": "pergunta vazia"}, 400)
                return
            cid = pedido.get("conversa") or uuid.uuid4().hex
            conversa = conversas.setdefault(cid, Conversa())

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def enviar(evento: dict) -> None:
                self.wfile.write(f"data: {json.dumps(evento, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                enviar({"tipo": "conversa", "id": cid, "fila": vez.locked()})
                with vez:
                    for evento in assistente.responder(pergunta, conversa, pensar=bool(pedido.get("pensar"))):
                        enviar(evento)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ErroLLM as erro:
                enviar({"tipo": "erro", "mensagem": str(erro)})

    assistente.modelo.garantir()
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    url = f"http://127.0.0.1:{porta}/"
    print(f"Assistente em {url} (Ctrl+C encerra)", flush=True)
    if abrir:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        servidor.server_close()
