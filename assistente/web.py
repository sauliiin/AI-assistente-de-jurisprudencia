"""Página local do assistente (só 127.0.0.1), com a resposta chegando aos poucos (SSE).

A página abre na hora; o assistente é preparado em segundo plano (na 1ª vez, baixa acervo,
llama.cpp e modelo) e a página mostra o progresso até ficar pronto.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import PASTA_DADOS, PORTA_WEB
from .llm import ErroLLM
from .respondedor import Assistente, Conversa

PAGINA = Path(__file__).with_name("pagina.html")
APP = "assistente-de-jurisprudencia"
# Sem a página aberta por este tempo (ela consulta /api/status a cada 30 s), o app encerra e libera a memória do modelo.
SEM_PAGINA_S = 180

Preparar = Callable[[Callable[[str, float | None], None]], Assistente]


def _ja_aberto(porta: int) -> bool:
    """Outra cópia deste app já responde na porta?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{porta}/api/status", timeout=1.5) as resp:
            return json.loads(resp.read()).get("app") == APP
    except (OSError, ValueError):
        return False


def servir(preparar: Preparar, porta: int | None = None, abrir: bool = True, encerrar_sem_pagina: bool = False) -> None:
    porta = porta or PORTA_WEB
    if _ja_aberto(porta):
        print(f"O assistente já está aberto em http://127.0.0.1:{porta}/", flush=True)
        if abrir:
            webbrowser.open(f"http://127.0.0.1:{porta}/")
        return

    conversas: dict[str, Conversa] = {}
    vez = threading.Lock()  # um único modelo: uma resposta por vez
    estado: dict = {"assistente": None, "etapa": "Iniciando…", "progresso": None, "erro": None, "preparando": False}
    contato = [time.time()]
    encerrando = threading.Event()

    def progresso(etapa: str, fracao: float | None) -> None:
        estado["etapa"], estado["progresso"] = etapa, fracao

    def preparar_em_segundo_plano() -> None:
        if estado["preparando"] or estado["assistente"]:
            return
        estado.update(preparando=True, erro=None)
        try:
            estado["assistente"] = preparar(progresso)
            print("Pronto.", flush=True)
            # No Linux o llama-server morre junto com a thread que o criou (PR_SET_PDEATHSIG): esta fica viva.
            encerrando.wait()
        except Exception as erro:  # mostrado na página, com o detalhe no log
            estado["erro"] = str(erro) or erro.__class__.__name__
            print(f"Erro: {estado['erro']}", flush=True)
            with (PASTA_DADOS / "assistente.log").open("a", encoding="utf-8") as log:
                log.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{traceback.format_exc()}\n")
        finally:
            estado["preparando"] = False

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
            contato[0] = time.time()
            if self.path in ("/", "/index.html"):
                corpo = PAGINA.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
            elif self.path == "/api/status":
                assistente: Assistente | None = estado["assistente"]
                if assistente is None:
                    self._json({"app": APP, "pronto": False, "etapa": estado["etapa"], "progresso": estado["progresso"],
                                "erro": estado["erro"]})
                    return
                self._json({
                    "app": APP,
                    "pronto": True,
                    "modelo": assistente.modelo.nome,
                    "decisoes": len(assistente.acervo.decisoes),
                    "entendimento": len(assistente.acervo.entendimento),
                    "artigos": len(assistente.acervo.legislacao),
                    "pareceres": len({p.file_id for p in assistente.acervo.pareceres}),
                    "ocupado": vez.locked(),
                })
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            contato[0] = time.time()
            if self.path == "/api/encerrar":
                self._json({"ok": True})
                threading.Thread(target=servidor.shutdown, daemon=True).start()
                return
            if self.path == "/api/preparar":  # "tentar de novo" depois de um erro
                threading.Thread(target=preparar_em_segundo_plano, daemon=True).start()
                self._json({"ok": True})
                return
            if self.path != "/api/perguntar":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            assistente: Assistente | None = estado["assistente"]
            if assistente is None:
                self._json({"erro": "o assistente ainda está sendo preparado"}, 503)
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
                        contato[0] = time.time()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ErroLLM as erro:
                enviar({"tipo": "erro", "mensagem": str(erro)})

    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    url = f"http://127.0.0.1:{porta}/"
    print(f"Assistente em {url}", flush=True)
    print("Para encerrar: botão Encerrar na página, Ctrl+C ou fechar esta janela.", flush=True)
    threading.Thread(target=preparar_em_segundo_plano, daemon=True).start()

    if encerrar_sem_pagina:
        def vigiar() -> None:
            while True:
                time.sleep(15)
                if time.time() - contato[0] > SEM_PAGINA_S and not vez.locked():
                    print("Nenhuma página aberta há alguns minutos; encerrando.", flush=True)
                    servidor.shutdown()
                    return

        threading.Thread(target=vigiar, daemon=True).start()
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
        encerrando.set()
        servidor.server_close()
        if estado["assistente"]:
            estado["assistente"].modelo.encerrar()
