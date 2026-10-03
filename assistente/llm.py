"""
Modelo de linguagem local, via llama.cpp (llama-server), sem internet.

O servidor escuta só em 127.0.0.1. Se já houver um rodando na porta, ele é
reaproveitado; senão é iniciado aqui e encerrado quando o programa termina.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from .config import CONTEXTO, PASTA_DADOS, PORTA_LLM, THREADS, llama_server, modelo_padrao


class ErroLLM(RuntimeError):
    pass


def _morrer_com_o_pai() -> None:
    """No Linux, o llama-server recebe SIGTERM se este programa morrer (inclusive por kill)."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _threads_padrao() -> int:
    # llama.cpp rende mais com um thread por núcleo físico do que com todos os threads lógicos.
    total = os.cpu_count() or 4
    return max(2, total - 2) if total <= 8 else max(4, total * 5 // 6)


class ModeloLocal:
    def __init__(self, modelo: Path | None = None, porta: int = PORTA_LLM, threads: int | None = THREADS) -> None:
        self.modelo = Path(modelo) if modelo else modelo_padrao()
        self.porta = porta
        self.threads = threads or _threads_padrao()
        self.base = f"http://127.0.0.1:{porta}"
        self.processo: subprocess.Popen | None = None

    @property
    def nome(self) -> str:
        return self.modelo.stem if self.modelo else "?"

    def _pronto(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def garantir(self, verboso: bool = True) -> None:
        """Sobe o llama-server se ainda não houver um respondendo na porta."""
        if self._pronto():
            return
        binario = llama_server()
        if not binario or not binario.exists():
            raise ErroLLM("llama-server não encontrado. Rode scripts/instalar.sh (uma vez, com internet).")
        if not self.modelo or not self.modelo.exists():
            raise ErroLLM(f"Modelo .gguf não encontrado em {PASTA_DADOS / 'modelos'}. Rode scripts/instalar.sh.")
        log = (PASTA_DADOS / "llama-server.log").open("ab")
        comando = [
            str(binario), "-m", str(self.modelo),
            "--host", "127.0.0.1", "--port", str(self.porta),
            "-c", str(CONTEXTO), "-t", str(self.threads), "-np", "1",
            "--jinja", "-fa", "on", "--no-webui",
        ]
        if "vulkan" in str(binario):
            comando += ["-ngl", "99"]
        if verboso:
            print(f"Carregando o modelo {self.nome} ({self.threads} threads)...", flush=True)
        env = {**os.environ, "LD_LIBRARY_PATH": f"{binario.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        self.processo = subprocess.Popen(
            comando, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=_morrer_com_o_pai
        )
        atexit.register(self.encerrar)
        limite = time.time() + 180
        while time.time() < limite:
            if self.processo.poll() is not None:
                raise ErroLLM(f"llama-server terminou ao iniciar; veja {PASTA_DADOS / 'llama-server.log'}")
            if self._pronto():
                return
            time.sleep(0.5)
        raise ErroLLM("llama-server não ficou pronto em 3 minutos.")

    def encerrar(self) -> None:
        if self.processo and self.processo.poll() is None:
            self.processo.terminate()
            try:
                self.processo.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.processo.kill()
        self.processo = None

    def conversar(
        self,
        mensagens: list[dict],
        max_tokens: int = 900,
        temperatura: float = 0.2,
        pensar: bool = False,
        estatisticas: dict | None = None,
    ) -> Iterator[str]:
        """Gera a resposta aos pedaços (streaming). Preenche `estatisticas` com tempos e tokens."""
        corpo = {
            "messages": mensagens,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperatura,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": pensar},
            "timings_per_token": False,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(corpo).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as erro:
            raise ErroLLM(f"llama-server respondeu {erro.code}: {erro.read()[:500]!r}") from erro
        with resp:
            for linha in resp:
                linha = linha.strip()
                if not linha.startswith(b"data:"):
                    continue
                dados = linha[5:].strip()
                if dados == b"[DONE]":
                    break
                evento = json.loads(dados)
                if estatisticas is not None and evento.get("timings"):
                    estatisticas.update(evento["timings"])
                for escolha in evento.get("choices", []):
                    pedaco = (escolha.get("delta") or {}).get("content")
                    if pedaco:
                        yield pedaco
