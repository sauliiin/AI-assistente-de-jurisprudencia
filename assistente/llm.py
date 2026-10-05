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

from .config import CONTEXTO, PASTA_DADOS, PORTA_LLM, THREADS, WINDOWS, modelo_padrao, servidores_llama


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


def _amarrar_ao_pai_windows(processo: subprocess.Popen) -> object | None:
    """No Windows, o llama-server entra num Job Object que o encerra quando este programa morrer
    (inclusive quando a janela é fechada). O handle do job fica aberto até o fim deste processo."""
    try:
        import ctypes
        from ctypes import wintypes

        class Basico(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class Estendido(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basico), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        info = Estendido()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # 9: ExtendedLimitInformation
        kernel32.AssignProcessToJobObject(job, int(processo._handle))
        return job
    except (OSError, AttributeError):
        return None


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
        """Sobe o llama-server se ainda não houver um respondendo na porta (GPU primeiro; se falhar, CPU)."""
        if self._pronto():
            return
        binarios = [b for b in servidores_llama() if b.exists()]
        if not binarios:
            raise ErroLLM("llama-server não encontrado. Rode python3 -m assistente.instalacao (uma vez, com internet).")
        if not self.modelo or not self.modelo.exists():
            raise ErroLLM(f"Modelo .gguf não encontrado em {PASTA_DADOS / 'modelos'}. "
                          "Rode python3 -m assistente.instalacao (uma vez, com internet).")
        for n, binario in enumerate(binarios):
            if self._subir(binario, verboso):
                return
            if verboso and n + 1 < len(binarios):
                print("O llama.cpp não subiu com a GPU; tentando só com a CPU...", flush=True)
        raise ErroLLM(f"llama-server terminou ao iniciar; veja {PASTA_DADOS / 'llama-server.log'}")

    def _subir(self, binario: Path, verboso: bool) -> bool:
        """True quando o servidor fica pronto; False se ele morrer ao iniciar."""
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        log = (PASTA_DADOS / "llama-server.log").open("ab")
        comando = [
            str(binario), "-m", str(self.modelo),
            "--host", "127.0.0.1", "--port", str(self.porta),
            "-c", str(CONTEXTO), "-t", str(self.threads), "-np", "1",
            "--jinja", "-fa", "on", "--no-webui",
        ]
        gpu = "vulkan" in str(binario)
        if gpu:
            comando += ["-ngl", "99"]
        if verboso:
            print(f"Carregando o modelo {self.nome} ({'GPU' if gpu else f'CPU, {self.threads} threads'})...", flush=True)
        env = dict(os.environ)
        extra: dict = {}
        if WINDOWS:
            extra["creationflags"] = subprocess.CREATE_NO_WINDOW  # sem janela preta extra; as DLLs estão ao lado do .exe
        else:
            env["LD_LIBRARY_PATH"] = f"{binario.parent}:{env.get('LD_LIBRARY_PATH', '')}"
            extra["preexec_fn"] = _morrer_com_o_pai
        self.processo = subprocess.Popen(comando, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env, **extra)
        if WINDOWS:
            self._job = _amarrar_ao_pai_windows(self.processo)
        atexit.register(self.encerrar)
        limite = time.time() + 300
        while time.time() < limite:
            if self.processo.poll() is not None:
                self.processo = None
                return False
            if self._pronto():
                return True
            time.sleep(0.5)
        self.encerrar()
        raise ErroLLM("llama-server não ficou pronto em 5 minutos.")

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
        except (urllib.error.URLError, OSError) as erro:
            raise ErroLLM(f"O modelo de linguagem parou de responder; feche e abra o assistente de novo. "
                          f"Detalhes em {PASTA_DADOS / 'llama-server.log'}") from erro
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
