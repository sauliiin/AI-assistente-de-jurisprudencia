"""
Baixa o que o assistente precisa e ainda não está nesta máquina (a única etapa que usa internet):

  - o acervo (só site_data/ do repositório jurisprudencia-juntas), quando não há um clone do Git;
    nas aberturas seguintes, baixa de novo só os arquivos que mudaram (ETag);
  - o llama.cpp, binários prontos: Vulkan (GPU) e CPU como reserva;
  - o modelo de linguagem Qwen3.5 (GGUF de 4 bits): 9B, ou 4B se a máquina tiver pouca memória.

Sem internet, segue com o que já tem; só falha se faltar algo.

    python3 -m assistente.instalacao [9b|4b]
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

from .config import ACERVO, EXECUTAVEL_LLAMA, PASTA_DADOS, PASTA_MODELOS, WINDOWS, modelo_padrao

VERSAO_LLAMA = "b11379"
MODELOS = {
    "4b": ("Qwen3.5-4B-Q4_K_M.gguf", "unsloth/Qwen3.5-4B-GGUF"),
    "9b": ("Qwen3.5-9B-Q4_K_M.gguf", "unsloth/Qwen3.5-9B-GGUF"),
}
ACERVO_URL = "https://raw.githubusercontent.com/sauliiin/jurisprudencia-juntas/main/"
ARQUIVOS_ACERVO = [
    "site_data/votos.jsonl",
    "site_data/pareceres.jsonl",
    "site_data/legislacao.jsonl",
    "site_data/ia/modelo.json",
    "site_data/ia/termos.bin",
    "site_data/ia/docs.bin",
]
AGENTE = {"User-Agent": "assistente-de-jurisprudencia"}

# progresso(etapa, fração de 0 a 1 ou None quando não se sabe)
Progresso = Callable[[str, float | None], None]


class ErroInstalacao(RuntimeError):
    pass


def _no_terminal(etapa: str, fracao: float | None) -> None:
    sufixo = f" {100 * fracao:5.1f}%" if fracao is not None else ""
    print(f"\r{etapa}{sufixo}   ", end="" if fracao is not None and fracao < 1 else "\n", flush=True)


def _abrir(url: str, cabecalhos: dict | None = None, timeout: float = 30):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**AGENTE, **(cabecalhos or {})}), timeout=timeout)


def _baixar(url: str, destino: Path, etapa: str, progresso: Progresso, continuar: bool = False) -> None:
    """Baixa para destino.part e renomeia no fim. continuar=True retoma um .part interrompido."""
    parcial = destino.with_name(destino.name + ".part")
    ja = parcial.stat().st_size if continuar and parcial.exists() else 0
    with _abrir(url, {"Range": f"bytes={ja}-"} if ja else None) as resp:
        if ja and resp.status != 206:  # servidor não retomou: recomeça
            ja = 0
        total = ja + int(resp.headers.get("Content-Length") or 0)
        feito, ultimo = ja, 0.0
        with parcial.open("ab" if ja else "wb") as fh:
            while bloco := resp.read(1 << 20):
                fh.write(bloco)
                feito += len(bloco)
                if time.time() - ultimo > 0.3:
                    progresso(etapa, feito / total if total else None)
                    ultimo = time.time()
    if total and feito < total:
        raise ErroInstalacao(f"download incompleto de {url}")
    os.replace(parcial, destino)
    progresso(etapa, 1.0)


# --- acervo ---------------------------------------------------------------------


def acervo_e_clone() -> bool:
    """Acervo mantido pelo Git (quem trabalha no código): não é tocado aqui."""
    return (ACERVO / ".git").exists()


def _atualizar_acervo(progresso: Progresso) -> None:
    marcas_arquivo = ACERVO / "etags.json"
    try:
        marcas = json.loads(marcas_arquivo.read_text())
    except (OSError, ValueError):
        marcas = {}
    for n, nome in enumerate(ARQUIVOS_ACERVO, 1):
        destino = ACERVO / nome
        etapa = f"Acervo: {nome.split('/')[-1]} ({n}/{len(ARQUIVOS_ACERVO)})"
        cabecalhos = {"Accept-Encoding": "gzip"}
        if destino.exists() and marcas.get(nome):
            cabecalhos["If-None-Match"] = marcas[nome]
        progresso(etapa, None)
        try:
            resp = _abrir(ACERVO_URL + nome, cabecalhos, timeout=15)
        except urllib.error.HTTPError as erro:
            if erro.code == 304:  # não mudou
                continue
            raise
        except (urllib.error.URLError, OSError):
            if all((ACERVO / a).exists() for a in ARQUIVOS_ACERVO):
                return  # sem internet: fica o acervo que já está aqui
            raise
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_name(destino.name + ".part")
        # O GitHub manda os .jsonl comprimidos (gzip): ~5x menos a baixar.
        descompactar = zlib.decompressobj(16 + zlib.MAX_WBITS) if resp.headers.get("Content-Encoding") == "gzip" else None
        total, feito = int(resp.headers.get("Content-Length") or 0), 0
        with resp, parcial.open("wb") as fh:
            while bloco := resp.read(1 << 20):
                feito += len(bloco)
                fh.write(descompactar.decompress(bloco) if descompactar else bloco)
                progresso(etapa, feito / total if total else None)
            if descompactar:
                fh.write(descompactar.flush())
        os.replace(parcial, destino)
        if resp.headers.get("ETag"):
            marcas[nome] = resp.headers["ETag"]
            marcas_arquivo.write_text(json.dumps(marcas, indent=1))


# --- llama.cpp ------------------------------------------------------------------


def _pacotes_llama() -> list[tuple[str, str]]:
    """[(pasta, arquivo do release)]: Vulkan (GPU) e CPU."""
    if WINDOWS:
        return [("bin-vulkan", f"llama-{VERSAO_LLAMA}-bin-win-vulkan-x64.zip"), ("bin", f"llama-{VERSAO_LLAMA}-bin-win-cpu-x64.zip")]
    return [("bin-vulkan", f"llama-{VERSAO_LLAMA}-bin-ubuntu-vulkan-x64.tar.gz"), ("bin", f"llama-{VERSAO_LLAMA}-bin-ubuntu-x64.tar.gz")]


def _instalar_llama(progresso: Progresso) -> None:
    for pasta, arquivo in _pacotes_llama():
        if any((PASTA_DADOS / pasta).rglob(EXECUTAVEL_LLAMA)):  # já instalado (esta ou outra versão)
            continue
        destino = PASTA_DADOS / pasta / VERSAO_LLAMA
        pacote = PASTA_DADOS / arquivo
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSAO_LLAMA}/{arquivo}"
        _baixar(url, pacote, f"llama.cpp ({'GPU' if 'vulkan' in arquivo else 'CPU'})", progresso)
        temporario = destino.with_name(destino.name + ".tmp")
        shutil.rmtree(temporario, ignore_errors=True)
        if arquivo.endswith(".zip"):
            with zipfile.ZipFile(pacote) as z:
                z.extractall(temporario)
        else:
            with tarfile.open(pacote) as t:
                t.extractall(temporario, filter="data")
        shutil.rmtree(destino, ignore_errors=True)
        os.replace(temporario, destino)
        pacote.unlink()
    _runtime_visual_c()


def _runtime_visual_c() -> None:
    """O llama.cpp do Windows precisa do runtime do Visual C++ (MSVCP140.dll...), que nem todo Windows tem.

    O .exe leva uma cópia (empacotamento/empacotar.py), posta aqui ao lado do llama-server.exe.
    """
    origem = Path(getattr(sys, "_MEIPASS", "")) / "vcruntime"
    if not WINDOWS or not origem.is_dir():
        return
    for pasta in ("bin-vulkan", "bin"):
        for servidor in (PASTA_DADOS / pasta).rglob(EXECUTAVEL_LLAMA):
            for dll in origem.glob("*.dll"):
                if not (servidor.parent / dll.name).exists():
                    shutil.copy2(dll, servidor.parent / dll.name)


# --- modelo ---------------------------------------------------------------------


def memoria_total_gb() -> float:
    try:
        if WINDOWS:
            class Estado(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("resto", ctypes.c_ulonglong * 6)]
            estado = Estado(dwLength=ctypes.sizeof(Estado))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
            return estado.ullTotalPhys / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError, AttributeError):
        return 16.0


def tamanho_recomendado() -> str:
    """O 9B usa ~7 GB de memória; com menos de 12 GB no total, o 4B (~3,5 GB) responde sem travar a máquina."""
    return os.environ.get("ASSISTENTE_TAMANHO") or ("9b" if memoria_total_gb() >= 12 else "4b")


def _instalar_modelo(tamanho: str, progresso: Progresso) -> None:
    arquivo, repo = MODELOS[tamanho]
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_MODELOS / arquivo
    if not destino.exists():
        livre = shutil.disk_usage(PASTA_MODELOS).free / 2**30
        precisa = {"9b": 6.0, "4b": 3.0}[tamanho]
        if livre < precisa:
            raise ErroInstalacao(f"Espaço insuficiente em {PASTA_MODELOS}: o modelo precisa de {precisa:.0f} GB "
                                 f"e há {livre:.1f} GB livres.")
        _baixar(f"https://huggingface.co/{repo}/resolve/main/{arquivo}", destino,
                f"Modelo de linguagem {tamanho.upper()} (só na primeira vez)", progresso, continuar=True)
    (PASTA_MODELOS / "padrao.txt").write_text(arquivo)


# --- tudo -----------------------------------------------------------------------


def instalar(tamanho: str | None = None, progresso: Progresso = _no_terminal) -> None:
    """Deixa tudo pronto. tamanho=None: mantém o modelo já instalado ou escolhe pela memória da máquina."""
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    try:
        if not acervo_e_clone():
            _atualizar_acervo(progresso)
        _instalar_llama(progresso)
        if tamanho or not modelo_padrao():
            _instalar_modelo(tamanho or tamanho_recomendado(), progresso)
    except urllib.error.HTTPError as erro:
        raise ErroInstalacao(f"Falha ao baixar {erro.url}: {erro}") from erro
    except urllib.error.URLError as erro:
        raise ErroInstalacao("Sem internet. Na primeira vez o assistente precisa baixar o acervo, o llama.cpp "
                             "e o modelo (depois funciona offline). Tente de novo com a internet ligada.") from erro
    except OSError as erro:  # conexão caiu no meio, disco cheio, sem permissão
        raise ErroInstalacao(f"Falha ao baixar ou gravar ({erro}). Abra de novo para continuar de onde parou.") from erro


def main() -> None:
    tamanho = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if tamanho and tamanho not in MODELOS:
        raise SystemExit("Uso: python3 -m assistente.instalacao [9b|4b]")
    try:
        instalar(tamanho)
    except ErroInstalacao as erro:
        raise SystemExit(f"Erro: {erro}") from erro
    print(f"Pronto em {PASTA_DADOS}. A partir daqui funciona sem internet.")


if __name__ == "__main__":
    main()
