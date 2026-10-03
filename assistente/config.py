"""Onde ficam o acervo, o modelo de linguagem, o llama.cpp e o cache do índice (fora do Git)."""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = Path(os.environ.get("ASSISTENTE_DADOS", Path.home() / ".local" / "share" / "assistente-de-jurisprudencia"))
# Acervo (votos, pareceres e a rede neural do site): o repositório jurisprudencia-juntas,
# por padrão ao lado deste. scripts/instalar.sh clona se não existir.
ACERVO = Path(os.environ.get("ASSISTENTE_ACERVO", RAIZ.parent / "jurisprudencia-juntas"))
PASTA_MODELOS = PASTA_DADOS / "modelos"
PORTA_LLM = int(os.environ.get("ASSISTENTE_PORTA_LLM", "8766"))
PORTA_WEB = int(os.environ.get("ASSISTENTE_PORTA_WEB", "8765"))
CONTEXTO = int(os.environ.get("ASSISTENTE_CONTEXTO", "12288"))
# Threads: medido com llama-bench no i5-1235U (2 núcleos P + 8 E); ajuste com ASSISTENTE_THREADS.
THREADS = int(os.environ.get("ASSISTENTE_THREADS", "0")) or None


# A GPU integrada (Vulkan) lê o contexto ~3x mais rápido que a CPU no i5-1235U; ASSISTENTE_GPU=0 desliga.
USAR_GPU = os.environ.get("ASSISTENTE_GPU", "1") != "0"


def llama_server() -> Path | None:
    """Binário do llama-server: variável de ambiente, build Vulkan, build CPU ou PATH."""
    if os.environ.get("ASSISTENTE_LLAMA_SERVER"):
        return Path(os.environ["ASSISTENTE_LLAMA_SERVER"])
    for pasta in (["bin-vulkan"] if USAR_GPU else []) + ["bin"]:
        candidatos = sorted((PASTA_DADOS / pasta).glob("*/llama-server"), reverse=True)
        if candidatos:
            return candidatos[0]
    for pasta in os.environ.get("PATH", "").split(os.pathsep):
        caminho = Path(pasta) / "llama-server"
        if caminho.exists():
            return caminho
    return None


def modelo_padrao() -> Path | None:
    """Modelo .gguf: ASSISTENTE_MODELO, o nome gravado em modelos/padrao.txt, ou o primeiro da pasta."""
    if os.environ.get("ASSISTENTE_MODELO"):
        return Path(os.environ["ASSISTENTE_MODELO"])
    escolhido = PASTA_MODELOS / "padrao.txt"
    if escolhido.exists() and (PASTA_MODELOS / escolhido.read_text().strip()).exists():
        return PASTA_MODELOS / escolhido.read_text().strip()
    modelos = sorted(m for m in PASTA_MODELOS.glob("*.gguf") if not m.name.startswith("mmproj"))
    return modelos[0] if modelos else None


def resolver_modelo(nome: str) -> Path:
    """Caminho de um .gguf, ou o da pasta de modelos cujo nome contém `nome` (ex.: "4b")."""
    caminho = Path(nome).expanduser()
    if caminho.exists():
        return caminho
    achados = [m for m in PASTA_MODELOS.glob("*.gguf") if nome.lower() in m.name.lower()]
    if len(achados) != 1:
        disponiveis = ", ".join(m.name for m in PASTA_MODELOS.glob("*.gguf")) or "nenhum"
        raise SystemExit(f"Modelo '{nome}' não encontrado (ou ambíguo). Disponíveis: {disponiveis}")
    return achados[0]
