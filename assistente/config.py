"""Onde ficam o acervo, o modelo de linguagem, o llama.cpp e o cache do índice (fora do Git)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Executável empacotado (.exe ou AppImage, feito com PyInstaller): tudo o que falta é baixado na 1ª abertura.
EMPACOTADO = bool(getattr(sys, "frozen", False))
WINDOWS = sys.platform == "win32"

RAIZ = Path(__file__).resolve().parent.parent


def _pasta_dados_padrao() -> Path:
    if WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "assistente-de-jurisprudencia"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "assistente-de-jurisprudencia"


PASTA_DADOS = Path(os.environ.get("ASSISTENTE_DADOS") or _pasta_dados_padrao())


def _acervo_padrao() -> Path:
    # Quem trabalha no código tem o repositório jurisprudencia-juntas ao lado deste (scripts/instalar.sh clona);
    # o executável baixa só o site_data do acervo para a pasta de dados (assistente/instalacao.py).
    vizinho = RAIZ.parent / "jurisprudencia-juntas"
    return vizinho if not EMPACOTADO and (vizinho / "site_data").exists() else PASTA_DADOS / "acervo"


ACERVO = Path(os.environ.get("ASSISTENTE_ACERVO") or _acervo_padrao())
# Entendimento das Juntas (Vade Mecum de jurisprudência administrativa), a primeira fonte consultada.
# Por padrão vem do acervo ("ENTENDIMENTO JUNTAS 2024.docx", entre os pareceres); ASSISTENTE_ENTENDIMENTO
# aponta para um .docx local, por exemplo uma versão mais nova ainda não publicada.
ENTENDIMENTO = Path(os.environ["ASSISTENTE_ENTENDIMENTO"]) if os.environ.get("ASSISTENTE_ENTENDIMENTO") else None
PASTA_MODELOS = PASTA_DADOS / "modelos"
PORTA_LLM = int(os.environ.get("ASSISTENTE_PORTA_LLM", "8766"))
PORTA_WEB = int(os.environ.get("ASSISTENTE_PORTA_WEB", "8765"))
CONTEXTO = int(os.environ.get("ASSISTENTE_CONTEXTO", "12288"))
# Threads: medido com llama-bench no i5-1235U (2 núcleos P + 8 E); ajuste com ASSISTENTE_THREADS.
THREADS = int(os.environ.get("ASSISTENTE_THREADS", "0")) or None


# A GPU integrada (Vulkan) lê o contexto ~3x mais rápido que a CPU no i5-1235U; ASSISTENTE_GPU=0 desliga.
USAR_GPU = os.environ.get("ASSISTENTE_GPU", "1") != "0"
EXECUTAVEL_LLAMA = "llama-server.exe" if WINDOWS else "llama-server"


def servidores_llama() -> list[Path]:
    """Binários do llama-server, na ordem de tentativa: variável de ambiente, build Vulkan, build CPU, PATH.

    Se o Vulkan não subir (sem driver de GPU), o seguinte da lista é usado.
    """
    if os.environ.get("ASSISTENTE_LLAMA_SERVER"):
        return [Path(os.environ["ASSISTENTE_LLAMA_SERVER"])]
    saida = []
    for pasta in (["bin-vulkan"] if USAR_GPU else []) + ["bin"]:
        saida += sorted((PASTA_DADOS / pasta).rglob(EXECUTAVEL_LLAMA), reverse=True)[:1]
    for pasta in os.environ.get("PATH", "").split(os.pathsep):
        caminho = Path(pasta) / EXECUTAVEL_LLAMA
        if pasta and caminho.exists():
            saida.append(caminho)
            break
    return saida


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
