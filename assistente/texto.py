"""
Normalização de texto compartilhada com o repositório do acervo.

Estas funções são cópia das de `treinar_ia.py` (repositório jurisprudencia-juntas),
que treina a rede neural usada na busca. O tokenizador precisa ser idêntico ao
do treino; tests/test_paridade.py confere isso quando o acervo está disponível.
"""

from __future__ import annotations

import re
import unicodedata

# Minúsculas, sem acentos, caracteres não-ASCII descartados, só sequências de 3+ letras.
_TOKEN = re.compile(r"[a-z]{3,}")
_NAO_ASCII = re.compile(r"[^\x00-\x7f]")
AUTO = re.compile(r"\b\d{8,14}[A-Z]{2}\b")
FISCAL = re.compile(
    r"considera[çc][õo]es fiscais\s*:?\s*(.+?)(?:Em sua defesa|FUNDAMENTA[ÇC][ÃA]O|$)", re.I | re.S
)

RESULTADOS = ["", "indeferido", "deferido", "parcialmente deferido", "não conhecido", "extinto", "diligência"]
_PADROES_RESULTADO = [
    (5, re.compile(r"EXTINT[OA]|PERDA\s+DO\s+OBJETO")),
    (4, re.compile(r"NAO\s+(?:SE\s+)?CONHEC")),
    (3, re.compile(r"PARCIALMENTE\s+DEFERID|DEFIRO\s+PARCIALMENTE|DEFIRO\s+EM\s+PARTE|PARCIAL\s+PROVIMENTO|DEFERIMENTO\s+PARCIAL")),
    (1, re.compile(r"INDEFIR|INDEFERID|NEGO\s+PROVIMENTO|NEGAR\s+PROVIMENTO|IMPROCEDENTE")),
    (2, re.compile(r"\bDEFIR|\bDEFERID|DOU\s+PROVIMENTO|DAR\s+PROVIMENTO|\bPROCEDENTE")),
    (6, re.compile(r"DILIGENCIA")),
]


def normalizar(texto: str) -> str:
    return _NAO_ASCII.sub("", unicodedata.normalize("NFD", texto.lower()))


def tokenizar(texto: str) -> list[str]:
    return _TOKEN.findall(normalizar(texto))


def chave_infracao(infracao: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalizar(infracao or ""))).strip()[:160]


def resultado_da_decisao(texto: str) -> int:
    """Código em RESULTADOS, lido do trecho 'Dispositivo da decisão' (última ocorrência)."""
    plano = normalizar(texto).upper()
    pos = plano.rfind("DISPOSITIVO DA DECIS")
    if pos < 0:
        return 0
    trecho = plano[pos : pos + 700]
    for codigo, padrao in _PADROES_RESULTADO:
        if padrao.search(trecho):
            return codigo
    return 0
