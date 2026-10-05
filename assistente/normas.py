"""
Fontes normativas, consultadas antes dos pareceres e das decisões:

  1. o ENTENDIMENTO DAS JUNTAS (Vade Mecum de jurisprudência administrativa,
     "ENTENDIMENTO JUNTAS 2024.docx"), dividido por tópico do sumário. Vem do
     acervo (pasta de pareceres do Drive) ou de um .docx local;
  2. a LEGISLAÇÃO do acervo (pasta "Legislação" do Drive, já extraída em
     site_data/legislacao.jsonl), dividida por artigo, sem a redação revogada.

Só biblioteca padrão: o .docx é lido direto do XML.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
import zipfile
from pathlib import Path

# --- Entendimento das Juntas --------------------------------------------------

_PARAGRAFO = re.compile(r"<w:p[ >].*?</w:p>", re.S)
_TEXTO_RUN = re.compile(r"<w:t(?: [^>]*)?>([^<]*)</w:t>")
_CAPITULO = re.compile(r"^C[ÁA]P[ÍI]TULO\s+([IVX]+)\s*[-–]?\s*(.*)$", re.I)
# "1.1 - Casa de shows", "2.1 Área pública...", "1.5– Notificação...". Incisos (I -) e "1)" não são tópicos.
_TOPICO = re.compile(r"^(\d+(?:\.\d+)*)\s*[-–]?\s*(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])")
_SUMARIO = re.compile(r"[-–]\s*Pg\.?\s*\d+\s*$", re.I)


def paragrafos_docx(caminho: Path) -> list[str]:
    with zipfile.ZipFile(caminho) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    saida = []
    for p in _PARAGRAFO.findall(xml):
        texto = html.unescape("".join(_TEXTO_RUN.findall(p)))
        texto = re.sub(r"\s+", " ", texto).strip()
        if texto:
            saida.append(texto)
    return saida


def paragrafos_texto(texto: str) -> list[str]:
    """Texto extraído pelo acervo: parágrafos separados por linha em branco."""
    return [p for p in (re.sub(r"\s+", " ", bloco).strip() for bloco in re.split(r"\n\s*\n", texto)) if p]


def topicos_entendimento(paragrafos: list[str]) -> list[dict]:
    """[{"titulo": "CAPÍTULO VI – TEMAS DE DIREITO › 1 - Dupla visita. › 1.1 - Mesas e cadeiras...", "texto": ...}].

    O sumário (linhas terminadas em "Pg. N") é pulado; a apresentação antes do
    primeiro capítulo vira um tópico próprio. O título leva os tópicos-pai, para
    que "1.1 - Mesas e cadeiras" seja achado também por "dupla visita".
    """
    topicos: list[dict] = []
    capitulo = "Apresentação"
    pais: dict[str, str] = {}
    atual = {"titulo": capitulo, "texto": []}

    def fechar() -> None:
        if atual["texto"]:
            topicos.append({"titulo": atual["titulo"], "texto": "\n\n".join(atual["texto"])})

    for p in paragrafos:
        if _SUMARIO.search(p) or p.upper() == "SUMÁRIO":
            continue
        cap = _CAPITULO.match(p)
        topico = _TOPICO.match(p)
        if cap:
            fechar()
            capitulo = f"CAPÍTULO {cap.group(1).upper()} – {cap.group(2).strip().upper()}".rstrip(" –")
            pais = {}
            atual = {"titulo": capitulo, "texto": []}
        elif topico:
            fechar()
            numero = topico.group(1)
            resto = p[topico.end():]
            # Título colado ao texto ("1- Alvará ... - O Alvará de Localização é..."): corta no traço seguinte.
            titulo, corpo = resto, ""
            if len(resto) > 200:
                m = re.search(r"\s[-–]\s", resto)
                titulo, corpo = (resto[: m.start()], resto[m.end():]) if m else (resto[:120], resto)
            pais[numero] = f"{numero} - {titulo.strip()}"
            caminho_pais = [pais[n] for n in pais if numero.startswith(n + ".")]
            atual = {"titulo": " › ".join([capitulo, *caminho_pais, pais[numero]]), "texto": [corpo] if corpo else []}
        else:
            atual["texto"].append(p)
    fechar()
    return topicos


# --- Legislação -----------------------------------------------------------------


def _sem_riscado(texto: str, riscado: list[list[int]], inicio: int, fim: int) -> str:
    """Trecho [inicio, fim) sem a redação revogada (riscada no original)."""
    partes, pos = [], inicio
    for a, b in sorted(riscado):
        if b <= pos or a >= fim:
            continue
        partes.append(texto[pos:max(a, pos)])
        pos = max(pos, b)
    partes.append(texto[pos:fim])
    linhas = [ln for ln in "".join(partes).split("\n") if not _e_nota(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(linhas)).strip()


# Nota de alteração ("Inciso I com redação dada pelo Decreto nº 16.278..."), não texto de lei.
_NOTA = re.compile(r"\b(com reda[çc][ãa]o dada|acrescentad[oa]|revogad[oa]) (pel[oa]|por)\b", re.I)
_DISPOSITIVO_LEI = re.compile(r"^\s*(Art\.?|§|[IVXLC]+|[a-z])\s*\S*\s*[-–)]")


def _e_nota(linha: str) -> bool:
    return bool(_NOTA.search(linha)) and "|" not in linha and not _DISPOSITIVO_LEI.match(linha)


def unidades_legislacao(caminho: Path) -> list[dict]:
    """Artigos (e itens dos anexos) vigentes: [{"norma", "rotulo", "contexto", "numero", "tipo", "texto", "link"}]."""
    saida = []
    with caminho.open(encoding="utf-8") as fh:
        for linha in fh:
            doc = json.loads(linha)
            texto, riscado = doc.get("texto") or "", doc.get("riscado") or []
            for u in doc.get("dispositivos") or []:
                if u.get("revogado") or u.get("tipo") not in ("artigo", "item"):
                    continue
                corpo = _sem_riscado(texto, riscado, u["inicio"], u["fim"])
                if len(corpo) < 30:
                    continue
                saida.append({
                    "norma": doc.get("categoria", ""),
                    "rotulo": u.get("rotulo", ""),
                    "contexto": u.get("contexto", ""),
                    "numero": u.get("numero", ""),
                    "tipo": u["tipo"],
                    "texto": corpo,
                    "link": doc.get("drive_view_url") or "",
                })
    return saida


def chave_norma(tipo: str, numero: str | int) -> tuple[str, int]:
    """('Lei', '9.725') -> ('lei', 9725). Cópia de legislacao.py do acervo."""
    return ("lei" if tipo.strip().upper().startswith("LEI") else "decreto", int(str(numero).replace(".", "")))


# Citações de artigos: cópia de legislacao.py do acervo (tests/test_paridade.py confere).
_CIT_NORMA = re.compile(r"\b(LEI|DECRETO|DEC)\.?\s*(?:MUNICIPAL\s*)?(?:N\s*[O.]?\s*)?(\d{1,2}\.?\d{3})(?:\s*/\s*\d{2,4})?")
_CIT_PARAGRAFO = re.compile(r"§+\s*\d+\s*O?|PARAGRAFO\s+\d+\s*O?")
_CIT_INTERVALO = re.compile(r"(\d+)\s*O?\s+(?:A|ATE)\s+(\d+)")
_CIT_ARTIGO = re.compile(r"(\d+)(?:\s*O)?(?:\s*-\s*([A-Z])(?![A-Z])|([A-Z])(?![A-Z0-9]))?")


def _plano(s: str) -> str:
    s = unicodedata.normalize("NFD", s.upper())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("º", "O").replace("°", "O")


def _expandir_intervalo(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    return ", ".join(str(n) for n in range(a, b + 1)) if 0 < b - a < 30 else m.group(0)


def citacoes(dispositivo: str) -> set[tuple[tuple[str, int], str]]:
    """'LEI 9725/09 - ART. 31, DECRETO 13842/10, ARTS. 84 E 89' -> {(('lei', 9725), '31'), (('decreto', 13842), '84'), ...}."""
    s = _plano(dispositivo or "")
    normas = list(_CIT_NORMA.finditer(s))
    saida = set()
    for k, m in enumerate(normas):
        trecho = s[m.end() : normas[k + 1].start() if k + 1 < len(normas) else len(s)]
        a = trecho.find("ART")
        if a < 0:
            continue
        trecho = _CIT_INTERVALO.sub(_expandir_intervalo, _CIT_PARAGRAFO.sub(" ", trecho[a:]))
        norma = chave_norma(m.group(1), m.group(2))
        for n in _CIT_ARTIGO.finditer(trecho):
            sufixo = n.group(2) or n.group(3)
            saida.add((norma, n.group(1) + (f"-{sufixo}" if sufixo else "")))
    return saida


# Na pergunta o artigo costuma vir antes da norma: "art. 13 da Lei 8.616".
_ARTIGO_ANTES = re.compile(
    r"\bART(?:IGO)?\.?\s*(\d+)(?:\s*O)?(?:\s*-\s*([A-Z])\b)?[^.;?\n]{0,40}?\b(LEI|DECRETO|DEC)\.?\s*(?:MUNICIPAL\s*)?"
    r"(?:N\s*[O.]?\s*)?(\d{1,2}\.?\d{3})"
)


def citacoes_da_pergunta(pergunta: str) -> list[tuple[tuple[str, int], str]]:
    s = _plano(pergunta)
    achadas = [(chave_norma(m.group(3), m.group(4)), m.group(1) + (f"-{m.group(2)}" if m.group(2) else ""))
               for m in _ARTIGO_ANTES.finditer(s)]
    return list(dict.fromkeys(achadas + sorted(citacoes(pergunta))))
