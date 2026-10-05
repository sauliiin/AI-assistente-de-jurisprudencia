"""
Índice do acervo para o assistente: encontra as decisões e os trechos que
respondem a uma pergunta, sem internet.

Duas buscas são combinadas (fusão por posição, RRF):

  - BM25 sobre o texto inteiro de cada decisão (palavras exatas, números de
    lei, artigos);
  - a rede neural do site do acervo (site_data/ia, treinada pelo treinar_ia.py
    do repositório jurisprudencia-juntas), que
    aproxima decisões sobre a mesma infração mesmo com palavras diferentes.

Das decisões encontradas saem só os trechos que importam para a pergunta e o
dispositivo (o que foi decidido); o modelo de linguagem não precisa ler a
decisão inteira. Para perguntas sobre tendência, o panorama é contado a partir
do gabarito do SIF (infração de cada auto) e do resultado de cada decisão.
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import normas
from .config import ACERVO, ENTENDIMENTO, PASTA_DADOS
from .texto import RESULTADOS, chave_infracao, normalizar, resultado_da_decisao, tokenizar

VOTOS = ACERVO / "site_data" / "votos.jsonl"
PARECERES = ACERVO / "site_data" / "pareceres.jsonl"
LEGISLACAO = ACERVO / "site_data" / "legislacao.jsonl"
ENTENDIMENTO_ARQUIVO = re.compile(r"ENTENDIMENTO JUNTAS", re.I)
IA_SITE = ACERVO / "site_data" / "ia"
VERSAO_INDICE = 9
# Peso de cada lista na fusão (BM25 da decisão, BM25 do melhor trecho, rede neural); ver assistente/avaliar.py.
PESOS_FUSAO = (1.0, 1.0, 1.0)

STOPWORDS = set(
    """
    a ao aos as ate com como da das de dela dele deles do dos e ela elas ele eles em entre era essa esse esta
    este eu foi foram ha isso isto ja la lhe mais mas me mesmo meu muito na nao nas nem no nos nossa o os ou
    para pela pelas pelo pelos por qual quando que quem se sem ser seu sua sao so tambem te tem ter um uma
    voce vai sobre quais quanto quantos quantas qual quais caso casos decisao decisoes junta juntas foi ser
    sido esta estao pode podem deve devem como onde porque ainda apos
    """.split()
)
_TERMO = re.compile(r"[a-z0-9]{2,}")
_DATA_SESSAO = re.compile(r"SESS[ÃA]O[^\n]{0,60}?(\d{2})/(\d{2})/(20\d{2})", re.I)
_DATA_EXTENSO = re.compile(
    r"\b(\d{1,2})\s+de\s+(janeiro|fevereiro|mar[çc]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
    r"\s+de\s+(20\d{2})",
    re.I,
)
_DATA_NUM = re.compile(r"\b(\d{2})/(\d{2})/(20\d{2})\b")
_JUNTA = re.compile(r"Junta Integrada de Julgamento Fiscal\s*(?:-\s*)?([IV]+|\d)\b|JIJFI-?\s*([IV]+)\b", re.I)
_DISPOSITIVO = re.compile(
    r"DISPOSITIVO|ANTE O EXPOSTO|DIANTE DO EXPOSTO|PELO EXPOSTO|POR TODO O EXPOSTO|ISTO POSTO|DESTA FORMA, (?:DEFIRO|INDEFIRO)|"
    r"\bVOTO\b\s*\n|CONCLUS[ÃA]O\s*\n",
    re.I,
)
_PAGINA = re.compile(r"^\s*\d{1,2}\s*/\s*\d{1,2}\s*$")
ROMANOS = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}
MESES = {m: i for i, m in enumerate(
    "janeiro fevereiro marco abril maio junho julho agosto setembro outubro novembro dezembro".split(), 1)}


def termos(texto: str) -> list[str]:
    """Termos do BM25: sem acento, minúsculos, com números, sem palavras vazias, sem plural, radical de 7 letras."""
    return [
        (t[:-1] if len(t) > 4 and t.endswith("s") and not t.isdigit() else t)[:7]
        for t in _TERMO.findall(normalizar(texto)) if t not in STOPWORDS
    ]


# Teses processuais aparecem com palavras diferentes das que o usuário usa ("fora do prazo" x "intempestiva").
EXPANSOES = [
    (r"fora do prazo|prazo|intempestiv|tempestiv|atrasad", "intempestiva intempestividade tempestividade prazo ciência"),
    (r"notifica", "notificação prévia auto de notificação ciência"),
    (r"\bnul|vicio|anula", "nulidade vício formal anulação"),
    (r"legitim|procura|represent", "ilegitimidade legitimidade procuração representação"),
    (r"prescri|decad", "prescrição decadência"),
    (r"microempres|pequeno porte|dupla visita|\bmei\b", "microempresa dupla visita lei complementar 123"),
    (r"reincid", "reincidência"),
    (r"multa|valor|dosimetr", "multa valor penalidade"),
    (r"prorroga", "prorrogação prazo exigências"),
    (r"\bfoto|\bprova", "prova fotos comprovação"),
    (r"endereco|local errad|local incorret", "endereço local constatação erro"),
    (r"ciencia|correio|diario oficial", "ciência correios publicação diário oficial"),
]


# Palavras de pedido/processo: não identificam a infração de que a pergunta trata.
TERMOS_PROCESSUAIS = set(termos(
    "apresentar apresentada defesa prazo conhecida conhecimento recurso pedido pedidos cancelamento cancelar "
    "deferido deferida deferidas deferidos indeferido indeferida auto autos infração multa notificação julgamento "
    "julgam decidem decide decidiu resultado juntas sessão fiscal processo anula anular nulidade fora dentro antes falta ausência prévia previamente vício erro falha prova provas"
))


def expandir(pergunta: str) -> str:
    plano = normalizar(pergunta)
    extras = [termos for padrao, termos in EXPANSOES if re.search(padrao, plano)]
    return " ".join([pergunta, *extras])


def so_digitos(texto: str) -> str:
    return re.sub(r"\D", "", texto or "")


@dataclass
class Decisao:
    indice: int
    file_id: str
    tipo: str  # "entendimento", "legislacao", "parecer", "voto" ou "panorama"
    titulo: str
    protocolo: str = ""
    assunto: str = ""
    instancia: str = ""
    resultado: str = ""
    data: str = ""  # AAAA-MM-DD da sessão, quando o texto traz
    junta: str = ""
    autos: list[dict] = field(default_factory=list)
    link: str = ""
    trechos: list[str] = field(default_factory=list)
    dispositivo: str = ""
    fundamentacao: int = -1  # posição do trecho onde começa a fundamentação (-1: não identificada)

    @property
    def ano(self) -> str:
        return self.data[:4]

    def cabecalho(self) -> str:
        """Ficha da decisão escrita a partir dos dados estruturados (não do texto livre)."""
        if self.tipo == "entendimento":
            return f"ENTENDIMENTO DAS JUNTAS (Vade Mecum de jurisprudência administrativa, 2024): {self.titulo}"
        if self.tipo == "legislacao":
            return f"LEGISLAÇÃO (texto vigente): {self.titulo}" + (f" ({self.assunto})" if self.assunto else "")
        if self.tipo == "parecer":
            return f"PARECER TÉCNICO: {self.titulo}"
        partes = [f"Protocolo {self.protocolo}" if self.protocolo else f"Arquivo {self.titulo[:80]}"]
        if self.instancia:
            partes.append(self.instancia)
        if self.junta:
            partes.append(f"JIJF {self.junta}")
        if self.data:
            a, m, d = self.data.split("-")
            partes.append(f"sessão de {d}/{m}/{a}")
        partes.append(f"resultado: {self.resultado or 'não identificado'}")
        linhas = [" · ".join(partes)]
        if self.assunto:
            linhas.append(f"Assunto: {self.assunto}")
        for auto in self.autos[:4]:
            linha = f"Auto {auto.get('numero', '')}: {(auto.get('infracao') or '').strip().capitalize()}"
            if auto.get("dispositivo_legal_transgredido"):
                linha += f" ({auto['dispositivo_legal_transgredido'].strip()})"
            linhas.append(linha)
        if len(self.autos) > 4:
            linhas.append(f"(+{len(self.autos) - 4} autos)")
        return "\n".join(linhas)


def _limpar(texto: str) -> list[str]:
    """Linhas sem cabeçalhos de página repetidos e sem numeração de página."""
    linhas = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in (texto or "").splitlines()]
    contagem = collections.Counter(ln for ln in linhas if ln and len(ln) < 90)
    vistas: set[str] = set()
    saida = []
    for ln in linhas:
        if _PAGINA.match(ln):
            continue
        if ln and contagem[ln] > 1 and len(ln) < 90:
            if ln in vistas:
                continue
            vistas.add(ln)
        saida.append(ln)
    return saida


def dividir_em_trechos(texto: str, alvo: int = 650) -> list[str]:
    """Parágrafos agrupados em trechos de ~alvo caracteres, quebrando preferencialmente em fim de frase."""
    linhas = _limpar(texto)
    paragrafos: list[str] = []
    atual: list[str] = []
    for ln in linhas:
        if not ln:
            if atual:
                paragrafos.append(" ".join(atual))
                atual = []
            continue
        atual.append(ln)
        # PDFs quebram a linha no meio da frase; linha que termina em ponto/dois-pontos fecha o parágrafo.
        if ln.endswith((".", ":", ";")) and sum(map(len, atual)) > 200:
            paragrafos.append(" ".join(atual))
            atual = []
    if atual:
        paragrafos.append(" ".join(atual))

    trechos: list[str] = []
    buffer = ""
    for p in paragrafos:
        while len(p) > alvo * 1.6:
            corte = p.rfind(". ", int(alvo * 0.6), int(alvo * 1.4))
            corte = corte + 1 if corte > 0 else alvo
            pedaco, p = p[:corte].strip(), p[corte:].strip()
            if buffer:
                trechos.append(buffer)
                buffer = ""
            trechos.append(pedaco)
        if buffer and len(buffer) + len(p) > alvo:
            trechos.append(buffer)
            buffer = p
        else:
            buffer = f"{buffer} {p}".strip()
    if buffer:
        trechos.append(buffer)
    return [t for t in trechos if len(t) > 25]


def extrair_dispositivo(texto: str) -> str:
    """O que foi decidido: a partir da última marca de conclusão, ou o final da decisão."""
    limpo = "\n".join(ln for ln in _limpar(texto) if ln)
    marcas = list(_DISPOSITIVO.finditer(limpo))
    inicio = marcas[-1].start() if marcas and marcas[-1].start() > len(limpo) * 0.4 else max(0, len(limpo) - 700)
    trecho = re.sub(r"\s+", " ", limpo[inicio : inicio + 900]).strip()
    # Corta assinaturas e rodapés depois da conclusão.
    fim = re.search(r"Belo Horizonte,|Relator\(a\)\s*$|_{5,}", trecho)
    return trecho[: fim.start()].strip() if fim and fim.start() > 40 else trecho


_FUNDAMENTACAO = re.compile(r"\bFUNDAMENTA[ÇC][ÃA]O\b|\bDO M[ÉE]RITO\b|^M[ÉE]RITO\b|\bVOTO\s*(?:DO RELATOR)?\s*$|\bAN[ÁA]LISE\b", re.I | re.M)


def inicio_fundamentacao(trechos: list[str]) -> int:
    """Primeiro trecho da fundamentação (o porquê da decisão), procurado depois do relatório."""
    for pos in range(len(trechos) // 4, len(trechos)):
        if _FUNDAMENTACAO.search(trechos[pos][:200]):
            return pos
    return -1


def extrair_data(texto: str, protocolo: str = "") -> str:
    """Data da sessão (AAAA-MM-DD). Descarta datas anteriores ao ano do protocolo (erro de modelo de documento)."""
    ano_minimo = re.search(r"(20\d{2})\D{0,2}\d{2}\s*$", protocolo or "")
    ano_minimo = ano_minimo.group(1) if ano_minimo else "2000"
    candidatas = [(a, mes, d) for d, mes, a in _DATA_SESSAO.findall(texto[:1500])]
    candidatas += [(a, f"{MESES[normalizar(mes)]:02d}", f"{int(d):02d}") for d, mes, a in _DATA_EXTENSO.findall(texto[-2500:])]
    for a, mes, d in candidatas:
        if ano_minimo <= a <= "2030" and "01" <= mes <= "12":
            return f"{a}-{mes}-{d}"
    return ""


def limpar_assunto(assunto: str) -> str:
    """Alguns PDFs trazem o cabeçalho inteiro no campo assunto; fica só o pedido."""
    assunto = re.split(r"\bAUTUADO\(S\)|\bCPF/CNPJ|\bREPRESENTANTE", assunto or "")[0]
    assunto = re.sub(r"^\s*\d{2}\.[\d./_-]+\s*", "", assunto)
    return re.sub(r"\s+", " ", assunto).strip()[:300]


def extrair_junta(texto: str) -> str:
    m = _JUNTA.search(texto[:1500])
    if not m:
        return ""
    valor = (m.group(1) or m.group(2) or "").upper()
    return ROMANOS.get(valor, valor)


def _bm25(docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> tuple[sp.csr_matrix, dict[str, int], np.ndarray]:
    """Matriz documento x termo já com os pesos BM25 (a consulta só soma colunas)."""
    vocab: dict[str, int] = {}
    linhas, colunas, valores = [], [], []
    for i, doc in enumerate(docs):
        for termo, n in collections.Counter(doc).items():
            linhas.append(i)
            colunas.append(vocab.setdefault(termo, len(vocab)))
            valores.append(n)
    tf = sp.csr_matrix((valores, (linhas, colunas)), shape=(len(docs), len(vocab)), dtype=np.float32)
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
    tam = np.array([len(d) for d in docs], dtype=np.float32)
    norma = k1 * (1 - b + b * tam / max(tam.mean(), 1.0))
    tf = tf.tocoo()
    pesos = tf.data * (k1 + 1) / (tf.data + norma[tf.row]) * idf[tf.col]
    matriz = sp.csr_matrix((pesos, (tf.row, tf.col)), shape=tf.shape, dtype=np.float32).tocsc()
    return matriz, vocab, idf


class Acervo:
    """Entendimento das Juntas, legislação, pareceres e votos, com os índices de busca. Use Acervo.carregar()."""

    def __init__(self) -> None:
        self.decisoes: list[Decisao] = []
        self.pareceres: list[Decisao] = []
        self.entendimento: list[Decisao] = []
        self.legislacao: list[Decisao] = []
        self.bm25_ent: sp.csc_matrix
        self.vocab_ent: dict[str, int] = {}
        self.idf_ent: np.ndarray
        self.bm25_leg: sp.csc_matrix
        self.vocab_leg: dict[str, int] = {}
        self.idf_leg: np.ndarray
        # (('lei', 9725), '31') -> posição do artigo em self.legislacao
        self.artigo_citado: dict[tuple[tuple[str, int], str], int] = {}
        self.bm25: sp.csc_matrix
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray
        self.bm25_trechos: sp.csc_matrix
        self.vocab_trechos: dict[str, int] = {}
        self.idf_trechos: np.ndarray
        self.trecho_doc: np.ndarray
        self.bm25_par: sp.csc_matrix
        self.vocab_par: dict[str, int] = {}
        self.idf_par: np.ndarray
        self.chaves: list[set[str]] = []
        self.nome_infracao: dict[str, str] = {}
        self.por_protocolo: dict[str, list[int]] = {}
        self.por_auto: dict[str, list[int]] = {}
        # Rede neural do site (opcional: sem ela, a busca fica só no BM25).
        self.neural_vocab: dict[str, int] = {}
        self.neural_idf: np.ndarray | None = None
        self.neural_w: np.ndarray | None = None
        self.neural_docs: np.ndarray | None = None

    # ------------------------------------------------------------------ construção

    @classmethod
    def carregar(cls, verboso: bool = True) -> "Acervo":
        cache = PASTA_DADOS / "indice.pkl"
        assinatura = [VERSAO_INDICE] + [
            (p.name, p.stat().st_size, int(p.stat().st_mtime))
            for p in (VOTOS, PARECERES, LEGISLACAO, ENTENDIMENTO) if p and p.exists()
        ]
        inicio = time.time()
        acervo = None
        if cache.exists():
            try:
                with cache.open("rb") as fh:
                    dados = pickle.load(fh)
                if dados.get("assinatura") == assinatura:
                    acervo = dados["acervo"]
            except Exception:  # cache corrompido ou de outra versão: refaz
                acervo = None
        if acervo is None:
            if verboso:
                print("Montando o índice do acervo (só na primeira vez ou quando o acervo muda)...", flush=True)
            acervo = cls._construir()
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as fh:
                pickle.dump({"assinatura": assinatura, "acervo": acervo}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        acervo._carregar_rede_neural()
        if verboso:
            print(
                f"Acervo: {len(acervo.entendimento)} tópicos do Entendimento das Juntas, "
                f"{len(acervo.legislacao)} artigos de legislação, {len({p.file_id for p in acervo.pareceres})} pareceres "
                f"e {len(acervo.decisoes)} decisões ({time.time() - inicio:.1f} s)",
                flush=True,
            )
            if not acervo.entendimento:
                print("[aviso] Entendimento das Juntas não encontrado no acervo nem em ASSISTENTE_ENTENDIMENTO.", flush=True)
        return acervo

    @classmethod
    def _construir(cls) -> "Acervo":
        acervo = cls()
        docs_bm25 = []
        with VOTOS.open(encoding="utf-8") as fh:
            for i, linha in enumerate(fh):
                v = json.loads(linha)
                texto = v.get("texto") or ""
                d = Decisao(
                    indice=i,
                    file_id=v["file_id"],
                    tipo="voto",
                    titulo=v.get("nome_arquivo", ""),
                    protocolo=v.get("protocolo") or "",
                    assunto=limpar_assunto(v.get("assunto") or ""),
                    instancia="2ª instância" if v.get("instancia") == "2a_instancia" else "1ª instância",
                    resultado=RESULTADOS[resultado_da_decisao(texto)],
                    data=extrair_data(texto, v.get("protocolo") or ""),
                    junta=extrair_junta(texto),
                    autos=[
                        {k: a.get(k) for k in ("numero", "tipo", "infracao", "dispositivo_legal_transgredido", "lei")}
                        for a in v.get("autos") or []
                    ],
                    link=v.get("drive_view_url") or "",
                    trechos=dividir_em_trechos(texto),
                    dispositivo=extrair_dispositivo(texto),
                )
                d.fundamentacao = inicio_fundamentacao(d.trechos)
                acervo.decisoes.append(d)
                infracoes = " ".join(
                    f"{a.get('infracao') or ''} {a.get('dispositivo_legal_transgredido') or ''}" for a in d.autos
                )
                docs_bm25.append(termos(f"{d.assunto} {infracoes} {texto}"))
                chaves = {chave_infracao(a["infracao"]) for a in d.autos if a.get("infracao")}
                acervo.chaves.append(chaves)
                for a in d.autos:
                    if a.get("infracao"):
                        acervo.nome_infracao.setdefault(chave_infracao(a["infracao"]), a["infracao"].strip())
                    if a.get("numero"):
                        acervo.por_auto.setdefault(a["numero"].upper(), []).append(i)
                        acervo.por_auto.setdefault(so_digitos(a["numero"]), []).append(i)
                if d.protocolo:
                    acervo.por_protocolo.setdefault(so_digitos(d.protocolo), []).append(i)
        acervo.bm25, acervo.vocab, acervo.idf = _bm25(docs_bm25)
        del docs_bm25
        # Índice por trecho: teses processuais (prazo, nulidade, legitimidade) ficam em parágrafos específicos.
        acervo.trecho_doc = np.array([d.indice for d in acervo.decisoes for _ in d.trechos], dtype=np.int32)
        acervo.bm25_trechos, acervo.vocab_trechos, acervo.idf_trechos = _bm25(
            [termos(t) for d in acervo.decisoes for t in d.trechos]
        )

        # Pareceres: indexados por trecho, porque são longos e tratam de vários temas.
        # O Entendimento das Juntas também está nessa pasta do Drive, mas é fonte própria (ver abaixo).
        docs_par = []
        entendimento = None
        if PARECERES.exists():
            with PARECERES.open(encoding="utf-8") as fh:
                for linha in fh:
                    p = json.loads(linha)
                    if ENTENDIMENTO_ARQUIVO.match(p.get("nome_arquivo", "")):
                        if "2024" in p["nome_arquivo"]:
                            entendimento = p
                        continue
                    trechos = dividir_em_trechos(p.get("texto") or "", alvo=900)
                    for n, trecho in enumerate(trechos):
                        acervo.pareceres.append(
                            Decisao(
                                indice=len(acervo.pareceres),
                                file_id=p["file_id"],
                                tipo="parecer",
                                titulo=f"{re.sub(r'\.(pdf|docx?)$', '', p.get('nome_arquivo', ''), flags=re.I)}"
                                f" ({p.get('categoria', '')}, parte {n + 1}/{len(trechos)})",
                                link=p.get("drive_view_url") or "",
                                trechos=[trecho],
                            )
                        )
                        docs_par.append(termos(f"{p.get('nome_arquivo', '')} {trecho}"))
        if docs_par:
            acervo.bm25_par, acervo.vocab_par, acervo.idf_par = _bm25(docs_par)

        # Entendimento das Juntas: um item por tópico do sumário.
        if ENTENDIMENTO and ENTENDIMENTO.exists():
            paragrafos, link = normas.paragrafos_docx(ENTENDIMENTO), ""
        elif entendimento:
            paragrafos, link = normas.paragrafos_texto(entendimento.get("texto") or ""), entendimento.get("drive_view_url") or ""
        else:
            paragrafos, link = [], ""
        for topico in normas.topicos_entendimento(paragrafos):
            acervo.entendimento.append(Decisao(
                indice=len(acervo.entendimento), file_id=f"entendimento#{len(acervo.entendimento)}",
                tipo="entendimento", titulo=topico["titulo"], link=link,
                trechos=dividir_em_trechos(topico["texto"], alvo=700),
            ))
        if acervo.entendimento:
            # O nome do próprio tópico conta duas vezes: "1.4 - Autuação para roçar lote penhorado" diz do que ele trata.
            acervo.bm25_ent, acervo.vocab_ent, acervo.idf_ent = _bm25(
                [termos(f"{e.titulo.split(' › ')[-1]} {e.titulo} {' '.join(e.trechos)}") for e in acervo.entendimento]
            )

        # Legislação: um item por artigo (ou item de anexo) vigente.
        if LEGISLACAO.exists():
            for u in normas.unidades_legislacao(LEGISLACAO):
                pos = len(acervo.legislacao)
                acervo.legislacao.append(Decisao(
                    indice=pos, file_id=f"{u['norma']}#{u['rotulo']}", tipo="legislacao",
                    titulo=f"{u['norma']}, {u['rotulo']}", assunto=u["contexto"], link=u["link"],
                    trechos=dividir_em_trechos(u["texto"], alvo=700),
                ))
                if u["tipo"] == "artigo" and " " in u["norma"]:
                    tipo, numero = u["norma"].split(" ", 1)
                    acervo.artigo_citado.setdefault((normas.chave_norma(tipo, numero), u["numero"]), pos)
            acervo.bm25_leg, acervo.vocab_leg, acervo.idf_leg = _bm25(
                [termos(f"{d.titulo} {d.assunto} {' '.join(d.trechos)}") for d in acervo.legislacao]
            )
        return acervo

    def _carregar_rede_neural(self) -> None:
        modelo = IA_SITE / "modelo.json"
        if not modelo.exists():
            return
        info = json.loads(modelo.read_text(encoding="utf-8"))
        if info.get("ids") != [d.file_id for d in self.decisoes]:
            print("[aviso] rede neural do site desatualizada em relação ao acervo; usando só BM25.", flush=True)
            return
        dim = info["dim"]
        self.neural_vocab = {t: i for i, t in enumerate(info["vocab"])}
        self.neural_idf = np.array(info["idf"], dtype=np.float32)
        w = np.fromfile(IA_SITE / "termos.bin", dtype=np.int8).reshape(-1, dim)
        self.neural_w = w.astype(np.float32)
        self.neural_docs = np.fromfile(IA_SITE / "docs.bin", dtype=np.int8).reshape(-1, dim).astype(np.float32) / 127

    # ------------------------------------------------------------------ busca

    def _pontuar_bm25(self, consulta: list[str], matriz, vocab, idf) -> tuple[np.ndarray, float]:
        colunas = [vocab[t] for t in dict.fromkeys(consulta) if t in vocab]
        if not colunas:
            return np.zeros(matriz.shape[0], dtype=np.float32), 0.0
        pontos = np.asarray(matriz[:, colunas].sum(axis=1)).ravel()
        return pontos, float(idf[colunas].sum())

    def _pontuar_neural(self, pergunta: str) -> np.ndarray | None:
        if self.neural_docs is None:
            return None
        contagem = collections.Counter(self.neural_vocab[t] for t in tokenizar(pergunta) if t in self.neural_vocab)
        if not contagem:
            return None
        x = np.zeros(len(self.neural_vocab), dtype=np.float32)
        for j, n in contagem.items():
            x[j] = (1 + math.log(n)) * self.neural_idf[j]
        x /= np.linalg.norm(x) or 1.0
        q = x @ self.neural_w
        q /= np.linalg.norm(q) or 1.0
        return self.neural_docs @ q

    def referencias_diretas(self, pergunta: str) -> list[int]:
        """Decisões citadas pelo número do protocolo ou do auto na própria pergunta."""
        achados: list[int] = []
        for m in re.finditer(r"\b\d{8,14}\s?[A-Za-z]{2}\b", pergunta):
            achados += self.por_auto.get(m.group(0).replace(" ", "").upper(), [])
        for m in re.finditer(r"\d[\d./_-]{9,}\d", pergunta):
            digitos = so_digitos(m.group(0))
            achados += self.por_protocolo.get(digitos, []) or self.por_auto.get(digitos, [])
        return list(dict.fromkeys(achados))

    def buscar(self, pergunta: str, k: int = 6, filtros: dict | None = None, profundidade: int = 400) -> list[tuple[int, float]]:
        """Decisões mais relevantes: [(índice, nota de 0 a 1)], melhores primeiro."""
        filtros = filtros or {}
        n = len(self.decisoes)
        consulta = termos(expandir(pergunta))
        bm25, _ = self._pontuar_bm25(consulta, self.bm25, self.vocab, self.idf)
        pontos_trechos, _ = self._pontuar_bm25(consulta, self.bm25_trechos, self.vocab_trechos, self.idf_trechos)
        por_trecho = np.zeros(n, dtype=np.float32)
        np.maximum.at(por_trecho, self.trecho_doc, pontos_trechos)
        neural = self._pontuar_neural(pergunta)

        mascara = np.ones(n, dtype=bool)
        if filtros.get("instancia"):
            mascara &= np.array([d.instancia.startswith(filtros["instancia"]) for d in self.decisoes])
        if filtros.get("ano"):
            mascara &= np.array([d.ano == filtros["ano"] for d in self.decisoes])
        if not mascara.any():
            mascara[:] = True

        # Fusão por posição (RRF): cada lista contribui 1/(60 + posição).
        fusao = np.zeros(n, dtype=np.float32)
        for pontos, peso in zip((bm25, por_trecho, neural), PESOS_FUSAO, strict=True):
            if pontos is None or not pontos.any():
                continue
            pontos = np.where(mascara, pontos, -np.inf)
            topo = np.argsort(-pontos)[:profundidade]
            fusao[topo] += peso / (60 + np.arange(1, len(topo) + 1))
        diretas = [i for i in self.referencias_diretas(pergunta)]
        for i in diretas:
            fusao[i] += 1.0
        ordem = np.argsort(-fusao)[:k]
        maximo = float(fusao[ordem[0]]) if len(ordem) and fusao[ordem[0]] > 0 else 1.0
        return [(int(i), float(fusao[i]) / maximo) for i in ordem if fusao[i] > 0]

    def semelhantes(self, i: int, k: int = 3) -> list[int]:
        """Decisões mais próximas de outra, no espaço da rede neural do site."""
        if self.neural_docs is None:
            return []
        sim = self.neural_docs @ self.neural_docs[i]
        sim[i] = -9
        return [int(j) for j in np.argsort(-sim)[:k]]

    def _buscar_com_cobertura(self, pergunta: str, itens: list[Decisao], matriz, vocab, idf, k: int
                              ) -> list[tuple[int, float, float]]:
        """Melhores itens pelo BM25: [(índice, cobertura, força)], um por arquivo.

        cobertura: fração da consulta (ponderada pelo IDF) presente no item, de 0 a 1;
        força: nota BM25 dividida pelo IDF da consulta (acima de ~0,85, o item trata do assunto).
        """
        if not itens:
            return []
        consulta = termos(pergunta)
        pontos, total_idf = self._pontuar_bm25(consulta, matriz, vocab, idf)
        if not total_idf:
            return []
        ordem = np.argsort(-pontos)[: k * 4]
        saida, vistos = [], set()
        for i in ordem:
            item = itens[int(i)]
            if item.file_id in vistos or pontos[i] <= 0:
                continue
            vistos.add(item.file_id)
            presentes = set(termos(" ".join(item.trechos))) | set(termos(item.titulo)) | set(termos(item.assunto))
            cobertura = sum(idf[vocab[t]] for t in set(consulta) if t in presentes and t in vocab)
            saida.append((int(i), float(cobertura / total_idf), float(pontos[i] / total_idf)))
            if len(saida) == k:
                break
        return saida

    def buscar_entendimento(self, pergunta: str, k: int = 2) -> list[tuple[int, float, float]]:
        """Tópicos do Entendimento das Juntas, com a fração da consulta coberta."""
        return self._buscar_com_cobertura(pergunta, self.entendimento, getattr(self, "bm25_ent", None),
                                          self.vocab_ent, getattr(self, "idf_ent", None), k)

    def buscar_legislacao(self, pergunta: str, k: int = 2) -> list[tuple[int, float, float]]:
        """Artigos de legislação, com a fração da consulta coberta."""
        return self._buscar_com_cobertura(pergunta, self.legislacao, getattr(self, "bm25_leg", None),
                                          self.vocab_leg, getattr(self, "idf_leg", None), k)

    def buscar_pareceres(self, pergunta: str, k: int = 2) -> list[tuple[int, float, float]]:
        """Trechos de pareceres, com a fração da consulta coberta."""
        return self._buscar_com_cobertura(pergunta, self.pareceres, getattr(self, "bm25_par", None),
                                          self.vocab_par, getattr(self, "idf_par", None), k)

    def artigos_citados(self, pergunta: str, decisoes: list[int], maximo: int = 2) -> list[int]:
        """Artigos citados na pergunta ("art. 13 da Lei 8.616") e, depois, os mais citados como
        dispositivo transgredido nos autos das decisões encontradas (a base legal da infração)."""
        saida = [self.artigo_citado[c] for c in normas.citacoes_da_pergunta(pergunta) if c in self.artigo_citado]
        contagem: collections.Counter[int] = collections.Counter()
        for i in decisoes:
            citados = set()
            for auto in self.decisoes[i].autos:
                citados |= {self.artigo_citado[c] for c in normas.citacoes(auto.get("dispositivo_legal_transgredido") or "")
                            if c in self.artigo_citado}
            contagem.update(citados)
        # Base legal comum a pelo menos duas decisões (ou à única decisão consultada).
        minimo = 1 if len(decisoes) == 1 else 2
        saida += [a for a, n in contagem.most_common() if n >= minimo]
        return list(dict.fromkeys(saida))[:maximo]

    def fundamentacao_completa(self, decisao: Decisao, pergunta: str, limite: int = 3600) -> list[str]:
        """Para o caso citado na pergunta: a fundamentação inteira (ou os melhores trechos, se não achada)."""
        if decisao.fundamentacao < 0:
            return self.melhores_trechos(decisao, pergunta, n=4, limite=limite)
        saida, usado = [], 0
        for trecho in decisao.trechos[decisao.fundamentacao :]:
            if decisao.dispositivo[:60] and decisao.dispositivo[:60] in trecho:
                break
            saida.append(trecho[: limite - usado])
            usado += len(saida[-1])
            if usado >= limite:
                break
        return saida

    def melhores_trechos(self, decisao: Decisao, pergunta: str, n: int = 2, limite: int = 1300) -> list[str]:
        """Os trechos da decisão que mais têm a ver com a pergunta (sem repetir o dispositivo)."""
        consulta = set(termos(expandir(pergunta)))
        notas = []
        for pos, trecho in enumerate(decisao.trechos):
            presentes = set(termos(trecho))
            nota = sum(float(self.idf[self.vocab[t]]) for t in consulta & presentes if t in self.vocab)
            # Fundamentação (o porquê) vale mais que relatório e alegações da defesa.
            if decisao.fundamentacao >= 0:
                nota *= 1.6 if pos >= decisao.fundamentacao else 1.0
            else:
                nota *= 1 + 0.5 * pos / max(len(decisao.trechos), 1)
            notas.append((nota, pos))
        escolhidos = sorted(pos for nota, pos in sorted(notas, reverse=True)[:n] if nota > 0)
        saida, usado = [], 0
        disp = decisao.dispositivo[:120]
        for pos in escolhidos:
            trecho = decisao.trechos[pos]
            if disp and disp[:80] in trecho:
                continue
            trecho = trecho[: max(200, limite - usado)]
            usado += len(trecho)
            saida.append(trecho)
            if usado >= limite:
                break
        return saida

    # ------------------------------------------------------------------ panorama

    def infracoes_da_pergunta(self, pergunta: str, resultados: list[tuple[int, float]], maximo: int = 2) -> list[list[str]]:
        """Infrações (gabarito do SIF) de que a pergunta trata, para o panorama.

        Vale a infração cujo nome cobre boa parte do peso (IDF) dos termos da pergunta
        (cunha, mesa e cadeira...) ou, quando a pergunta usa outras palavras, a que domina
        claramente as decisões encontradas. Teses processuais ("fora do prazo") não puxam panorama.
        Códigos do SIF para a mesma conduta (mesmo início de nome) vêm agrupados.
        """
        if not hasattr(self, "_termos_infracao"):
            self._termos_infracao = {c: set(termos(n)) for c, n in self.nome_infracao.items()}
            self._contagem = collections.Counter(k for ks in self.chaves for k in ks)
        consulta = {t for t in termos(pergunta) if t in self.vocab and t not in TERMOS_PROCESSUAIS}
        total_idf = sum(float(self.idf[self.vocab[t]]) for t in consulta)
        peso: collections.Counter[str] = collections.Counter()
        total = 0.0
        for pos, (i, _) in enumerate(resultados):
            w = 1 / (pos + 3)
            total += w
            for chave in self.chaves[i]:
                peso[chave] += w / len(self.chaves[i])
        candidatas = []
        for chave, nome in self._termos_infracao.items():
            comum = sum(float(self.idf[self.vocab[t]]) for t in consulta & nome)
            cobertura = comum / total_idf if total_idf else 0.0
            parcela = peso[chave] / total if total else 0.0
            if cobertura >= 0.3 or parcela >= 0.5:
                nota = cobertura + 0.5 * parcela + 0.05 * math.log10(1 + self._contagem[chave])
                candidatas.append((nota, self._contagem[chave], chave))
        candidatas.sort(reverse=True)
        if not candidatas:
            return []
        melhor = candidatas[0][0]
        grupos: list[list[str]] = []
        for nota, _, chave in candidatas:
            if nota < melhor - 0.1 or len(grupos) == maximo:
                break
            raiz = " ".join(chave.split()[:6])
            if any(" ".join(g[0].split()[:6]) == raiz for g in grupos):
                continue
            grupos.append(sorted((c for c in self._termos_infracao if " ".join(c.split()[:6]) == raiz),
                                 key=lambda c: -self._contagem[c]))
        return grupos

    def panorama(self, chaves: list[str] | None, filtros: dict | None = None) -> dict:
        """Contagem exata de resultados das decisões sobre uma infração (ou do acervo inteiro, se chaves=None)."""
        filtros = filtros or {}
        alvo = set(chaves or [])
        idx = [
            i for i, ks in enumerate(self.chaves) if (chaves is None or ks & alvo)
            and (not filtros.get("instancia") or self.decisoes[i].instancia.startswith(filtros["instancia"]))
            and (not filtros.get("ano") or self.decisoes[i].ano == filtros["ano"])
        ]
        resultados = collections.Counter(self.decisoes[i].resultado or "não identificado" for i in idx)
        instancias = collections.Counter(self.decisoes[i].instancia for i in idx)
        anos = collections.Counter(self.decisoes[i].ano for i in idx if self.decisoes[i].ano)
        return {
            "infracao": (
                self.nome_infracao.get(chaves[0], chaves[0]).capitalize()
                + (f" (e mais {len(chaves) - 1} variante(s) do código no SIF)" if len(chaves) > 1 else "")
                if chaves else "todas (acervo inteiro)"
            ),
            "total": len(idx),
            "resultados": dict(resultados.most_common()),
            "instancias": dict(instancias.most_common()),
            "anos": dict(sorted(anos.items())),
        }


def filtros_da_pergunta(pergunta: str) -> dict:
    """Filtros explícitos: instância e ano citados na pergunta."""
    # Números de protocolo e de auto contêm um ano ("31.00960807/2025-31"), que não é filtro.
    plano = normalizar(re.sub(r"\d[\d./_-]{9,}\d|\b\d{8,14}\s?[A-Za-z]{2}\b", " ", pergunta))
    filtros: dict = {}
    if re.search(r"\b(2a|2|segunda)\s+instancia|\bccf\b|conselho", plano):
        filtros["instancia"] = "2ª"
    elif re.search(r"\b(1a|1|primeira)\s+instancia", plano):
        filtros["instancia"] = "1ª"
    anos = re.findall(r"\b(20[12]\d)\b", plano)
    if len(anos) == 1:
        filtros["ano"] = anos[0]
    return filtros


PERGUNTA_QUANTITATIVA = re.compile(
    r"quant[oa]s|percent|porcent|propor|taxa|estatistic|frequen|maioria|costuma|tendencia|geralmente|"
    r"normalmente|chance|probab|em geral|com que frequencia",
)


def e_quantitativa(pergunta: str) -> bool:
    return bool(PERGUNTA_QUANTITATIVA.search(normalizar(pergunta)))
