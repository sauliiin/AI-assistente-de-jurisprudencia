"""
Junta as peças: busca no acervo -> monta as fontes -> o modelo local escreve a
resposta citando [n] -> o programa confere as citações e lista as fontes.

As fontes vão ao modelo em ordem de autoridade, que é também a ordem em que ele
deve consultá-las: Entendimento das Juntas, legislação, pareceres, decisões
(e, por fim, o panorama com as contagens).

A lista de fontes (protocolo, data, resultado, link) é escrita pelo programa a
partir dos dados, nunca pelo modelo; assim ela não pode ser inventada.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .acervo import Acervo, Decisao, e_quantitativa, filtros_da_pergunta, termos
from .llm import ModeloLocal

SISTEMA = """Você é o assistente de jurisprudência das Juntas Integradas de Julgamento Fiscal (JIJF) de Belo Horizonte. \
Responde perguntas de servidores e julgadores consultando o Entendimento das Juntas, a legislação municipal, \
pareceres técnicos e decisões reais (votos de 1ª e 2ª instância).

Ordem de consulta. As FONTES vêm nesta ordem e devem ser lidas e usadas nesta ordem:
  1º ENTENDIMENTO DAS JUNTAS: a posição consolidada das Juntas. Se uma fonte de entendimento trata do tema, ela é a base da resposta.
  2º LEGISLAÇÃO: o texto vigente da lei ou decreto. Use para dar o fundamento legal (artigo e norma, copiados da fonte).
  3º PARECER TÉCNICO: orientação dos órgãos técnicos sobre a aplicação da lei.
  4º DECISÕES: casos concretos. Servem para mostrar como o entendimento e a lei foram aplicados; não prevalecem sobre eles.
  Se nenhuma fonte de um nível foi enviada, ou se ela não trata do que foi perguntado, passe ao nível seguinte. Se uma decisão contrariar o entendimento ou a lei, \
apresente o entendimento como a posição das Juntas e aponte a decisão divergente.

Regras:
1. Use SOMENTE as FONTES enviadas na mensagem. Não use conhecimento externo sobre leis ou casos.
2. Depois de cada afirmação tirada de uma fonte, cite-a entre colchetes: [1], [2]. Só existem as fontes numeradas que foram enviadas.
3. Nunca invente protocolos, números de auto, artigos, valores, datas ou nomes. Copie-os exatamente das fontes.
4. Responda com o que as fontes mostram, mesmo que parcialmente. Só diga que as fontes não tratam do tema se nenhuma delas tratar; nesse caso diga o que mostram de mais próximo.
4a. Só atribua à Junta um fundamento que esteja escrito no Entendimento, nos Trechos ou na Decisão. Alegações da defesa ou do fiscal não são fundamentos da Junta: deixe claro quem disse. Se o motivo da decisão não aparece no material, diga que o trecho disponível não mostra o motivo.
5. Para "quantos", "costuma", "tendência" ou "chance", use os números da fonte PANORAMA (contagem de todo o acervo), não a amostra de decisões. Copie o número da linha exata do resultado perguntado: DEFERIDO e INDEFERIDO são resultados opostos; se a linha diz 0, a resposta é zero. Não reaproveite números de respostas anteriores.
6. Explique o porquê: os fundamentos usados pela Junta (prova, prazo, legitimidade, nulidade do auto, dispositivo legal) e o resultado.
7. Se as fontes divergem entre si, mostre a divergência e quais decisões foram em cada sentido.

Formato (português do Brasil, direto, sem floreios):
- Primeiro a resposta em 1 a 3 frases.
- Depois os fundamentos em tópicos curtos, cada um com sua citação, na ordem: entendimento, legislação, parecer, decisões.
- Não repita a lista de fontes no final; o sistema já mostra."""


@dataclass
class Fonte:
    numero: int
    decisao: Decisao
    trechos: list[str]
    motivo: str = ""  # "citada na pergunta", "semelhante", ""

    def para_o_modelo(self) -> str:
        if self.decisao.tipo in ("entendimento", "legislacao", "parecer"):
            return "\n".join([f"[{self.numero}] {self.decisao.cabecalho()}"] + [f"Trecho: «{t}»" for t in self.trechos])
        if self.decisao.tipo == "panorama":
            return f"[{self.numero}] PANORAMA (contagem exata de todo o acervo, pelo gabarito do SIF)\n{self.trechos[0]}"
        partes = [f"[{self.numero}] {self.decisao.cabecalho()}"]
        for trecho in self.trechos:
            partes.append(f"Trecho: «{trecho}»")
        if self.decisao.dispositivo:
            partes.append(f"Decisão: «{self.decisao.dispositivo[:700]}»")
        return "\n".join(partes)

    def resumo(self) -> dict:
        d = self.decisao
        return {
            "numero": self.numero,
            "tipo": d.tipo,
            "titulo": d.titulo if d.tipo != "voto" else f"Protocolo {d.protocolo}",
            "contexto": d.assunto if d.tipo == "legislacao" else "",
            "texto": self.trechos[0] if d.tipo == "panorama" else "",
            "protocolo": d.protocolo or re.sub(r"\.(pdf|docx?)$", "", d.titulo, flags=re.I)[:70],
            "instancia": d.instancia,
            "data": d.data,
            "resultado": d.resultado,
            "infracoes": sorted({(a.get("infracao") or "").strip().capitalize() for a in d.autos if a.get("infracao")}),
            "link": d.link,
            "motivo": self.motivo,
            "dispositivo": d.dispositivo[:400],
        }


@dataclass
class Turno:
    pergunta: str
    resposta: str


@dataclass
class Conversa:
    turnos: list[Turno] = field(default_factory=list)


# Para o tópico do Entendimento ou o artigo achado pela busca irem ao modelo: fração da pergunta
# (ponderada pelo IDF) que ele contém e nota BM25 relativa (medidas em perguntas com e sem tópico).
LIMIAR_COBERTURA = 0.5
LIMIAR_FORCA = 0.85

ORDEM_RESULTADOS = ["deferido", "parcialmente deferido", "indeferido", "não conhecido", "extinto", "diligência", "não identificado"]


def _formatar_panorama(p: dict) -> str:
    """Panorama em ordem fixa e com zeros explícitos: evita que o modelo troque deferido por indeferido."""
    total = p["total"]
    if not total:
        return ""
    linhas = [f"Infração: {p['infracao']}"]
    inst = "; ".join(f"{nome}: {n}" for nome, n in p["instancias"].items())
    linhas.append(f"Total de decisões no acervo: {total} ({inst})")
    for nome in ORDEM_RESULTADOS:
        n = p["resultados"].get(nome, 0)
        if n or nome in ORDEM_RESULTADOS[:4]:
            linhas.append(f"- {nome.upper()}: {n} de {total} ({100 * n / total:.0f}%)")
    anos = ", ".join(f"{a}: {n}" for a, n in list(p["anos"].items())[-6:])
    linhas.append(f"Decisões por ano (últimos): {anos}")
    return "\n".join(linhas)


class Assistente:
    def __init__(self, acervo: Acervo, modelo: ModeloLocal, n_fontes: int = 5, chars_por_fonte: int = 1100) -> None:
        # n_fontes e chars_por_fonte valem para as decisões; as fontes normativas têm limites próprios.
        self.acervo = acervo
        self.modelo = modelo
        self.n_fontes = n_fontes
        self.chars_por_fonte = chars_por_fonte

    def _consulta(self, pergunta: str, conversa: Conversa | None) -> str:
        """Pergunta de acompanhamento curta ("e na 2ª instância?") herda o tema da anterior."""
        if conversa and conversa.turnos and len(termos(pergunta)) < 7:
            return f"{conversa.turnos[-1].pergunta} {pergunta}"
        return pergunta

    def preparar(self, pergunta: str, conversa: Conversa | None = None) -> list[Fonte]:
        """Fontes numeradas na ordem de consulta: entendimento, legislação, parecer, decisões e panorama."""
        consulta = self._consulta(pergunta, conversa)
        filtros = filtros_da_pergunta(pergunta) or filtros_da_pergunta(consulta)
        diretas = self.acervo.referencias_diretas(pergunta)

        # As decisões são buscadas primeiro (a base legal dos autos delas indica a legislação),
        # mas vão para o fim da lista.
        decisoes: list[Fonte] = []
        if diretas:
            # Caso específico: ele mesmo (com mais texto) e alguns semelhantes para comparação.
            for i in diretas[:3]:
                d = self.acervo.decisoes[i]
                decisoes.append(Fonte(0, d, self.acervo.fundamentacao_completa(d, consulta), "citada na pergunta"))
            for i in self.acervo.semelhantes(diretas[0], k=2):
                d = self.acervo.decisoes[i]
                decisoes.append(Fonte(0, d, self.acervo.melhores_trechos(d, consulta, n=1, limite=600), "semelhante"))
            encontrados = [(i, 1.0) for i in diretas]
        else:
            encontrados = self.acervo.buscar(consulta, k=20, filtros=filtros)
            for i, _ in encontrados[: self.n_fontes]:
                d = self.acervo.decisoes[i]
                decisoes.append(Fonte(0, d, self.acervo.melhores_trechos(d, consulta, n=2, limite=self.chars_por_fonte)))

        # 1º Entendimento das Juntas: o tópico que trata da pergunta, se houver.
        entendimento = [
            Fonte(0, self.acervo.entendimento[i], self._texto(self.acervo.entendimento[i], consulta, 1800), "entendimento")
            for i, cobertura, forca in self.acervo.buscar_entendimento(consulta, k=1)
            if cobertura >= LIMIAR_COBERTURA and forca >= LIMIAR_FORCA
        ]

        # 2º Legislação: artigos citados na pergunta ou nos autos das decisões, e o que melhor cobre a pergunta.
        artigos = self.acervo.artigos_citados(consulta, [i for i, _ in encontrados[: self.n_fontes]])
        artigos += [i for i, cobertura, forca in self.acervo.buscar_legislacao(consulta, k=1)
                    if cobertura >= LIMIAR_COBERTURA and forca >= LIMIAR_FORCA]
        legislacao = [
            Fonte(0, self.acervo.legislacao[i], self._texto(self.acervo.legislacao[i], consulta, 1000), "legislação")
            for i in list(dict.fromkeys(artigos))[:3]
        ]

        # 3º Parecer técnico: só quando cobre bem a pergunta (ou quando é pedido).
        pareceres: list[Fonte] = []
        pede_parecer = re.search(r"parecer|orienta|dilu|gesle|nota orientativa", consulta, re.I)
        for i, cobertura, _ in self.acervo.buscar_pareceres(consulta, k=1):
            if cobertura >= 0.7 or (pede_parecer and cobertura >= 0.4):
                p = self.acervo.pareceres[i]
                pareceres.append(Fonte(0, p, [p.trechos[0][:1100]], "parecer"))

        # Panorama: contagem exata do acervo, enviada como mais uma fonte numerada (citável).
        panoramas = []
        for chaves in self.acervo.infracoes_da_pergunta(consulta, encontrados):
            panoramas.append(_formatar_panorama(self.acervo.panorama(chaves, filtros)))
        if e_quantitativa(pergunta) and not panoramas and not diretas:
            panoramas.append(_formatar_panorama(self.acervo.panorama(None, filtros)))
        panorama = [Fonte(0, Decisao(-1, "", "panorama", "Panorama do acervo"), [t], "panorama") for t in filter(None, panoramas)]

        fontes = entendimento + legislacao + pareceres + decisoes + panorama
        for numero, fonte in enumerate(fontes, 1):
            fonte.numero = numero
        return fontes

    def _texto(self, item: Decisao, consulta: str, limite: int) -> list[str]:
        """Tópico ou artigo inteiro, se couber; senão, os trechos que mais têm a ver com a pergunta."""
        if sum(map(len, item.trechos)) <= limite:
            return item.trechos
        return self.acervo.melhores_trechos(item, consulta, n=3, limite=limite) or [item.trechos[0][:limite]]

    def mensagens(self, pergunta: str, fontes: list[Fonte], conversa: Conversa | None) -> list[dict]:
        msgs = [{"role": "system", "content": SISTEMA}]
        for turno in (conversa.turnos[-2:] if conversa else []):
            msgs.append({"role": "user", "content": turno.pergunta})
            msgs.append({"role": "assistant", "content": turno.resposta[:800]})
        blocos = ["FONTES:"] + [f.para_o_modelo() for f in fontes]
        blocos.append(f"PERGUNTA: {pergunta}")
        msgs.append({"role": "user", "content": "\n\n".join(blocos)})
        return msgs

    def responder(self, pergunta: str, conversa: Conversa | None = None, pensar: bool = False) -> Iterator[dict]:
        """Eventos: {"tipo": "fontes"}, vários {"tipo": "texto"}, e {"tipo": "fim"} com a checagem das citações."""
        inicio = time.time()
        fontes = self.preparar(pergunta, conversa)
        yield {"tipo": "fontes", "fontes": [f.resumo() for f in fontes], "busca_s": round(time.time() - inicio, 2)}
        mensagens = self.mensagens(pergunta, fontes, conversa)
        stats: dict = {}
        partes = []
        for pedaco in self.modelo.conversar(mensagens, pensar=pensar, max_tokens=3500 if pensar else 900, estatisticas=stats):
            partes.append(pedaco)
            yield {"tipo": "texto", "texto": pedaco}
        resposta = re.sub(r"<think>.*?</think>\s*", "", "".join(partes), flags=re.S).strip()
        citadas = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", resposta)})
        invalidas = [n for n in citadas if not 1 <= n <= len(fontes)]
        if conversa is not None:
            conversa.turnos.append(Turno(pergunta, resposta))
        yield {
            "tipo": "fim",
            "resposta": resposta,
            "citadas": [n for n in citadas if n not in invalidas],
            "citacoes_invalidas": invalidas,
            "segundos": round(time.time() - inicio, 1),
            "tokens_contexto": stats.get("prompt_n"),
            "tokens_resposta": stats.get("predicted_n"),
            "leitura_tps": round(stats.get("prompt_per_second") or 0, 1),
            "escrita_tps": round(stats.get("predicted_per_second") or 0, 1),
        }
