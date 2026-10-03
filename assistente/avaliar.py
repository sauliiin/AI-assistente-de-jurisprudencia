"""
Mede o assistente com gabaritos tirados do próprio acervo.

    python3 -m assistente.avaliar              # só a busca (rápido, sem modelo de linguagem)
    python3 -m assistente.avaliar --respostas 10   # também 10 perguntas respondidas pelo modelo

Busca:
  - protocolo: perguntando pelo número do protocolo, a decisão certa vem em 1º?
  - narrativa: a descrição do fiscal (sem o nome da infração) traz, entre as 5
    fontes, decisões sobre a mesma infração (gabarito do SIF)?
Respostas:
  - "Qual foi o resultado do protocolo X?": a resposta traz o resultado certo
    (lido do dispositivo da decisão) e cita só fontes que existem?
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

from .acervo import VOTOS, Acervo, termos
from .config import PASTA_DADOS
from .llm import ModeloLocal
from .respondedor import Assistente
from .texto import AUTO, FISCAL, normalizar

SEMENTE = 11


def avaliar_busca(acervo: Acervo, n: int = 400) -> dict:
    rng = random.Random(SEMENTE)
    amostra = rng.sample(range(len(acervo.decisoes)), n)

    acertos_protocolo = []
    for i in amostra[:200]:
        d = acervo.decisoes[i]
        if not d.protocolo:
            continue
        r = acervo.buscar(f"Qual foi o resultado do protocolo {d.protocolo}?", k=1)
        alvo = acervo.por_protocolo.get(re.sub(r"\D", "", d.protocolo), [])
        acertos_protocolo.append(bool(r) and r[0][0] in alvo)

    texto = {}
    with open(VOTOS, encoding="utf-8") as fh:
        for pos, linha in enumerate(fh):
            if pos in set(amostra):
                texto[pos] = json.loads(linha).get("texto") or ""
    p5, p5_bm25, consultas = [], [], 0
    for i in amostra:
        achado = FISCAL.search(texto.get(i, ""))
        if not achado or len(acervo.decisoes[i].autos) != 1 or not acervo.chaves[i]:
            continue
        consulta = AUTO.sub("", achado.group(1))[:600]
        if len(consulta) < 60:
            continue
        consultas += 1
        chave = next(iter(acervo.chaves[i]))
        r = [j for j, _ in acervo.buscar(consulta, k=6) if j != i][:5]
        p5.append(np.mean([chave in acervo.chaves[j] for j in r]) if r else 0.0)
        pontos, _ = acervo._pontuar_bm25(termos(consulta), acervo.bm25, acervo.vocab, acervo.idf)
        pontos[i] = -1
        topo = np.argsort(-pontos)[:5]
        p5_bm25.append(np.mean([chave in acervo.chaves[j] for j in topo]))
    return {
        "protocolo_top1": round(float(np.mean(acertos_protocolo)), 3),
        "protocolos_testados": len(acertos_protocolo),
        "narrativa_p5_fusao": round(float(np.mean(p5)), 3),
        "narrativa_p5_so_bm25": round(float(np.mean(p5_bm25)), 3),
        "consultas_narrativa": consultas,
    }


# Perguntas sobre teses, escritas sem a palavra do gabarito; o gabarito vem do texto da decisão.
TESES = [
    ("Defesa apresentada fora do prazo é analisada?", r"intempestiv"),
    ("O recurso foi protocolado depois do prazo legal, a Junta conhece?", r"intempestiv"),
    ("Quem assina a defesa sem procuração pode recorrer?", r"ilegitim|sem procura|nao (esta|estando) (legalmente )?representad"),
    ("Pessoa que não é dona do imóvel pode apresentar defesa?", r"ilegitim"),
    ("O que acontece quando o auto já foi pago ou a irregularidade deixou de existir antes do julgamento?", r"perda d[oe] objeto|extint"),
    ("A Junta concede mais tempo para cumprir as exigências da notificação?", r"prorroga"),
    ("Erro no endereço do auto de infração leva ao cancelamento?", r"endere[cç]o"),
    ("Microempresa tem direito à dupla visita antes da multa?", r"dupla visita"),
    ("A Junta pode converter o julgamento em diligência para ouvir a fiscalização?", r"dilig[eê]ncia"),
    ("Auto lavrado sem descrição suficiente da conduta é nulo?", r"nulidade|\bnulo\b|vicio|v[ií]cio"),
]


def avaliar_teses(acervo: Acervo) -> dict:
    """Fração das 5 fontes que de fato tratam da tese (gabarito por expressão no texto da decisão)."""
    p5 = []
    for pergunta, gabarito in TESES:
        r = acervo.buscar(pergunta, k=5)
        p5.append(np.mean([
            bool(re.search(gabarito, normalizar(" ".join(acervo.decisoes[i].trechos) + acervo.decisoes[i].dispositivo)))
            for i, _ in r
        ]))
    return {"teses_p5": round(float(np.mean(p5)), 3), "teses": len(TESES)}


SINONIMOS = {
    "indeferido": r"indefer|negad|improced|mantid",
    "deferido": r"(?<!in)defer|cancelad|proced",
    "parcialmente deferido": r"parcial",
    "não conhecido": r"nao (foi )?conhec|nao conheceu|nao se conhec|intempestiv|ilegitim",
}


def avaliar_respostas(acervo: Acervo, n: int) -> dict:
    rng = random.Random(SEMENTE + 1)
    candidatos = [d for d in acervo.decisoes if d.protocolo and d.resultado in SINONIMOS]
    amostra = rng.sample(candidatos, n)
    modelo = ModeloLocal()
    modelo.garantir()
    assistente = Assistente(acervo, modelo)
    linhas = []
    for d in amostra:
        pergunta = f"Qual foi o resultado do protocolo {d.protocolo}?"
        fim = [ev for ev in assistente.responder(pergunta) if ev["tipo"] == "fim"][0]
        primeira = normalizar(fim["resposta"][:400])
        certo = bool(re.search(SINONIMOS[d.resultado], primeira))
        linhas.append({
            "protocolo": d.protocolo, "gabarito": d.resultado, "acertou": certo,
            "citou": bool(fim["citadas"]), "citacao_invalida": bool(fim["citacoes_invalidas"]),
            "segundos": fim["segundos"], "resposta": fim["resposta"][:300],
        })
        print(f"  {'✓' if certo else '✗'} {d.protocolo} ({d.resultado}) {fim['segundos']} s", flush=True)
    modelo.encerrar()
    return {
        "modelo": modelo.nome,
        "perguntas": n,
        "resultado_correto": round(np.mean([x["acertou"] for x in linhas]), 3),
        "com_citacao": round(np.mean([x["citou"] for x in linhas]), 3),
        "citacao_invalida": round(np.mean([x["citacao_invalida"] for x in linhas]), 3),
        "segundos_medio": round(np.mean([x["segundos"] for x in linhas]), 1),
        "detalhes": linhas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--respostas", type=int, default=0, help="quantas perguntas responder com o modelo")
    args = parser.parse_args()
    acervo = Acervo.carregar()
    inicio = time.time()
    relatorio = {"busca": {**avaliar_busca(acervo), **avaliar_teses(acervo)}}
    print("Busca:", json.dumps(relatorio["busca"], ensure_ascii=False), flush=True)
    if args.respostas:
        relatorio["respostas"] = avaliar_respostas(acervo, args.respostas)
        resumo = {k: v for k, v in relatorio["respostas"].items() if k != "detalhes"}
        print("Respostas:", json.dumps(resumo, ensure_ascii=False))
    destino = PASTA_DADOS / "avaliacao.json"
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Relatório em {destino} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
