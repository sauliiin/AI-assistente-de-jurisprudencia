"""
Assistente de jurisprudência.

    python3 -m assistente                      # conversa no terminal
    python3 -m assistente "pergunta"           # uma pergunta e sai
    python3 -m assistente --web                # página local em http://127.0.0.1:8765
    python3 -m assistente --so-busca "pergunta"  # mostra as fontes, sem o modelo de linguagem

Tudo roda nesta máquina: o índice do acervo, a busca e o modelo de linguagem
(llama.cpp). A internet só é usada para baixar o que falta (na 1ª vez: acervo,
llama.cpp e modelo) e para atualizar o acervo quando ele muda.

O executável (.exe/AppImage) sem argumentos abre a página local.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .acervo import Acervo
from .config import EMPACOTADO, resolver_modelo
from .instalacao import ErroInstalacao, instalar
from .llm import ErroLLM, ModeloLocal
from .respondedor import Assistente, Conversa

CINZA, NEGRITO, AZUL, FIM = "\033[90m", "\033[1m", "\033[36m", "\033[0m"


def _imprimir_fontes(evento: dict) -> None:
    print(f"{CINZA}Fontes consultadas (busca em {evento['busca_s']} s):")
    for f in evento["fontes"]:
        if f["tipo"] == "panorama":
            print(f"  [{f['numero']}] Panorama do acervo")
            print(textwrap.indent(f["texto"], "       │ "))
            continue
        if f["tipo"] in ("entendimento", "legislacao", "parecer"):
            rotulo = {"entendimento": "Entendimento das Juntas", "legislacao": "Legislação", "parecer": "Parecer"}[f["tipo"]]
            print(f"  [{f['numero']}] {rotulo}: {f['titulo']}")
            continue
        extra = f" — {f['motivo']}" if f["motivo"] else ""
        data = "/".join(reversed(f["data"].split("-"))) if f["data"] else "data ?"
        infr = (f["infracoes"][0][:70] + "…") if f["infracoes"] else ""
        print(f"  [{f['numero']}] {f['protocolo']} · {f['instancia']} · {data} · {f['resultado'] or 'resultado ?'}{extra}")
        if infr:
            print(f"       {infr}")
    print(FIM, flush=True)


def _rodar(assistente: Assistente, pergunta: str, conversa: Conversa | None, pensar: bool) -> None:
    for evento in assistente.responder(pergunta, conversa, pensar=pensar):
        if evento["tipo"] == "fontes":
            _imprimir_fontes(evento)
        elif evento["tipo"] == "texto":
            print(evento["texto"], end="", flush=True)
        elif evento["tipo"] == "fim":
            print()
            if evento["citacoes_invalidas"]:
                print(f"{NEGRITO}[atenção] citou fontes inexistentes: {evento['citacoes_invalidas']}{FIM}")
            elif not evento["citadas"]:
                print(f"{NEGRITO}[atenção] a resposta não citou nenhuma fonte; confira antes de usar.{FIM}")
            print(
                f"{CINZA}{evento['segundos']} s · contexto {evento['tokens_contexto']} tokens "
                f"({evento['leitura_tps']} tok/s) · resposta {evento['tokens_resposta']} tokens "
                f"({evento['escrita_tps']} tok/s){FIM}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m assistente", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pergunta", nargs="*")
    parser.add_argument("--web", action="store_true", help="abre a página local")
    parser.add_argument("--porta", type=int, help="porta da página local (padrão 8765)")
    parser.add_argument("--nao-abrir", action="store_true", help="com --web, não abre o navegador")
    parser.add_argument("--modelo", help="arquivo .gguf ou parte do nome (ex.: 4b para o modelo rápido)")
    parser.add_argument("--pensar", action="store_true", help="deixa o modelo raciocinar antes (mais lento, às vezes melhor)")
    parser.add_argument("--so-busca", action="store_true", help="só mostra as fontes encontradas")
    parser.add_argument("--fontes", type=int, default=5, help="quantas decisões enviar ao modelo (padrão 5)")
    args = parser.parse_args()
    if EMPACOTADO and not args.pergunta and not args.so_busca:
        args.web = True  # aberto com dois cliques: a página

    if args.web:
        from .web import servir

        def preparar(progresso) -> Assistente:
            instalar(progresso=progresso)
            progresso("Montando o índice do acervo (só quando o acervo muda)…", None)
            acervo = Acervo.carregar()
            modelo = ModeloLocal(resolver_modelo(args.modelo) if args.modelo else None)
            progresso(f"Carregando o modelo de linguagem {modelo.nome}…", None)
            modelo.garantir()
            return Assistente(acervo, modelo, n_fontes=args.fontes)

        servir(preparar, args.porta, abrir=not args.nao_abrir, encerrar_sem_pagina=EMPACOTADO)
        return

    try:
        instalar()
    except ErroInstalacao as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        sys.exit(1)
    acervo = Acervo.carregar()
    modelo = ModeloLocal(resolver_modelo(args.modelo) if args.modelo else None)
    assistente = Assistente(acervo, modelo, n_fontes=args.fontes)

    if args.so_busca:
        fontes = assistente.preparar(" ".join(args.pergunta))
        _imprimir_fontes({"fontes": [f.resumo() for f in fontes], "busca_s": "-"})
        for f in fontes:
            print(f.para_o_modelo(), "\n")
        return

    try:
        modelo.garantir()
        if args.pergunta:
            _rodar(assistente, " ".join(args.pergunta), None, args.pensar)
            return
        conversa = Conversa()
        print(f"{AZUL}Assistente de jurisprudência · modelo {modelo.nome}. "
              f"Pergunte em português; 'nova' recomeça a conversa, 'sair' encerra.{FIM}\n")
        while True:
            try:
                pergunta = input(f"{NEGRITO}Você:{FIM} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not pergunta:
                continue
            if pergunta.lower() in {"sair", "exit", "quit"}:
                break
            if pergunta.lower() == "nova":
                conversa = Conversa()
                print(f"{CINZA}Conversa recomeçada.{FIM}\n")
                continue
            _rodar(assistente, pergunta, conversa, args.pensar)
    except ErroLLM as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
