# Assistente de jurisprudência

Responde perguntas em português sobre as decisões das Juntas Integradas de
Julgamento Fiscal de Belo Horizonte, consultando o
[acervo público](https://github.com/sauliiin/jurisprudencia-juntas) nesta ordem:

1. o **Entendimento das Juntas** (Vade Mecum, `ENTENDIMENTO JUNTAS 2024.docx`);
2. a **legislação** municipal (texto vigente, artigo por artigo);
3. os **pareceres** técnicos;
4. as **decisões** (votos de 1ª e 2ª instância). Roda **sem
internet e sem API paga**: a busca e o modelo de linguagem ficam no computador.

Cada resposta cita as fontes usadas (`[1]`, `[2]`…). A lista de fontes
(protocolo, data, resultado, link) é montada pelo programa a partir dos dados,
não pelo modelo, e o programa confere se as citações da resposta existem.

> **English summary.** Offline question answering over ~12k administrative
> decisions (Belo Horizonte tax/urban-code appeal boards). Hybrid retrieval
> (document BM25 + passage BM25 + a small from-scratch neural encoder, fused
> with RRF) feeds a local LLM (Qwen3.5 9B via llama.cpp on an integrated GPU
> through Vulkan). Exact outcome counts come from structured data, not the LLM;
> citations are validated in code. Pure Python standard library + NumPy/SciPy.

## Como funciona

```text
pergunta
  -> 1º Entendimento das Juntas: o tópico do Vade Mecum que trata da pergunta, se houver
  -> 2º legislação: artigos citados na pergunta, a base legal dos autos das decisões encontradas
     e o artigo que melhor cobre a pergunta (sem a redação revogada)
  -> 3º parecer técnico, quando cobre bem a pergunta
  -> 4º decisões: BM25 da decisão + BM25 do melhor trecho + rede neural do acervo (fusão RRF);
     5 decisões, só com os trechos que importam (de preferência da fundamentação) e o dispositivo
  -> panorama: contagem exata de resultados da infração no acervo inteiro (gabarito do SIF)
  -> modelo local (Qwen3.5 via llama.cpp) escreve a resposta citando as fontes, nessa ordem de autoridade:
     o entendimento prevalece; decisões divergentes são apontadas como divergência
  -> o programa confere se as citações existem
```

| Arquivo | Papel |
|---|---|
| `assistente/acervo.py` | Índice (BM25 por decisão e por trecho, rede neural), busca, panorama |
| `assistente/normas.py` | Lê o Entendimento das Juntas (por tópico) e a legislação (por artigo vigente) |
| `assistente/respondedor.py` | Monta as fontes e o prompt, confere as citações |
| `assistente/llm.py` | Sobe e conversa com o `llama-server` local (só 127.0.0.1); GPU, ou CPU se a GPU falhar |
| `assistente/instalacao.py` | Baixa o que falta (acervo, llama.cpp, modelo) e atualiza o acervo quando muda |
| `assistente/web.py`, `pagina.html` | Página local com a resposta chegando aos poucos |
| `assistente/texto.py` | Normalização de texto; cópia da usada no treino da rede neural |
| `assistente/avaliar.py` | Avaliação com gabaritos tirados do próprio acervo |

## Executável (Windows e Linux)

Para quem só quer usar: baixe da página de Releases do repositório o
`Assistente-de-jurisprudencia.exe` (Windows) ou o
`Assistente-de-jurisprudencia-x86_64.AppImage` (Linux; marque como executável)
e abra com dois cliques. Não precisa instalar Python nem nada.

Na primeira abertura a página mostra o progresso enquanto o programa baixa o
acervo, o llama.cpp e o modelo (9B; 4B se a máquina tiver menos de 12 GB de
RAM). Depois funciona sem internet; com internet, cada abertura confere se o
acervo mudou e baixa só o que mudou. Os dados ficam em
`%LOCALAPPDATA%\assistente-de-jurisprudencia` (Windows) ou
`~/.local/share/assistente-de-jurisprudencia` (Linux).

O botão **Encerrar** da página fecha o programa e libera a memória do modelo; o
programa também encerra sozinho alguns minutos depois que a página é fechada.
No Windows, fechar a janela preta também encerra.

Os executáveis são gerados pelo GitHub Actions (`.github/workflows/empacotar.yml`)
a cada tag `v*` (`git tag v1.0 && git push --tags`), que os publica numa
Release. Para gerar localmente no sistema atual:

```bash
pip install -r requirements.txt pyinstaller certifi
python empacotamento/empacotar.py      # -> dist/
```

## Instalação para desenvolvimento

A instalação é a única etapa que usa internet:

```bash
pip install -r requirements.txt   # numpy e scipy
scripts/instalar.sh               # modelo 9B (padrão, 5,7 GB)
scripts/instalar.sh 4b            # modelo 4B (2,7 GB, ~1,5x mais rápido, erra mais)
```

O script:

- clona o acervo em `../jurisprudencia-juntas` se ele ainda não estiver lá;
- baixa o llama.cpp: a versão Vulkan, para a GPU integrada, e a de CPU, como reserva;
- baixa o modelo para `~/.local/share/assistente-de-jurisprudencia`, fora do Git
  (`python3 -m assistente.instalacao`, o mesmo código que o executável usa).

O último modelo instalado vira o padrão.

## Uso

```bash
python3 -m assistente --web          # página local em http://127.0.0.1:8765
python3 -m assistente                # conversa no terminal ("nova" recomeça, "sair" encerra)
python3 -m assistente "Defesa apresentada fora do prazo é conhecida?"
python3 -m assistente --modelo 4b    # usa o modelo rápido nesta execução
python3 -m assistente --so-busca "cunha de terra no passeio"   # só as fontes, sem o modelo
python3 -m assistente.avaliar --respostas 10                   # mede busca e respostas
python3 -m unittest                                            # testes
```

O que ele sabe fazer:

- **Teses e entendimentos**: "A falta de notificação prévia anula o auto?",
  "Quem assina sem procuração pode recorrer?".
- **Tendência com números exatos**: "Quantas decisões sobre mesas e cadeiras
  foram deferidas?" usa a contagem do acervo inteiro, não a amostra.
- **Caso específico**: citando o protocolo ou o número do auto, ele lê a
  fundamentação inteira daquela decisão e mostra dois casos semelhantes.
- **Filtros na própria pergunta**: "na 2ª instância", "em 2024".
- **Conversa**: perguntas curtas de acompanhamento ("e na 2ª instância?")
  herdam o tema da anterior.
- **Pareceres**: entram quando cobrem bem a pergunta (no máximo um por
  resposta). Para puxá-los, cite "parecer", "DILU", "GESLE" ou "nota
  orientativa" na pergunta.

## Configuração

| Variável | Padrão | Para quê |
|---|---|---|
| `ASSISTENTE_ACERVO` | `../jurisprudencia-juntas` | Onde está o acervo (`site_data/`) |
| `ASSISTENTE_DADOS` | `~/.local/share/assistente-de-jurisprudencia` | Modelos, llama.cpp, cache do índice |
| `ASSISTENTE_ENTENDIMENTO` | o do acervo | Um `.docx` local do Entendimento das Juntas (ex.: versão ainda não publicada) |
| `ASSISTENTE_MODELO` | `modelos/padrao.txt` | Arquivo .gguf a usar |
| `ASSISTENTE_GPU` | `1` | `0` roda só na CPU |
| `ASSISTENTE_THREADS` | automático | Threads da CPU |
| `ASSISTENTE_PORTA_WEB` | `8765` | Porta da página local |

O índice é montado na primeira execução (~20 s) e refeito sozinho quando o
acervo muda. A rede neural usada na busca é a do acervo (`site_data/ia/`,
treinada pelo `treinar_ia.py` de lá). Se estiver desatualizada, a busca segue
só com BM25. O tokenizador de `assistente/texto.py` precisa continuar idêntico
ao do treino; `tests/test_paridade.py` confere isso.

## Desempenho

i5-1235U sem placa de vídeo, usando a GPU integrada (Vulkan), na bateria e no
perfil "Desempenho":

| Modelo | Lê o contexto | Escreve | Tempo por resposta |
|---|---|---|---|
| Qwen3.5 9B (padrão) | ~70 tokens/s | ~5 tokens/s | 1 a 2 min |
| Qwen3.5 4B | ~105 tokens/s | ~8 tokens/s | 40 s a 1 min 15 s |

A GPU integrada lê o contexto ~3x mais rápido que a CPU (4B: 111 contra 33
tokens/s). No modo de economia de energia a CPU fica presa em ~900 MHz e tudo
fica várias vezes mais lento.

## Avaliação

`python3 -m assistente.avaliar`, com gabaritos tirados do próprio acervo:

| Métrica | Resultado |
|---|---|
| Pergunta pelo número do protocolo traz a decisão certa em 1º | 100% (190 protocolos) |
| Perguntas sobre teses (prazo, legitimidade, perda do objeto…): fontes que tratam da tese | 80% das 5 fontes |
| Relato do fiscal → fontes sobre a mesma infração | 55% das 5 fontes (só BM25: 39%) |
| "Qual foi o resultado do protocolo X?" respondido pelo 9B | 8 de 8 corretos, todos citando a fonte, nenhuma citação inventada |

Em testes manuais o 9B respondeu direto e separou o que a defesa alegou do que a
Junta decidiu. O 4B às vezes atribuía à Junta argumentos da defesa. Para não
trocar "deferido" por "indeferido", o panorama vai ao modelo com as categorias
em ordem fixa e zeros explícitos. Mesmo assim é um modelo pequeno: confira a
decisão citada antes de usar a resposta.
