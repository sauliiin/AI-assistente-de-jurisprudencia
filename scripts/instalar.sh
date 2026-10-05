#!/usr/bin/env bash
# Prepara o ambiente de desenvolvimento do Assistente de jurisprudência (único passo que usa internet):
#   - clona o acervo (repositório jurisprudencia-juntas) ao lado deste, se ainda não estiver lá;
#   - baixa o llama.cpp e o modelo de linguagem (assistente/instalacao.py, o mesmo código que o
#     executável usa na primeira abertura).
# Modelo e llama.cpp vão para ~/.local/share/assistente-de-jurisprudencia (ou $ASSISTENTE_DADOS), fora do Git.
#
# Uso: scripts/instalar.sh [9b|4b]   (padrão: 9b, responde melhor; 4b é ~1,5x mais rápido e erra mais)
set -euo pipefail

TAMANHO="${1:-9b}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACERVO="${ASSISTENTE_ACERVO:-$(dirname "$RAIZ")/jurisprudencia-juntas}"

if [ -f "$ACERVO/site_data/votos.jsonl" ]; then
  echo "Acervo encontrado em $ACERVO"
else
  echo "Clonando o acervo em $ACERVO..."
  git clone --depth 1 https://github.com/sauliiin/jurisprudencia-juntas.git "$ACERVO"
fi

cd "$RAIZ"
python3 -m assistente.instalacao "$TAMANHO"

echo
echo "A partir daqui funciona sem internet:"
echo "  python3 -m assistente --web     # página local"
echo "  python3 -m assistente           # conversa no terminal"
