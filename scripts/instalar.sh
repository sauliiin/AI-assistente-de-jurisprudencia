#!/usr/bin/env bash
# Instala o que o Assistente de jurisprudência precisa (único passo que usa internet):
#   - o acervo (repositório jurisprudencia-juntas), se ainda não estiver ao lado deste;
#   - llama.cpp (binários prontos: Vulkan para a GPU integrada e CPU como reserva);
#   - o modelo de linguagem Qwen3.5 (GGUF quantizado em 4 bits).
# Modelo e llama.cpp vão para ~/.local/share/assistente-de-jurisprudencia (ou $ASSISTENTE_DADOS), fora do Git.
#
# Uso: scripts/instalar.sh [9b|4b]   (padrão: 9b, responde melhor; 4b é ~1,5x mais rápido e erra mais)
set -euo pipefail

TAMANHO="${1:-9b}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DADOS="${ASSISTENTE_DADOS:-$HOME/.local/share/assistente-de-jurisprudencia}"
ACERVO="${ASSISTENTE_ACERVO:-$(dirname "$RAIZ")/jurisprudencia-juntas}"
VERSAO_LLAMA="b11379"
case "$TAMANHO" in
  4b) MODELO="Qwen3.5-4B-Q4_K_M.gguf"; REPO="unsloth/Qwen3.5-4B-GGUF" ;;
  9b) MODELO="Qwen3.5-9B-Q4_K_M.gguf"; REPO="unsloth/Qwen3.5-9B-GGUF" ;;
  *) echo "Uso: $0 [9b|4b]" >&2; exit 1 ;;
esac

mkdir -p "$DADOS/modelos"

if [ -f "$ACERVO/site_data/votos.jsonl" ]; then
  echo "Acervo encontrado em $ACERVO"
else
  echo "Clonando o acervo em $ACERVO..."
  git clone --depth 1 https://github.com/sauliiin/jurisprudencia-juntas.git "$ACERVO"
fi

baixar_llama() {  # $1 = pasta, $2 = variante do pacote (x64 ou vulkan-x64)
  if ls "$DADOS/$1"/*/llama-server >/dev/null 2>&1; then
    echo "llama.cpp ($2) já instalado em $DADOS/$1"
    return
  fi
  echo "Baixando llama.cpp $VERSAO_LLAMA ($2)..."
  mkdir -p "$DADOS/$1"
  curl -fL --retry 3 -o "$DADOS/llama.tgz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$VERSAO_LLAMA/llama-$VERSAO_LLAMA-bin-ubuntu-$2.tar.gz"
  tar xzf "$DADOS/llama.tgz" -C "$DADOS/$1"
  rm -f "$DADOS/llama.tgz"
}

baixar_llama bin x64
if ls /usr/share/vulkan/icd.d/*.json >/dev/null 2>&1; then
  baixar_llama bin-vulkan vulkan-x64
else
  echo "Sem driver Vulkan: o modelo vai rodar só na CPU (mais lento)."
fi

if [ -f "$DADOS/modelos/$MODELO" ]; then
  echo "Modelo $MODELO já baixado."
else
  echo "Baixando o modelo $MODELO (alguns GB)..."
  curl -fL --retry 3 -C - -o "$DADOS/modelos/$MODELO.part" "https://huggingface.co/$REPO/resolve/main/$MODELO"
  mv "$DADOS/modelos/$MODELO.part" "$DADOS/modelos/$MODELO"
fi
echo "$MODELO" > "$DADOS/modelos/padrao.txt"

echo
echo "Pronto. A partir daqui funciona sem internet:"
echo "  python3 -m assistente --web     # página local"
echo "  python3 -m assistente           # conversa no terminal"
