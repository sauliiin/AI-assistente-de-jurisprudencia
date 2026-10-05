"""
Gera o executável do assistente para o sistema em que roda:

  - Windows: dist/Assistente-de-jurisprudencia.exe (arquivo único);
  - Linux:   dist/Assistente-de-jurisprudencia-x86_64.AppImage.

Nenhum dos dois leva o modelo nem o acervo: na 1ª abertura o próprio programa baixa o que falta
(assistente/instalacao.py) e mostra o progresso na página.

    pip install -r requirements.txt pyinstaller
    python empacotamento/empacotar.py

O GitHub Actions (.github/workflows/empacotar.yml) roda este script no Windows e no Linux.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
AQUI = RAIZ / "empacotamento"
OBRA = RAIZ / "build"
SAIDA = RAIZ / "dist"
NOME = "Assistente-de-jurisprudencia"
APPIMAGETOOL = "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"

DESKTOP = """[Desktop Entry]
Type=Application
Name=Assistente de jurisprudência
Comment=Perguntas sobre as decisões das Juntas Integradas de Julgamento Fiscal, sem internet
Exec={nome}
Icon=assistente-de-jurisprudencia
Categories=Office;
Terminal=false
"""

APPRUN = """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/assistente/{nome}" "$@"
"""


def pyinstaller(arquivo_unico: bool) -> Path:
    separador = ";" if sys.platform == "win32" else ":"
    comando = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", NOME,
        "--onefile" if arquivo_unico else "--onedir",
        "--console",  # no Windows, a janela mostra o andamento; fechá-la encerra o assistente
        "--icon", str(AQUI / "icone.ico"),
        "--add-data", f"{RAIZ / 'assistente' / 'pagina.html'}{separador}assistente",
        "--exclude-module", "tkinter",
        "--distpath", str(OBRA / "dist"),
        "--workpath", str(OBRA / "work"),
        "--specpath", str(OBRA),
        "--paths", str(RAIZ),
    ]
    if sys.platform == "win32":
        # O llama.cpp do Windows precisa do runtime do Visual C++, que nem todo Windows tem:
        # vai uma cópia no .exe, posta ao lado do llama-server.exe na 1ª abertura (assistente/instalacao.py).
        for dll in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            caminho = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / dll
            if not caminho.exists():
                raise SystemExit(f"{caminho} não encontrado (instale o Visual C++ Redistributable para empacotar).")
            comando += ["--add-binary", f"{caminho}{separador}vcruntime"]
    comando.append(str(AQUI / "iniciar.py"))
    subprocess.run(comando, check=True, cwd=RAIZ)
    return OBRA / "dist" / (f"{NOME}.exe" if arquivo_unico else NOME)


def appimage(pasta_pyinstaller: Path) -> Path:
    appdir = OBRA / "AppDir"
    shutil.rmtree(appdir, ignore_errors=True)
    shutil.copytree(pasta_pyinstaller, appdir / "usr" / "lib" / "assistente", symlinks=True)
    (appdir / "AppRun").write_text(APPRUN.format(nome=NOME))
    (appdir / "AppRun").chmod(0o755)
    (appdir / "assistente-de-jurisprudencia.desktop").write_text(DESKTOP.format(nome=NOME))
    shutil.copy(AQUI / "icone.png", appdir / "assistente-de-jurisprudencia.png")

    ferramenta = OBRA / "appimagetool"
    if not ferramenta.exists():
        print("Baixando o appimagetool...", flush=True)
        urllib.request.urlretrieve(APPIMAGETOOL, ferramenta)
        ferramenta.chmod(ferramenta.stat().st_mode | stat.S_IEXEC)
    destino = SAIDA / f"{NOME}-x86_64.AppImage"
    # APPIMAGE_EXTRACT_AND_RUN: roda sem FUSE (contêineres e GitHub Actions).
    env = {**os.environ, "ARCH": "x86_64", "APPIMAGE_EXTRACT_AND_RUN": "1"}
    subprocess.run([str(ferramenta), "--no-appstream", str(appdir), str(destino)], check=True, env=env)
    return destino


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    if sys.platform == "win32":
        exe = pyinstaller(arquivo_unico=True)
        destino = SAIDA / exe.name
        shutil.copy(exe, destino)
    elif sys.platform.startswith("linux"):
        destino = appimage(pyinstaller(arquivo_unico=False))
    else:
        raise SystemExit("Empacotamento só para Windows (.exe) e Linux (AppImage).")
    print(f"\nPronto: {destino} ({destino.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
