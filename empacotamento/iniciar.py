"""Ponto de entrada do executável (.exe / AppImage), montado pelo PyInstaller (empacotamento/empacotar.py)."""

import os
import sys

# O PyInstaller aponta LD_LIBRARY_PATH para as bibliotecas que ele embute. Programas abertos daqui
# (navegador, llama-server) precisam das do sistema: volta o valor original antes de abrir qualquer um.
if "LD_LIBRARY_PATH_ORIG" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH_ORIG"]
else:
    os.environ.pop("LD_LIBRARY_PATH", None)

# O OpenSSL embutido procura os certificados no caminho da distribuição em que foi compilado; em outra
# distribuição o HTTPS falharia. Usa o pacote de certificados do sistema (ou o do certifi, embutido).
if sys.platform.startswith("linux") and not os.environ.get("SSL_CERT_FILE"):
    import ssl

    if not os.path.exists(ssl.get_default_verify_paths().cafile or ""):
        for pacote in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                       "/etc/ssl/ca-bundle.pem", "/etc/ssl/cert.pem"):
            if os.path.exists(pacote):
                os.environ["SSL_CERT_FILE"] = pacote
                break
        else:
            try:
                import certifi

                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                pass

# Console do Windows sem UTF-8: um caractere estranho não pode derrubar o programa.
for fluxo in (sys.stdout, sys.stderr):
    if fluxo is not None and hasattr(fluxo, "reconfigure"):
        fluxo.reconfigure(errors="replace")

from assistente.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
