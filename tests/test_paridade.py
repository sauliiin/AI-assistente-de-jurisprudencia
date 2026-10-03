"""O tokenizador copiado precisa ser idêntico ao do treino da rede neural (treinar_ia.py do acervo)."""

import importlib.util
import json
import unittest

from assistente import texto
from assistente.config import ACERVO

TREINO = ACERVO / "treinar_ia.py"


@unittest.skipUnless(TREINO.exists(), f"acervo não encontrado em {ACERVO}")
class TestParidade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("treinar_ia", TREINO)
        cls.treino = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.treino)
        with (ACERVO / "site_data" / "votos.jsonl").open(encoding="utf-8") as fh:
            cls.textos = [json.loads(next(fh))["texto"] for _ in range(200)]

    def test_tokenizar_identico(self):
        for t in self.textos:
            self.assertEqual(texto.tokenizar(t), self.treino.tokenizar(t))

    def test_resultado_identico(self):
        for t in self.textos:
            self.assertEqual(texto.resultado_da_decisao(t), self.treino.resultado_da_decisao(t))

    def test_chave_infracao_identica(self):
        nome = "DEIXAR DE CONSTRUIR, MANTER OU CONSERVAR EM PERFEITO ESTADO O PASSEIO"
        self.assertEqual(texto.chave_infracao(nome), self.treino.chave_infracao(nome))


if __name__ == "__main__":
    unittest.main()
