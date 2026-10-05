"""O tokenizador copiado precisa ser idêntico ao do treino da rede neural (treinar_ia.py do acervo)."""

import importlib.util
import json
import sys
import unittest

from assistente import normas, texto
from assistente.config import ACERVO

TREINO = ACERVO / "treinar_ia.py"
LEGISLACAO = ACERVO / "legislacao.py"
# treinar_ia.py importa legislacao.py, que fica ao lado dele.
sys.path.insert(0, str(ACERVO))


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


@unittest.skipUnless(LEGISLACAO.exists(), f"acervo não encontrado em {ACERVO}")
class TestParidadeLegislacao(unittest.TestCase):
    """As citações dos autos precisam ser lidas como no site (legislacao.py do acervo)."""

    def test_citacoes_identicas(self):
        spec = importlib.util.spec_from_file_location("legislacao", LEGISLACAO)
        legislacao = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legislacao)
        with (ACERVO / "site_data" / "votos.jsonl").open(encoding="utf-8") as fh:
            dispositivos = [a.get("dispositivo_legal_transgredido") or "" for _ in range(500)
                            for a in json.loads(next(fh)).get("autos") or []]
        self.assertTrue(dispositivos)
        for d in dispositivos:
            self.assertEqual(normas.citacoes(d), legislacao.citacoes(d))
        self.assertEqual(normas.chave_norma("Lei", "9.725"), legislacao.chave_norma("Lei", "9.725"))


if __name__ == "__main__":
    unittest.main()
