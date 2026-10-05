"""Entendimento das Juntas (tópicos) e legislação (artigos vigentes, citações)."""

import unittest

from assistente.normas import _sem_riscado, citacoes_da_pergunta, paragrafos_texto, topicos_entendimento


class TestEntendimento(unittest.TestCase):
    PARAGRAFOS = [
        "VADE MECUM",
        "As Juntas julgam processos de posturas.",
        "SUMÁRIO",
        "CAPÍTULO VI - Temas de Direito - Pg. 29",
        "1 - Dupla visita. - Pg. 34",
        "CÁPITULO VI – TEMAS DE DIREITO",
        "1 - Dupla visita.",
        "A LC 123/06 prevê a dupla visita.",
        "1.1 - Mesas e cadeiras no logradouro público sem licença",
        "Não se aplica a dupla visita.",
        "I - inciso citado não é tópico;",
        "2 - Órgão público pode ser autuado? " + "Sim, " * 50 + "- com ressalvas.",
    ]

    def test_topicos_com_caminho_e_sem_sumario(self):
        topicos = topicos_entendimento(self.PARAGRAFOS)
        titulos = [t["titulo"] for t in topicos]
        self.assertEqual(titulos[0], "Apresentação")
        self.assertNotIn("Pg.", " ".join(titulos))
        self.assertIn("CAPÍTULO VI – TEMAS DE DIREITO › 1 - Dupla visita. › 1.1 - Mesas e cadeiras no logradouro público sem licença",
                      titulos)
        mesas = topicos[titulos.index(titulos[2])]
        self.assertIn("inciso citado", mesas["texto"])

    def test_titulo_colado_ao_texto(self):
        ultimo = topicos_entendimento(self.PARAGRAFOS)[-1]
        self.assertTrue(ultimo["titulo"].endswith("2 - Órgão público pode ser autuado? " + "Sim, " * 49 + "Sim,"))
        self.assertEqual(ultimo["texto"], "com ressalvas.")

    def test_paragrafos_do_texto_extraido(self):
        self.assertEqual(paragrafos_texto("Um\nparágrafo.\n\n\nOutro."), ["Um parágrafo.", "Outro."])


class TestLegislacao(unittest.TestCase):
    def test_sem_redacao_revogada_nem_notas(self):
        texto = "Art. 1º - Velho texto.\nArt. 1º - Novo texto.\nArt. 1º com redação dada pela Lei nº 9.999, de 1/1/2020\n"
        riscado = [[0, 22]]
        self.assertEqual(_sem_riscado(texto, riscado, 0, len(texto)), "Art. 1º - Novo texto.")

    def test_nota_nao_apaga_paragrafo_de_lei(self):
        texto = "§ 2º - A autorização poderá ser revogada pelo Executivo."
        self.assertEqual(_sem_riscado(texto, [], 0, len(texto)), texto)

    def test_citacoes_na_pergunta(self):
        self.assertEqual(
            citacoes_da_pergunta("O que diz o art. 13 da Lei 8.616/03? E a LEI 9725/09 - ART. 31?"),
            [(("lei", 8616), "13"), (("lei", 9725), "31")],
        )


if __name__ == "__main__":
    unittest.main()
