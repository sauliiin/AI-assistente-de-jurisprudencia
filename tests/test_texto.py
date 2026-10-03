"""Testes das partes determinísticas: termos de busca, datas, trechos, protocolos."""

import unittest

from assistente.acervo import dividir_em_trechos, expandir, extrair_data, filtros_da_pergunta, limpar_assunto, termos
from assistente.texto import RESULTADOS, resultado_da_decisao


class TestTermos(unittest.TestCase):
    def test_sem_acento_sem_plural_e_radical(self):
        self.assertEqual(termos("Cadeiras e MESAS no passeio"), ["cadeira", "mesa", "passeio"])

    def test_expande_teses_processuais(self):
        self.assertIn("intempestividade", expandir("defesa fora do prazo"))


class TestExtracao(unittest.TestCase):
    def test_data_da_sessao(self):
        texto = "JIJFI-III\n350ª SESSÃO ORDINÁRIA - 11/02/2026\nImpugnação"
        self.assertEqual(extrair_data(texto, "31.00960807/2025-31"), "2026-02-11")

    def test_data_anterior_ao_protocolo_e_descartada(self):
        texto = "SESSÃO ORDINÁRIA - 17/12/2016\n..."
        self.assertEqual(extrair_data(texto, "31.00334321-2025-02"), "")

    def test_assunto_sem_cabecalho(self):
        bruto = "31.00960807/2025-31 Cancelamento: autos 2025AN AUTUADO(S): CONDOMÍNIO"
        self.assertEqual(limpar_assunto(bruto), "Cancelamento: autos 2025AN")

    def test_trechos_respeitam_tamanho(self):
        texto = "\n".join(f"Frase número {i} da fundamentação da decisão." for i in range(200))
        trechos = dividir_em_trechos(texto, alvo=650)
        self.assertGreater(len(trechos), 5)
        self.assertTrue(all(len(t) <= 650 * 1.6 for t in trechos))

    def test_resultado_do_dispositivo(self):
        texto = "Relatório...\nDispositivo da DECISÃO\nPelo exposto, CONHEÇO DA DEFESA, DEFIRO PARCIALMENTE o pedido."
        self.assertEqual(RESULTADOS[resultado_da_decisao(texto)], "parcialmente deferido")


class TestFiltros(unittest.TestCase):
    def test_instancia_e_ano(self):
        self.assertEqual(filtros_da_pergunta("e na 2ª instância em 2024?"), {"instancia": "2ª", "ano": "2024"})

    def test_ano_dentro_do_protocolo_nao_e_filtro(self):
        self.assertEqual(filtros_da_pergunta("resultado do protocolo 31.00960807/2025-31 e do auto 20250036024AI?"), {})


if __name__ == "__main__":
    unittest.main()
