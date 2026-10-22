"""Funções de comparação usadas nos testes e nas medições."""

import numpy as np


def erro_l1(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """
    Distância L1: soma das diferenças absolutas, posição por posição.
    É a nossa "régua" para dizer se dois PageRanks são iguais.
    """
    # o NumPy subtrai os dois vetores posição por posição, np.abs tira o sinal
    # e .sum() soma tudo. Ex.: [0,5; 0,3; 0,2] e [0,4; 0,4; 0,2]
    # -> diferenças [0,1; -0,1; 0] -> absolutas [0,1; 0,1; 0] -> erro 0,2.
    # float() converte o número do NumPy em um float comum do Python
    return float(np.abs(vetor_a - vetor_b).sum())