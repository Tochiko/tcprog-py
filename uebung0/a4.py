import time
import pytest
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# a)--------------------------------------------------------------------------------------------------------------------
def lucas_number(n):
    if n <= 0:
        return 2
    elif n == 1:
        return 1
    return lucas_number(n - 1) + lucas_number(n - 2)
# Ju: verstehe den Aufbau der Funktion nicht. Ich hätte das so gemacht:
    # if n <= 0:
    #     return 2
    # elif n == 1:
    #     return 1
    # eliif n > 1:
    #   return cached_lucas_number(n - 1) + cached_lucas_number(n - 2)
# Warum ist mein Aufbau evtl. nicht möglich oder was macht ihn weniger schön 
# oder effizient als den hier?

"""
Unter einem Cache x wird ein Zwischenspeicher für sich wiederholende Verarbeitungen des Types x verstanden. In der Regel werden die 
Ergebnisse der Verarbeitungen des Types x in der Form Key-Value bzw. Input-Output im Arbeitsspeicher zwischengespeichert. 
Bevor eine neue Verarbeitung des Types x gestartet wird, wird zunächst im Cache x nachgesehen, ob die Verarbeitung mit dem
entsprechenden Input bereits gecached ist. Ist die Vearbeitung gecached, so kann das Ergebnis einfach ausgelesen werden,
statt durch die Vearbeitung des Types x neu erzeugt zu werden.

In der Praxis haben Caches eine Zahl maximaler Einträge, so dass der Speicherbedarf nicht endlos wächst - Oder aber, die 
gecaschten Einträge haben eine Lebenszeit von Sekunden bis Tage.

Im Fall der Rekursion bedeutet das: 1. Aufruf cached_lucas_number(3), dann 2. Aufruf cached_lucas_number(4) -> ...(3)+(2)
diese sind aber bereits durch den vorherigen Aufruf berechnet worden und im Cache zwischengespeichert. Für den Fall großer
rekursiver Folgen kann dies die Bearbeitungszeit signifikant reduzieren.

Ein anderes Beispiel, in dem Caches verwendet werden, um die Last der Infrastruktur zu reduzieren ist das WWW. Im Web werden
die Ressourcen für gewöhnlich im Browser des Aufrufers für eine gewisse Zeit zwischengespeichert. Dabei referenziert die URI
eineindeutig eine Ressource, die zwischengespeichert wird. Bei einem Reload der URI werden die Ressourcen also nicht zwangs-
läufig neu geladen, sondern nur dann, wenn der Cache abgelaufen ist - Falls gecached wird. Nicht alle Ressourcen lassen
sich sinnvoll cachen.
"""


@lru_cache(maxsize=4)
def cached_lucas_number(n):
    if n <= 0:
        return 2
    elif n == 1:
        return 1
    return cached_lucas_number(n - 1) + cached_lucas_number(n - 2)


@pytest.mark.parametrize('n, expected',
                         [(0, 2), (1, 1), (2, 3), (3, 4), (4, 7)])
def test_lucas_number(n, expected):
    lucas_num = lucas_number(n)
    assert lucas_num == expected
    assert type(lucas_num) is int


@pytest.mark.parametrize('n, expected',
                         [(0, 2), (1, 1), (2, 3), (3, 4), (4, 7)])
def test_cached_lucas_number(n, expected):
    lucas_num = cached_lucas_number(n)
    assert lucas_num == expected
    assert type(lucas_num) is int


# b)--------------------------------------------------------------------------------------------------------------------

size = 32
values_input = np.arange(0, size, 1, dtype=int)
time_lucas_numbers = []
for n in range(size):
    start = time.time()
    lucas_number(n)
    time_lucas_numbers.append(time.time() - start)

time_cached_lucas_numbers = []
for n in range(size):
    start = time.time()
    cached_lucas_number(n)
    time_cached_lucas_numbers.append(time.time() - start)

fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].plot(values_input, np.array(time_lucas_numbers))
# axs[0].set_xlabel(r"lucas_number(n)")
axs[0].set_xlabel(r"n") # Ju: Vorschlag zur Achsenbeschriftung, weil eig
# ist auf der x-Achse nicht die Lucasnummer sondern nur das n aufgetragen
axs[0].set_ylabel(r"$\Delta t$ [s]")

axs[1].plot(values_input, np.array(time_cached_lucas_numbers) * 1e6, color='crimson')
axs[1].set_xlabel(r"n (cached)")
axs[1].set_ylabel(r"$\Delta t$ [$10^{-6}$ s]")

plt.show()


# c)-------------------------------------------------------------------------------------------------
def hermite(n):
    z = sp.Symbol('z')
    if n <= 0:
        return sp.sympify(1)
    elif n == 1:
        return sp.sympify(2) * z
    return sp.sympify(2) * z * hermite(n - 1) - sp.sympify(2) * n * hermite(n - 2)


@pytest.mark.parametrize('n, z, expected',
                         [(0, 0, 1), (1, 1, 2), (2, 1, 0), (3, 1, -12), (3, 2, 24), (3, 3, 156)])
def test_hermite(n, z, expected):
    f = sp.lambdify(sp.Symbol('z'), hermite(n))
    assert f(z) == expected
# H(n=3)(z) = 8z**3 - 20z