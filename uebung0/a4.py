import pytest
from functools import lru_cache


def lucas_number(n):
    if n <= 0:
        return 2
    elif n == 1:
        return 1
    return lucas_number(n - 1) + lucas_number(n - 2)

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
@lru_cache(maxsize=32)
def cached_lucas_number(n):
    if n <= 0:
        return 2
    elif n == 1:
        return 1
    return lucas_number(n - 1) + lucas_number(n - 2)

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