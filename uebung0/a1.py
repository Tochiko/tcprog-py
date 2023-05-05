import pytest

def naive_div(a, b):
    if b == 0:
        raise ZeroDivisionError('division by zero is not defined')
    elif b < 0:
        raise NotImplementedError('division by negative numbers is not implemented')
    r = -1
    while a >= 0:
        a -= b
        r += 1
    return r, b+a

@pytest.mark.parametrize('a, b, expected',
                         [(1, 1, (1, 0)), (3, 2, (1, 1))])
def test_nativ_dif(a, b, expected):
    assert naive_div(a, b) == expected
    with pytest.raises(ZeroDivisionError):
        naive_div(1, 0)
    with pytest.raises(NotImplementedError):
        naive_div(1, -1)