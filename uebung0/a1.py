import pytest


def naive_div(a, b):
    """Returns the quotient of a/b such as the remainder of the division as (quotient, remainder)"""
    if b == 0:
        raise ZeroDivisionError('division by zero is not defined')
    elif b < 0:
        raise NotImplementedError('division by negative numbers is not implemented')
    elif a == 0:
       return 0, 0
    r = -1
    while a >= 0:
        a -= b
        r += 1
    return r, b + a


@pytest.mark.parametrize('a, b, expected',
                         [(1, 1, (1, 0)), (3, 2, (1, 1)), (0, 1, (0, 0))])
def test_nativ_dif(a, b, expected):
    assert naive_div(a, b) == expected
    with pytest.raises(ZeroDivisionError):
        naive_div(1, 0)
    with pytest.raises(NotImplementedError):
        naive_div(1, -1)


def div(a, b):
    """Returns the quotient of a/b such as the remainder of the division as (quotient, remainder)"""
    if b == 0:
        raise ZeroDivisionError('division by zero is not defined')
    elif b < 0:
        raise NotImplementedError('division by negative numbers is not implemented')
    elif a == 0:
        return 0, 0
    n = a.bit_length()
    tmp = b << n
    r = 0
    for _ in range(0, n + 1):
        r <<= 1
        if tmp <= a:
            a -= tmp
            r += 1
        tmp >>= 1
    return r, a

@pytest.mark.parametrize('a, b, expected',
                         [(1, 1, (1, 0)), (3, 2, (1, 1)), (0, 1, (0, 0))])
def test_div(a, b, expected):
    assert div(a,b) == expected
    with pytest.raises(ZeroDivisionError):
        div(1, 0)
    with pytest.raises(NotImplementedError):
        div(1, -1)
