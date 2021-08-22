from numpy import cos, sin, pi, exp, sqrt, abs



def ackley_func(x, y):
    a = 20
    b = 0.2
    c = 2 * pi
    return -a * exp(-b * sqrt(0.5 * sphere(x, y))) \
           - exp(0.5 * (cos(c * x) + cos(c * y))) + a + exp(1)


def ackley_n2_func(x, y):
    return -200 * exp(-0.2 * sqrt(sphere(x, y)))


def ackley_n3_func(x, y):
    return ackley_n2_func(x, y) + (5 * exp(cos(3 * x) + sin(3 * y)))


def ackley_n4_func(x, y):
    return exp(-0.2) * sqrt(sphere(x, y)) + 3 * (cos(2 * x) + sin(2 * y))


def adjiman_func(x, y):
    return cos(x) * sin(y) - (x / (y ** 2 + 1))


def alpine_n1_func(x, y):
    return alpine_n1_helper(x) + alpine_n1_helper(y)


def alpine_n1_helper(x):
    return abs(x * sin(x) + 0.1 * x)


def alpine_n2_func(x, y):
    return alpine_n2_helper(x) * alpine_n2_helper(y)


def alpine_n2_helper(x):
    return sqrt(x) * sin(x)


def beale_func(x, y):
    return (1.5 - x + (x * y)) ** 2 + (2.25 - x + (x * y ** 2)) ** 2 + (2.625 - x + (x * y ** 3)) ** 2


def bartels_conn_func(x, y):
    return abs(sphere(x, y) + (x * y)) + abs(sin(x)) + abs(cos(y))


def bird(x, y):
    return (sin(x) * exp((1 - cos(y)) ** 2)) \
           + (cos(y) * exp((1 - sin(x)) ** 2)) \
           + (x - y) ** 2


def bohachevsky_n1(x, y):
    return (x ** 2) + (2 * y ** 2) - (0.3 * cos(3 * pi * x)) \
        - (0.4 * cos(4 * pi * y)) + 0.7


def bohachevsky_n2(x, y):
    return (x ** 2) + (2 * y ** 2) \
           - (0.3 * cos(3 * pi * x) * cos(4 * pi * y)) + 0.3


def booth(x, y):
    return (x + (2 * y) - 7) ** 2 + ((2 * x) + y - 5) ** 2


def brent(x, y):
    return (x + 10) ** 2 + (y + 10) ** 2 + (exp(-(x ** 2) - (y ** 2)))


def brown(x, y):
    return brown_helper(x, y) + brown_helper(y, x)


def brown_helper(x, y):
    return (x ** 2) ** ((y ** 2) + 1)


def bukin_n6(x, y):
    return 100 * sqrt(abs(y - 0.01 * x ** 2)) + 0.01 * abs(x + 10)


def cross_in_tray(x, y):
    return -0.0001 * (abs(sin(x) * sin(y) * exp(cit_helper(x, y))) + 1) ** 0.1


def cit_helper(x, y):
    return abs(100 - (sqrt(sphere(x, y)) / pi))


def deckkers_aarts(x, y):
    return 10 ** 5 * sphere(x, y) - \
           (sphere(x, y)) ** 2 + \
           10 ** -5 * (sphere(x, y)) ** 4


def drop_wave(x, y):
    return - (1 + cos(12 * sqrt(sphere(x, y)))) / \
           (0.5 * (sphere(x, y)) + 2)


def easom(x, y):
    return -cos(x) * cos(y) * exp(-(x - pi) ** 2 - (y - pi) ** 2)


def egg_crate(x, y):
    return sphere(x, y) + 25 * (sin(x) ** 2 + sin(y) ** 2)


def egg_holder(x, y):
    return -(y + 47) * sin(sqrt(abs(y + x / 2 + 47))) + sin(sqrt(abs(x - (y + 47)))) * (-x)


def exponential(x, y):
    return -exp(-0.5 * (sphere(x, y)))


def goldstein_price(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - (14 * x) + (3 * x ** 2) - 14 * y + (6 * x * y) + (3 * y ** 2))) \
           * (30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + (12 * x ** 2) + 4 * y - (36 * x * y) + (27 * y ** 2)))


def griewank(x, y):
    return 1 + (griewank_sum(x) + griewank_sum(y)) - (griewank_product(x, 1) * griewank_product(y, 2))


def griewank_sum(x):
    return x ** 2 / 4000


def griewank_product(x, i):
    return cos(x / sqrt(i))


def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def holder_table(x, y):
    return -abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(sphere(x, y)) / pi))))


def keane(x, y):
    return -(sin(x - y) ** 2 * sin(x + y) ** 2) / (sqrt(sphere(x, y)))


def leon(x, y):
    return 100 * (y - x ** 3) ** 2 + (1 - x) ** 2


def levi_n13(x, y):
    return sin(3 * pi * x) ** 2 + (x - 1) ** 2 * (1 + sin(3 * pi * y) ** 2) + (y - 1) ** 2 * (1 + sin(2 * pi * y) ** 2)


def matyas(x, y):
    return 0.26 * (sphere(x, y)) - 0.48 * x * y


def mccormick(x, y):
    return sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def michalewicz(x, y):
    return -(mich_helper(x, 1) + mich_helper(y, 2))


def mich_helper(x, i):
    return sin(x) * sin((i * x ** 2) / pi) ** 20


def periodic(x, y):
    return 1 + sin(x) ** 2 + sin(y) ** 2 - 0.1 * exp(-(sphere(x, y)))


def qing(x, y):
    return (x - 1) ** 2 + (y - 2) ** 2


def rastrigin(x, y):
    return 20 + rastrigin_helper(x) + rastrigin_helper(y)


def rastrigin_helper(x):
    return x ** 2 - 10 * cos(2 * pi * x)


def ridge(x, y):
    return x + (y ** 2) ** 0.5


def rosenbrock(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def salomon(x, y):
    v = sqrt(sphere(x, y))
    return 1 - cos(2 * pi * v) + (0.1 * v)


def schaffer_n2(x, y):
    return 0.5 + (sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (sphere(x, y))) ** 2


def schaffer_n3(x, y):
    return 0.5 + (sin(cos(abs(x ** 2 - y ** 2))) ** 2 - 0.5) / (1 + 0.001 * (sphere(x, y))) ** 2


def schwefel_220(x, y):
    return abs(x) + abs(y)


def schwefel_222(x, y):
    return schwefel_220(x, y) + (abs(x) * abs(y))


def schwefel_223(x, y):
    return x ** 10 + y ** 10


def shubert_3(x, y):
    return shubert_3_helper(x) + shubert_3_helper(y)


def shubert_3_helper(x):
    total = 0
    for j in range(1, 6):
        total += j * sin(x * (j + 1) + j)

    return total


def shubert(x, y):
    return shubert_helper(x) * shubert_helper(y)


def shubert_helper(x):
    total = 0
    for j in range(1, 6):
        total += cos(x * (j + 1) + j)
    return total


def sphere(x, y):
    return x ** 2 + y ** 2


def styblinski_tang(x, y):
    return 0.5 * (styblinski_helper(x) + styblinski_helper(y))


def styblinski_helper(x):
    return x ** 4 - (16 * x ** 2) + 5 ** x


def sum_squares(x, y):
    return x ** 2 + (2 * y ** 2)


def three_hump_camel(x, y):
    return (2 * x ** 2) - (1.05 * x ** 4) + (x ** 6 / 6) + x * y + y ** 2


def xin_she_yang_n2(x, y):
    return (abs(x) + abs(y)) * exp(-(sin(x ** 2) + sin(y ** 2)))


def xin_she_yang_n3(x, y):
    return exp(-(xsy3_helper(x) + xsy3_helper(y))) - 2 * exp(-sphere(x, y)) * cos(x) ** 2 * cos(y) ** 2


def xsy3_helper(x):
    return (x / 15) ** 10


def xin_she_yang_n4(x, y):
    return (sin(x) ** 2 + sin(y) ** 2 - exp(-sphere(x, y))) * exp(-(xsy4_helper(x) + xsy4_helper(y)))


def xsy4_helper(x):
    return sin(sqrt(abs(x))) ** 2


def zakharov(x, y):
    return sphere(x, y) + (0.5 * x + y) ** 2 + (0.5 * x + y) ** 4
