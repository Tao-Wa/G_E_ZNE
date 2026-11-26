import numpy as np
# 外推函数
'''
def linear_extrapolation(x, a, b, c, d, f):  # , c):
    """线性外推函数"""
    e = 2 * d
    # return a + b * x  # + c * x ** 2
    # return (a + b * (x)) * np.exp(c * x + d * (x ** 2))
    # return a * np.exp(c * x + d * (x ** 2)) + b * x
    return (a + (b + f * x) * (np.sqrt(np.exp(e * (x ** 2)) - 1))) * np.exp(c * x + d * (x ** 2))
    # return b * np.exp(c * x + d * (x ** 2)) + a
'''
def poly_exp_extrapolation(x, a, b, c, d, e):  # , c):
    """多指数外推"""
    f = 2 * b
    return (a + (d + e*x) * (np.sqrt(np.exp(f * (x ** 2)) - 1))) * np.exp(c * x + b * (x ** 2))

def poly_exp_extrapolation1(x, a, b, c, d,e):
    """多指数外推"""
    f = 2 * b
    return (a + d * (np.sqrt(np.exp(f * (x ** 2)) - 1))) * np.exp(c * x + b * (x ** 2))+e

def poly_exp_extrapolation2(x, a, b, c, d, e):  # , c):
    """多指数外推"""
    return a * np.exp(c * x + b * (x ** 2)) + (d+e*x)* np.exp(c * x)

def poly_exp_extrapolation3(x, a, b, c,d, e):  # , c):
    """多指数外推"""
    return (a+e*x) * np.exp(c * x + b * (x ** 2)) + d

def poly_exp_extrapolation4(x, a,b, c, e):  # , c):
    """多指数外推"""
    return (a+e*x) * np.exp(c * x + b * (x ** 2))



def linear_extrapolation1(x, a, b, c, d):  # , c):
    """线性外推函数"""
    # return a + b * x  # + c * x ** 2
    # return (a + b * (x)) * np.exp(c * x + d * (x ** 2))
    # return a * np.exp(c * x + d * (x ** 2)) + b * x
    return (a + b * (x ** 2)) * np.exp(c * x + d * (x ** 2))
    # return b * np.exp(c * x + d * (x ** 2)) + a

def exp_extrapolation(x, a, b,c):
    """二次外推函数"""
    return a * (b ** x) + c

def exp_extrapolation1(x, a, b, c,d,e):
    """二次外推函数"""
    return a * (b ** x) + c * (d ** x) + e
def fid_extrapolation(x, a, b):
    """线性外推函数"""
    return a + b * x


def purity_sqrt_extrapolation(x, a, b, c):
    # return a * np.sqrt((x - b) / (1 - b))
    #b = 0.0625
    return a * np.exp(1 / 2 * np.log((x - b) / (1 - b))) + c

'''
def purity_sqrt_extrapolation(x, a, b, c):
    # p = a * noise_value^(b) +c
    return a * (np.abs(x) ** b) + c
'''
