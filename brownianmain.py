import ctypes

fib_lib = ctypes.CDLL('./parellbrownian.so')
fib = fib_lib.capture
fib.restype = ctypes.c_int
fib.argtypes = [ctypes.c_int]

print(capture(10,1))