import random

SYMS = ['+', '-', '*', '/']  

def gen_ops(total_q: int):
    # Generate random soal
    qs = []
    while len(qs) < total_q:
        op = len(qs) % 4
        a, b = random.randint(1, 9), random.randint(1, 9)
        if op == 0:
            c = a + b
        elif op == 1:
            c = a - b
        elif op == 2:
            c = a * b
        else:
            b = random.randint(1, 9); c = random.randint(1, 5); a = b * c  
        if 1 <= c <= 5:
            qs.append((f"{a} {SYMS[op]} {b} = ?", c))
    random.shuffle(qs)
    return qs

# Validasi nilai yang diberikan oleh jari
def is_valid_fingers(val: int) -> bool:
    return 1 <= val <= 5

# Cek jawaban benar atau salah
def evaluate_answer(user_val: int, correct_val: int) -> bool:
    return user_val == correct_val
