########################################
# codegen/generate_commutator_code.py
########################################

import sympy as sp
from sympy import expand, simplify, re, im

# ===== Density matrix entries =====
r0, r1, r2, r3 = sp.symbols("r0 r1 r2 r3", real=True)
r01, i01 = sp.symbols("r01 i01", real=True)
r02, i02 = sp.symbols("r02 i02", real=True)
r03, i03 = sp.symbols("r03 i03", real=True)
r12, i12 = sp.symbols("r12 i12", real=True)
r13, i13 = sp.symbols("r13 i13", real=True)
r23, i23 = sp.symbols("r23 i23", real=True)

rho_in_0, rho_in_1, rho_in_2, rho_in_3 = sp.symbols("rho_in_0 rho_in_1 rho_in_2 rho_in_3", real=True)
rho_in_4, rho_in_5 = sp.symbols("rho_in_4 rho_in_5", real=True)
rho_in_6, rho_in_7 = sp.symbols("rho_in_6 rho_in_7", real=True)
rho_in_8, rho_in_9 = sp.symbols("rho_in_8 rho_in_9", real=True)
rho_in_10, rho_in_11 = sp.symbols("rho_in_10 rho_in_11", real=True)
rho_in_12, rho_in_13 = sp.symbols("rho_in_12 rho_in_13", real=True)
rho_in_14, rho_in_15 = sp.symbols("rho_in_14 rho_in_15", real=True)

rho = sp.Matrix([
    [rho_in_0,               rho_in_4+sp.I*rho_in_5,    rho_in_6+sp.I*rho_in_7,   rho_in_8+sp.I*rho_in_9  ],
    [rho_in_4-sp.I*rho_in_5, rho_in_1,                 rho_in_10+sp.I*rho_in_11, rho_in_12+sp.I*rho_in_13],
    [rho_in_6-sp.I*rho_in_7, rho_in_10-sp.I*rho_in_11, rho_in_2,                 rho_in_14+sp.I*rho_in_15],
    [rho_in_8-sp.I*rho_in_9, rho_in_12-sp.I*rho_in_13, rho_in_14-sp.I*rho_in_15, rho_in_3                ]
])




def is_hermitian_matrix_elementwise(M):
    """
    Checks if a SymPy matrix is Hermitian by comparing each element with its conjugate transpose counterpart.
    
    A matrix M is Hermitian if M[i, j] == conjugate(M[j, i]) for all i, j.
    """
    rows, cols = M.shape
    if rows != cols:
        return False  # Must be square

    for i in range(rows):
        for j in range(cols):
            if not sp.simplify(M[i, j] - sp.conjugate(M[j, i])) == 0:
                return False
    return True



print(is_hermitian_matrix_elementwise(rho))











eps_t_substep = sp.symbols('eps_t_substep', real=True)
delta_C, B, one_div_m = sp.symbols('delta_C B one_div_m', real=True, positive=True)


# Diabatic Hamiltonian
H = - sp.Rational(1,2) * sp.Matrix([
    [ -B*eps_t_substep - one_div_m,  0,          0,              0                         ],
    [       0,                    eps_t_substep, delta_C,        0                         ],
    [       0,                    delta_C,    -eps_t_substep,    0                         ],
    [       0,                    0,          0,              B*eps_t_substep - one_div_m  ]
])





# Commutator [H, rho] = -I*(H*rho - rho*H)
comm = -sp.I*(H*rho - rho*H)


print(is_hermitian_matrix_elementwise(comm))


def print_cuda_style(expr, output_var):
    """Print expression in CUDA accumulation style."""
    # Force full expansion WITHOUT simplifying first
    expanded = sp.expand(expr)
    
    if expanded == 0:
        print(f"{output_var} = 0.0f;")
        return
    
    # Check if it's a simple product (no additions)
    if not expanded.is_Add:
        # Single term, just print it
        print(f"{output_var} = {sp.ccode(expanded)};")
        return
    
    terms = sp.Add.make_args(expanded)
    
    # Check if any term has 1/2 - check both Rational(1,2) and Float(0.5)
    has_half = any(
        sp.Rational(1,2) in t.atoms() or 
        sp.Rational(-1,2) in t.atoms() or
        any(isinstance(a, sp.Rational) and abs(a) == sp.Rational(1,2) for a in t.atoms())
        for t in terms
    )
    
    if has_half:
        # Multiply by 2 to remove all 1/2 factors
        inner = sp.expand(expanded * 2)
        inner_terms = sp.Add.make_args(inner)
        
        print("tmp = 0.0f;")
        for term in inner_terms:
            print(f"tmp += {sp.ccode(term)};")
        print("tmp *= 0.5f;")
        print(f"{output_var} = tmp;")
    else:
        # No 1/2, just print directly
        print(f"{output_var} = {sp.ccode(expanded)};")


def print_commutator_cuda():
    """Print the full CUDA-style commutator code"""
    
    # Diagonals
    for i in range(4):
        print_cuda_style(sp.re(comm[i, i]), f"drho_out_{i}")
        print()
    
    # Off-diagonals
    idx = 4
    for i in range(4):
        for j in range(i+1, 4):
            print(f"// ({i},{j})")
            print_cuda_style(sp.re(comm[i, j]), f"drho_out_{idx}")
            print()
            print_cuda_style(sp.im(comm[i, j]), f"drho_out_{idx+1}")
            print()
            idx += 2


# =====================================
# VERIFICATION: Paste your CUDA code here as Python
# =====================================

def cuda_code_as_python():
    """
    This is CUDA code translated directly to Python.
    Copy-paste from CUDA and change syntax minimally:
    - Change 0.0f to 0
    - Change 0.5f to sp.Rational(1,2)
    """
    
    drho_out_0 = 0;

    tmp = delta_C * rho_in_11;

    drho_out_1 = tmp;

    drho_out_2 = -tmp;

    drho_out_3 = 0;

    #// (0,1)
    tmp = 0;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_5;

    tmp += delta_C * rho_in_7;
    tmp *= sp.Rational(1,2);
    drho_out_4 = tmp;

    tmp = 0;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_4;

    tmp += -delta_C * rho_in_6;
    tmp *= sp.Rational(1,2);
    drho_out_5 = tmp;

    #// (0,2)
    tmp = 0;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_7;

    tmp += delta_C * rho_in_5;
    tmp *= sp.Rational(1,2);
    drho_out_6 = tmp;

    tmp = 0;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_6;

    tmp += -delta_C * rho_in_4;
    tmp *= sp.Rational(1,2);
    drho_out_7 = tmp;

    #// (0,3)
    drho_out_8 = B * eps_t_substep * rho_in_9;

    drho_out_9 = -B * eps_t_substep * rho_in_8;

    #// (1,2)
    drho_out_10 = -eps_t_substep * rho_in_11;

    tmp = 0;
    tmp = -delta_C * rho_in_1;
    tmp += delta_C * rho_in_2;
    tmp += 2 * eps_t_substep * rho_in_10;
    tmp *= sp.Rational(1,2);
    drho_out_11 = tmp;

    #// (1,3)
    tmp = 0;
    tmp = -eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_13;

    tmp += -delta_C * rho_in_15;
    tmp *= sp.Rational(1,2);
    drho_out_12 = tmp;

    tmp = 0;
    tmp = eps_t_substep;
    tmp += -B * eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_12;

    tmp += delta_C * rho_in_14;
    tmp *= sp.Rational(1,2);
    drho_out_13 = tmp;

    #// (2,3)
    tmp = 0;
    tmp = eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_15;

    tmp += -delta_C * rho_in_13;
    tmp *= sp.Rational(1,2);
    drho_out_14 = tmp;


    tmp = 0;
    tmp = -B * eps_t_substep;
    tmp += -eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_14;

    tmp += delta_C * rho_in_12;
    tmp *= sp.Rational(1,2);
    drho_out_15 = tmp;
    
    return [drho_out_0, drho_out_1, drho_out_2, drho_out_3,
            drho_out_4, drho_out_5, drho_out_6, drho_out_7,
            drho_out_8, drho_out_9, drho_out_10, drho_out_11,
            drho_out_12, drho_out_13, drho_out_14, drho_out_15]


def verify_cuda_code():
    """Compare CUDA code against SymPy reference"""
    
    # Get CUDA results
    cuda_results = cuda_code_as_python()
    
    # Get SymPy reference
    sympy_results = []
    for i in range(4):
        sympy_results.append(sp.re(comm[i, i]))
    
    idx = 4
    for i in range(4):
        for j in range(i+1, 4):
            sympy_results.append(sp.re(comm[i, j]))
            sympy_results.append(sp.im(comm[i, j]))
    
    # Compare
    print("="*60)
    print("VERIFICATION RESULTS:")
    print("="*60)
    
    all_correct = True
    for i, (cuda_val, sympy_val) in enumerate(zip(cuda_results, sympy_results)):
        diff = sp.simplify(sp.expand(cuda_val - sympy_val))
        is_correct = (diff == 0)
        
        if not is_correct:
            print(f"❌ drho_out_{i}: MISMATCH")
            print(f"   Difference: {diff}")
            all_correct = False
        else:
            print(f"✓ drho_out_{i}: OK")
    
    print("="*60)
    if all_correct:
        print("✅ ALL CHECKS PASSED!")
    else:
        print("❌ SOME CHECKS FAILED!")
    print("="*60)


# Run everything
print("Generated CUDA code:")
print("="*60)
print_commutator_cuda()
print()
print()
verify_cuda_code()



















