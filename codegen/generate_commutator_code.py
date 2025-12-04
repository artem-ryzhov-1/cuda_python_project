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

'''
rho = sp.Matrix([
    [r0,         r01+sp.I*i01, r02+sp.I*i02, r03+sp.I*i03],
    [r01-sp.I*i01, r1,         r12+sp.I*i12, r13+sp.I*i13],
    [r02-sp.I*i02, r12-sp.I*i12, r2,         r23+sp.I*i23],
    [r03-sp.I*i03, r13-sp.I*i13, r23-sp.I*i23, r3]
])
'''

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
delta_C, alpha, pi_alpha, B, m, one_div_m = sp.symbols('delta_C alpha pi_alpha B m one_div_m', real=True, positive=True)


# Diabatic Hamiltonian
H = - sp.Rational(1,2) * sp.Matrix([
    [ -B*eps_t_substep - one_div_m,  0,          0,              0                         ],
    [       0,                    eps_t_substep, delta_C,        0                         ],
    [       0,                    delta_C,    -eps_t_substep,    0                         ],
    [       0,                    0,          0,              B*eps_t_substep - one_div_m  ]
])


'''

H_d = - sp.Matrix([
    [ -B*varepsilon - one_div_m,  0,          0,              0                         ],
    [       0,                    varepsilon, Delta_C,        0                         ],
    [       0,                    Delta_C,    -varepsilon,    0                         ],
    [       0,                    0,          0,              B*varepsilon - one_div_m  ]
])

'''



# Commutator [H, rho] = -I*(H*rho - rho*H)
comm = -sp.I*(H*rho - rho*H)


print(is_hermitian_matrix_elementwise(comm))

# Display result
#sp.pprint(comm, use_unicode=True)



def print_terms2(name, mat):
    # diagonals
    for i in range(4):
        real = (sp.re((mat[i,i])))
        imag = (sp.im((mat[i,i])))
        print(f"{name}{i}{i} =", real)
        print()
        print(f"{name}{i}{i}_im =", imag)
        print()
    # off-diagonals (real + imag)
    for i in range(4):
        for j in range(i+1,4):
            real = (sp.re((mat[i,j])))
            imag = (sp.im((mat[i,j])))
            print(f"{name}{i}{j} =", real)
            print()
            print(f"{name}{i}{j}_im =", imag)
            print()


# ===== Print everything =====

print_terms2("drho_out_", comm)

# print("\n==== Mrho ====")






##########################################
# test






drho_out_0 = 0;


tmp = delta_C * rho_in_11;

drho_out_1 = tmp;


drho_out_2 = -tmp;


drho_out_3 = 0;

# (0,1)
tmp = 0;
tmp = B * eps_t_substep;
tmp += one_div_m;
tmp += eps_t_substep;
tmp *= rho_in_5;

tmp += delta_C * rho_in_7;
tmp *= 0.5;
drho_out_4 = tmp;

tmp = 0;
tmp = -B * eps_t_substep;
tmp += -one_div_m;
tmp += -eps_t_substep;
tmp *= rho_in_4;

tmp += -delta_C * rho_in_6;
tmp *= 0.5;
drho_out_5 = tmp;

# (0,2)
tmp = 0;
tmp = B * eps_t_substep;
tmp += one_div_m;
tmp += -eps_t_substep;
tmp *= rho_in_7;

tmp += delta_C * rho_in_5;
tmp *= 0.5;
drho_out_6 = tmp;

tmp = 0;
tmp = -B * eps_t_substep;
tmp += -one_div_m;
tmp += eps_t_substep;
tmp *= rho_in_6;

tmp += -delta_C * rho_in_4;
tmp *= 0.5;
drho_out_7 = tmp;

# (0,3)
drho_out_8 = B * eps_t_substep * rho_in_9;

drho_out_9 = -B * eps_t_substep * rho_in_8;

# (1,2)
drho_out_10 = -eps_t_substep * rho_in_11;

tmp = 0;
tmp = -delta_C * rho_in_1;
tmp += delta_C * rho_in_2;
tmp += 2 * eps_t_substep * rho_in_10;
tmp *= 0.5;
drho_out_11 = tmp;

# (1,3)
tmp = 0;
tmp = -eps_t_substep;
tmp += B * eps_t_substep;
tmp += -one_div_m;
tmp *= rho_in_13;

tmp += -delta_C * rho_in_15;
tmp *= 0.5;
drho_out_12 = tmp;

tmp = 0;
tmp = eps_t_substep;
tmp += -B * eps_t_substep;
tmp += one_div_m;
tmp *= rho_in_12;

tmp += delta_C * rho_in_14;
tmp *= 0.5;
drho_out_13 = tmp;

# (2,3)
tmp = 0;
tmp = eps_t_substep;
tmp += B * eps_t_substep;
tmp += -one_div_m;
tmp *= rho_in_15;

tmp += -delta_C * rho_in_13;
tmp *= 0.5;
drho_out_14 = tmp;


tmp = 0;
tmp = -B * eps_t_substep;
tmp += -eps_t_substep;
tmp += one_div_m;
tmp *= rho_in_14;

tmp += delta_C * rho_in_12;
tmp *= 0.5;
drho_out_15 = tmp;





# ================
# Compare with target
# ================
def check(expr, target):
    diff = sp.simplify(sp.expand(expr - target))
    return diff == 0, sp.simplify(diff)




print(im(comm[0,0]))
print(im(comm[1,1]))
print(im(comm[2,2]))
print(im(comm[3,3]))


print(check(drho_out_0,    re(comm[0,0])))
print(check(drho_out_1,    re(comm[1,1])))
print(check(drho_out_2,    re(comm[2,2])))
print(check(drho_out_3,    re(comm[3,3])))


print(check(drho_out_4,    re(comm[0,1])))
print(check(drho_out_5,    im(comm[0,1])))

print(check(drho_out_6,    re(comm[0,2])))
print(check(drho_out_7,    im(comm[0,2])))

print(check(drho_out_8,    re(comm[0,3])))
print(check(drho_out_9,    im(comm[0,3])))

print(check(drho_out_10,    re(comm[1,2])))
print(check(drho_out_11,    im(comm[1,2])))

print(check(drho_out_12,    re(comm[1,3])))
print(check(drho_out_13,    im(comm[1,3])))

print(check(drho_out_14,    re(comm[2,3])))
print(check(drho_out_15,    im(comm[2,3])))

























