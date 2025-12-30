import sympy as sp
from sympy import Integer, re, im, simplify, expand
import IPython
from IPython.display import display

# --- symbols ---
I = sp.I

# Gamma symbols
Gamma_10, Gamma_20, Gamma_30 = sp.symbols('Gamma_10 Gamma_20 Gamma_30 ', positive=True, real=True)
Gamma_21, Gamma_31, Gamma_32 = sp.symbols('Gamma_21 Gamma_31 Gamma_32',  positive=True, real=True)


# rho entries
r00, r11, r22, r33 = sp.symbols('r00 r11 r22 r33', real=True)
r01, i01 = sp.symbols('r01 i01', real=True)
r02, i02 = sp.symbols('r02 i02', real=True)
r03, i03 = sp.symbols('r03 i03', real=True)
r12, i12 = sp.symbols('r12 i12', real=True)
r13, i13 = sp.symbols('r13 i13', real=True)
r23, i23 = sp.symbols('r23 i23', real=True)

# U matrix symbols (4x4)
U00, U01, U02, U03 = sp.symbols('U00 U01 U02 U03', real=True)
U10, U11, U12, U13 = sp.symbols('U10 U11 U12 U13', real=True)
U20, U21, U22, U23 = sp.symbols('U20 U21 U22 U23', real=True)
U30, U31, U32, U33 = sp.symbols('U30 U31 U32 U33', real=True)

U = sp.Matrix([
    [U00, U01, U02, U03],
    [U10, U11, U12, U13],
    [U20, U21, U22, U23],
    [U30, U31, U32, U33]
])

# rho density matrix (Hermitian) in the diabatic basis
rho = sp.Matrix([
    [r00,           r01 + I*i01, r02 + I*i02, r03 + I*i03],
    [r01 - I*i01, r11,           r12 + I*i12, r13 + I*i13],
    [r02 - I*i02, r12 - I*i12, r22,           r23 + I*i23],
    [r03 - I*i03, r13 - I*i13, r23 - I*i23, r33]
])

# --- helper constructors ---
def L_adb_en(i, j, Gamma):
    M = sp.zeros(4)
    M[i, j] = sp.sqrt(Gamma)
    return M



# --- Lindblad dissipator ---
# def dissipator(L, rho):
#     return L * rho * L.H - sp.Rational(1,2) * (L.H * L * rho + rho * L.H * L)



def is_simmetrical_matrix_elementwise(M):
    """
    Checks if a SymPy matrix is Hermitian by comparing each element with its conjugate transpose counterpart.
    
    A matrix M is Hermitian if M[i, j] == conjugate(M[j, i]) for all i, j.
    """
    rows, cols = M.shape
    if rows != cols:
        return False  # Must be square

    for i in range(rows):
        for j in range(cols):
            if not sp.simplify(M[i, j] - M[j, i]) == 0:
                return False
    return True







L_adb_01 = L_adb_en(0,1,Gamma_10)
L_adb_02 = L_adb_en(0,2,Gamma_20)
L_adb_03 = L_adb_en(0,3,Gamma_30)
L_adb_12 = L_adb_en(1,2,Gamma_21)
L_adb_13 = L_adb_en(1,3,Gamma_31)
L_adb_23 = L_adb_en(2,3,Gamma_32)





def calulate_M_db(L_adb):
    
    M_adb = L_adb.T*L_adb
    M_db = U*M_adb*U.T
    return M_db



def dissipator_db(L_adb):
    
    L_db = U*L_adb*U.T
    
    # option 1 for calculation M_db (simpler)
    M_db = calulate_M_db(L_adb)
    
    # option 2 for calculation M_db (more complicated after unrolling matrix multiplications)
    #M_db = L_db.T*L_db
    
    
    
    
    
    dissipator_db = L_db * rho * L_db.T - sp.Rational(1,2) * (M_db * rho + rho * M_db)
    
    dissipator_db = expand(dissipator_db)
    
    return dissipator_db

    
    #L_db = U*L_adb*U.T
    #D_db = dissipator(L_db, rho)
    #D_db = expand(D_db)
    #return D_db


D_sum_db = sp.zeros(4)

D_sum_db += dissipator_db(L_adb_01)
D_sum_db += dissipator_db(L_adb_02)
D_sum_db += dissipator_db(L_adb_03)
D_sum_db += dissipator_db(L_adb_12)
D_sum_db += dissipator_db(L_adb_13)
D_sum_db += dissipator_db(L_adb_23)




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



check_is_hermitian_D_sum_db = is_hermitian_matrix_elementwise(D_sum_db)
print(check_is_hermitian_D_sum_db)


D_sum_db_re = re(D_sum_db)
D_sum_db_im = im(D_sum_db)


D_sum_db_re = expand(simplify(D_sum_db_re))
D_sum_db_im = expand(simplify(D_sum_db_im))


#D_sum_db_re_x2 = re(D_sum_db_x2)
#D_sum_db_im_x2 = im(D_sum_db_x2)

#D_sum_db_re_x2 = expand(simplify(D_sum_db_re_x2))
#D_sum_db_im_x2 = expand(simplify(D_sum_db_im_x2))











print(D_sum_db_im[0,0])
print(D_sum_db_im[1,1])
print(D_sum_db_im[2,2])
print(D_sum_db_im[3,3])


check_D_sum_db_im_00_is_zero = (D_sum_db_im[0,0] == 0)
check_D_sum_db_im_11_is_zero = (D_sum_db_im[1,1] == 0)
check_D_sum_db_im_22_is_zero = (D_sum_db_im[2,2] == 0)
check_D_sum_db_im_33_is_zero = (D_sum_db_im[3,3] == 0)





D_r00 = D_sum_db_re[0,0]
D_r11 = D_sum_db_re[1,1]
D_r22 = D_sum_db_re[2,2]
D_r33 = D_sum_db_re[3,3]

D_r01 = D_sum_db_re[0,1]
D_i01 = D_sum_db_im[0,1]

D_r02 = D_sum_db_re[0,2]
D_i02 = D_sum_db_im[0,2]

D_r03 = D_sum_db_re[0,3]
D_i03 = D_sum_db_im[0,3]

D_r12 = D_sum_db_re[1,2]
D_i12 = D_sum_db_im[1,2]

D_r13 = D_sum_db_re[1,3]
D_i13 = D_sum_db_im[1,3]

D_r23 = D_sum_db_re[2,3]
D_i23 = D_sum_db_im[2,3]







#################################################################
#################################################################
#################################################################





def substitute_expressions(expr_dict, subs_dict1, subs_dict2, suffix):
    """
    Applies substitutions to each expression in the dictionary and appends a suffix to the new keys.
    
    Parameters:
        expr_dict (dict): Dictionary with variable names (str) as keys and SymPy expressions as values.
        subs_dict (dict): Dictionary of substitutions to apply.
        suffix (str): Suffix to append to keys of substituted expressions. Default is "_1".

    Returns:
        dict: New dictionary with keys suffixed by `suffix`, containing substituted expressions.
    """
    return {f"{name}{suffix}": expr.subs(subs_dict1).subs(subs_dict2) for name, expr in expr_dict.items()}






expr_dict = {
    "D_r00": D_r00,
    "D_r11": D_r11,
    "D_r22": D_r22,
    "D_r33": D_r33,
    "D_r01": D_r01,
    "D_i01": D_i01,
    "D_r02": D_r02,
    "D_i02": D_i02,
    "D_r03": D_r03,
    "D_i03": D_i03,
    "D_r12": D_r12,
    "D_i12": D_i12,
    "D_r13": D_r13,
    "D_i13": D_i13,
    "D_r23": D_r23,
    "D_i23": D_i23,
}








U00_test, U01_test, U02_test, U03_test = sp.symbols('U00_test U01_test U02_test U03_test', real=True)
U10_test, U11_test, U12_test, U13_test = sp.symbols('U10_test U11_test U12_test U13_test', real=True)
U20_test, U21_test, U22_test, U23_test = sp.symbols('U20_test U21_test U22_test U23_test', real=True)
U30_test, U31_test, U32_test, U33_test = sp.symbols('U30_test U31_test U32_test U33_test', real=True)



subs_dict_test_col0_p = {
    U00_test: U00,
    U10_test: U10,
    U20_test: U20,
    U30_test: U30
}

subs_dict_test_col0_m = {
    U00_test: -U00,
    U10_test: -U10,
    U20_test: -U20,
    U30_test: -U30
}

subs_dict_test_col1_p = {
    U01_test: U01,
    U11_test: U11,
    U21_test: U21,
    U31_test: U31
}

subs_dict_test_col1_m = {
    U01_test: -U01,
    U11_test: -U11,
    U21_test: -U21,
    U31_test: -U31
}


subs_dict_test_col2_p = {
    U02_test: U02,
    U12_test: U12,
    U22_test: U22,
    U32_test: U32
}

subs_dict_test_col2_m = {
    U02_test: -U02,
    U12_test: -U12,
    U22_test: -U22,
    U32_test: -U32
}


subs_dict_test_col3_p = {
    U03_test: U03,
    U13_test: U13,
    U23_test: U23,
    U33_test: U33
}

subs_dict_test_col3_m = {
    U03_test: -U03,
    U13_test: -U13,
    U23_test: -U23,
    U33_test: -U33
}



###

D_db_test_col0_p = substitute_expressions(
    expr_dict,
    subs_dict_test_col0_p,
    {},
    suffix=""
    )

D_db_test_col0_m = substitute_expressions(
    expr_dict,
    subs_dict_test_col0_m,
    {},
    suffix=""
    )


test_col0 = (D_db_test_col0_p == D_db_test_col0_m)






D_db_test_col1_p = substitute_expressions(
    expr_dict,
    subs_dict_test_col1_p,
    {},
    suffix=""
    )

D_db_test_col1_m = substitute_expressions(
    expr_dict,
    subs_dict_test_col1_m,
    {},
    suffix=""
    )


test_col1 = (D_db_test_col1_p == D_db_test_col1_m)




D_db_test_col2_p = substitute_expressions(
    expr_dict,
    subs_dict_test_col2_p,
    {},
    suffix=""
    )

D_db_test_col2_m = substitute_expressions(
    expr_dict,
    subs_dict_test_col2_m,
    {},
    suffix=""
    )


test_col2 = (D_db_test_col2_p == D_db_test_col2_m)








D_db_test_col3_p = substitute_expressions(
    expr_dict,
    subs_dict_test_col3_p,
    {},
    suffix=""
    )

D_db_test_col3_m = substitute_expressions(
    expr_dict,
    subs_dict_test_col3_m,
    {},
    suffix=""
    )


test_col3 = (D_db_test_col3_p == D_db_test_col3_m)



check_column_invariance = test_col0 and test_col1 and test_col2 and test_col3

print(check_column_invariance)



###########################







g_p = sp.symbols('g_p', positive=True, real=True)
g_m = sp.symbols('g_m', positive=True, real=True)









###
# int 1

subs_dict_U_int1 = {
    U00: 1,     U01: 0,     U02: 0,     U03: 0,
    U10: 0,     U11: g_p,   U12: -g_m,  U13: 0,
    U20: 0,     U21: g_m,   U22: g_p,   U23: 0,
    U30: 0,     U31: 0,     U32: 0,     U33: 1
}


###
# int 2


subs_dict_U_int2 = {
    U00: 0,     U01: 1,     U02: 0,     U03: 0,
    U10: g_p,   U11: 0,     U12: -g_m,  U13: 0,
    U20: g_m,   U21: 0,     U22: g_p,   U23: 0,
    U30: 0,     U31: 0,     U32: 0,     U33: 1
}



###
# int 3

subs_dict_U_int3 = {
    U00: 0,     U01: 0,     U02: 1,     U03: 0,
    U10: g_p,   U11: -g_m,  U12: 0,     U13: 0,
    U20: g_m,   U21: g_p,   U22: 0,     U23: 0,
    U30: 0,     U31: 0,     U32: 0,     U33: 1
}


###
# int 4

subs_dict_U_int4 = {
    U00: 0,     U01: 0,     U02: 0,     U03: 1,
    U10: g_p,   U11: -g_m,  U12: 0,     U13: 0,
    U20: g_m,   U21: g_p,   U22: 0,     U23: 0,
    U30: 0,     U31: 0,     U32: 1,     U33: 0
}


###
# int 5

subs_dict_U_int5 = {
    U00: 0,     U01: 0,     U02: 0,     U03: 1,
    U10: g_p,   U11: 0,     U12: -g_m,  U13: 0,
    U20: g_m,   U21: 0,     U22: g_p,   U23: 0,
    U30: 0,     U31: 1,     U32: 0,     U33: 0
}

####
# int 6

subs_dict_U_int6 = {
    U00: 0,     U01: 0,     U02: 0,     U03: 1,
    U10: 0,     U11: g_p,   U12: -g_m,  U13: 0,
    U20: 0,     U21: g_m,   U22: g_p,   U23: 0,
    U30: 1,     U31: 0,     U32: 0,     U33: 0
}


################################################################
################################################################
################################################################
################################################################
################################################################
# simplifying Gammas




# helper functions for getting elements in compute_W



def get_U_element(m, n, subs_dict):
    """
    Retrieve the value of U[i][f] from a dictionary where keys are sympy symbols like U00, U01, ..., U33.

    Parameters:
    - i (int): row index (0–3)
    - f (int): column index (0–3)
    - subs_dict (dict): dictionary with keys like U00, U01, ..., U33 as sympy symbols

    Returns:
    - The value corresponding to U_if if it exists, else raises KeyError
    """
    
    
    #key = sp.Symbol(f"U{m}{n}")
    key = U[m,n]
    
    # Return the value from the dictionary
    return subs_dict[key]




#GammaL0 = sp.symbols('Gamma_L0', real=True, positive=True)
#GammaR0 = sp.symbols('Gamma_R0', real=True, positive=True)

GammaL0 = sp.symbols('GammaL0', real=True, positive=True)
GammaR0 = sp.symbols('GammaR0', real=True, positive=True)


def get_M_LR(i, f, subs_dict):
    """
    Compute M_L and M_R from the substitution dictionary and given indices i and f.

    Parameters:
    - i (int): row index (0–3)
    - f (int): column index (0–3)
    - subs_dict (dict): dictionary with keys like U00, U01, ..., U33 as sympy symbols

    Returns:
    - (M_L, M_R): tuple of expressions
    """
    U0f = get_U_element(0, f, subs_dict)
    U1f = get_U_element(1, f, subs_dict)
    U2f = get_U_element(2, f, subs_dict)
    #U3f = get_U_element(3, f, interval)
    
    #U0i = get_U_element(0, i, interval)
    U1i = get_U_element(1, i, subs_dict)
    U2i = get_U_element(2, i, subs_dict)
    U3i = get_U_element(3, i, subs_dict)
    
    M_L = U0f * U2i + U1f * U3i
    M_R = U0f * U1i + U2f * U3i
    
    
    
    return M_L, M_R


def get_Gamma_compute_W(i, f, subs_dict):
    
    
    M_L, M_R = get_M_LR(i, f, subs_dict)
    
    return GammaL0*M_L*M_L + GammaR0*M_R*M_R;





Gamma_10_int1 = get_Gamma_compute_W(1, 0, subs_dict_U_int1)
Gamma_20_int1 = get_Gamma_compute_W(2, 0, subs_dict_U_int1)
Gamma_30_int1 = sp.Integer(0) #Gamma_30_int1 = get_Gamma_compute_W(3, 0, subs_dict_U_int1)
Gamma_21_int1 = sp.Integer(0) #Gamma_21_int1 = get_Gamma_compute_W(2, 1, subs_dict_U_int1)
Gamma_31_int1 = get_Gamma_compute_W(3, 1, subs_dict_U_int1)
Gamma_32_int1 = get_Gamma_compute_W(3, 2, subs_dict_U_int1)


Gamma_10_int2 = get_Gamma_compute_W(0, 1, subs_dict_U_int2)   # reverse
Gamma_20_int2 = sp.Integer(0) #Gamma_20_int2 = get_Gamma_compute_W(2, 0, subs_dict_U_int2)
Gamma_30_int2 = get_Gamma_compute_W(3, 0, subs_dict_U_int2)
Gamma_21_int2 = get_Gamma_compute_W(2, 1, subs_dict_U_int2)
Gamma_31_int2 = sp.Integer(0) #Gamma_31_int2 = get_Gamma_compute_W(3, 1, subs_dict_U_int2)
Gamma_32_int2 = get_Gamma_compute_W(3, 2, subs_dict_U_int2)


Gamma_10_int3 = sp.Integer(0) #Gamma_10_int3 = get_Gamma_compute_W(1, 0, subs_dict_U_int3)
Gamma_20_int3 = get_Gamma_compute_W(0, 2, subs_dict_U_int3)   # reverse
Gamma_30_int3 = get_Gamma_compute_W(3, 0, subs_dict_U_int3)
Gamma_21_int3 = get_Gamma_compute_W(1, 2, subs_dict_U_int3)   # reverse
Gamma_31_int3 = get_Gamma_compute_W(3, 1, subs_dict_U_int3)
Gamma_32_int3 = sp.Integer(0) #Gamma_32_int3 = get_Gamma_compute_W(3, 2, subs_dict_U_int3)


Gamma_10_int4 = sp.Integer(0) #Gamma_10_int4 = get_Gamma_compute_W(1, 0, subs_dict_U_int4)
Gamma_20_int4 = get_Gamma_compute_W(2, 0, subs_dict_U_int4)
Gamma_30_int4 = get_Gamma_compute_W(0, 3, subs_dict_U_int4)   # reverse
Gamma_21_int4 = get_Gamma_compute_W(2, 1, subs_dict_U_int4)
Gamma_31_int4 = get_Gamma_compute_W(1, 3, subs_dict_U_int4)   # reverse
Gamma_32_int4 = sp.Integer(0) #Gamma_32_int4 = get_Gamma_compute_W(3, 2, subs_dict_U_int4)


Gamma_10_int5 = get_Gamma_compute_W(1, 0, subs_dict_U_int5)
Gamma_20_int5 = sp.Integer(0) #Gamma_20_int5 = get_Gamma_compute_W(2, 0, subs_dict_U_int5)
Gamma_30_int5 = get_Gamma_compute_W(0, 3, subs_dict_U_int5)   # reverse
Gamma_21_int5 = get_Gamma_compute_W(1, 2, subs_dict_U_int5)   # reverse
Gamma_31_int5 = sp.Integer(0) #Gamma_31_int5 = get_Gamma_compute_W(3, 1, subs_dict_U_int5)
Gamma_32_int5 = get_Gamma_compute_W(2, 3, subs_dict_U_int5)   # reverse


Gamma_10_int6 = get_Gamma_compute_W(0, 1, subs_dict_U_int6)   # reverse
Gamma_20_int6 = get_Gamma_compute_W(0, 2, subs_dict_U_int6)   # reverse
Gamma_30_int6 = sp.Integer(0) #Gamma_30_int6 = get_Gamma_compute_W(3, 0, subs_dict_U_int6)
Gamma_21_int6 = sp.Integer(0) #Gamma_21_int6 = get_Gamma_compute_W(2, 1, subs_dict_U_int6)
Gamma_31_int6 = get_Gamma_compute_W(1, 3, subs_dict_U_int6)   # reverse
Gamma_32_int6 = get_Gamma_compute_W(2, 3, subs_dict_U_int6)   # reverse




    
def print_all_Gammas_for_interval(interval):
    """
    Prints the already computed symbolic Gamma_ij_int{interval} values
    using the fixed list of ij pairs.

    Parameters:
    - interval (int): the interval number (e.g. 1, 2, 3, ...)
    """
    ij_pairs = [
        (1, 0), (2, 0), (3, 0),
        (2, 1), (3, 1),
        (3, 2)
    ]
    
    for i, j in ij_pairs:
        var_name = f"Gamma_{i}{j}_int{interval}"
        try:
            value = eval(var_name)
            print(f"{var_name} = {value};")
        except NameError:
            pass  # Variable not defined — skip


print()
print("Interval 1")
print_all_Gammas_for_interval(1)

print()
print("Interval 2")
print_all_Gammas_for_interval(2)

print()
print("Interval 3")
print_all_Gammas_for_interval(3)

print()
print("Interval 4")
print_all_Gammas_for_interval(4)

print()
print("Interval 5")
print_all_Gammas_for_interval(5)

print()
print("Interval 6")
print_all_Gammas_for_interval(6)




Gamma_lprm = GammaL0*g_p**2 + GammaR0*g_m**2
Gamma_lmrp = GammaL0*g_m**2 + GammaR0*g_p**2

GammaLR0 = GammaL0 + GammaR0

###############
# proof that
# Gamma_lprm + Gamma_lmrp = GammaL0 + GammaR0 = GammaLR0 = const

gp_sqr = sp.symbols('gp_sqr', real=True, positive=True)
gm_sqr = sp.symbols('gm_sqr', real=True, positive=True)


expr1 = Gamma_lprm + Gamma_lmrp

expr1_simplified1 = expr1.subs({g_p**2: gp_sqr, g_m**2: gm_sqr}).subs({gp_sqr: sp.Integer(1)-gm_sqr})

expr1_simplified = simplify(expr1_simplified1)

check_Gamma_equivalence1 = expr1_simplified.equals(GammaLR0)

print(check_Gamma_equivalence1)


# end of proof
###############

###############
# proof that
# Gamma_lmrp*gm_sqr + Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr + Gamma_lprm*gp_sqr =
# = GammaL0 + GammaR0 = GammaLR0 = const

expr2 = Gamma_lmrp*gm_sqr + Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr + Gamma_lprm*gp_sqr

expr2_simplified1 = expand(expr2)

# Group expression by GammaL0 and GammaR0
expr2_simplified2 = sp.collect(expr2_simplified1, [GammaL0*gp_sqr, GammaL0*gm_sqr, GammaR0*gp_sqr, GammaR0*gm_sqr])

expr2_simplified3 = expr2_simplified2.subs({g_p**2 + g_m**2: 1}) 

expr2_simplified = sp.expand(expr2_simplified3.subs({gp_sqr: sp.Integer(1)-gm_sqr}))

check_Gamma_equivalence2 = expr2_simplified.equals(GammaLR0)

print(check_Gamma_equivalence2)

# end of proof
###############

###############
# proof that
# Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr = ?


Gamma_lprm = GammaL0*g_p**2 + GammaR0*g_m**2
Gamma_lmrp = GammaL0*g_m**2 + GammaR0*g_p**2

expr3 = Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr

expr4 = expand(expr3.subs({gp_sqr: g_p**2, gm_sqr: g_m**2}))









# end of proof
###############

GammaLR0 = sp.symbols('GammaLR0', real=True, positive=True)


test_equivalence = (
    Gamma_10_int1.equals(Gamma_lmrp) and
    Gamma_20_int1.equals(Gamma_lprm) and
    Gamma_30_int1.equals(0) and
    Gamma_21_int1.equals(0) and
    Gamma_31_int1.equals(Gamma_lprm) and
    Gamma_32_int1.equals(Gamma_lmrp) and
    
    Gamma_10_int2.equals(Gamma_lmrp) and
    Gamma_20_int2.equals(0) and
    Gamma_30_int2.equals(Gamma_lprm) and
    Gamma_21_int2.equals(Gamma_lprm) and
    Gamma_31_int2.equals(0) and
    Gamma_32_int2.equals(Gamma_lmrp) and
    
    Gamma_10_int3.equals(0) and
    Gamma_20_int3.equals(Gamma_lmrp) and
    Gamma_30_int3.equals(Gamma_lprm) and
    Gamma_21_int3.equals(Gamma_lprm) and
    Gamma_31_int3.equals(Gamma_lmrp) and
    Gamma_32_int3.equals(0) and
    
    Gamma_10_int4.equals(0) and
    Gamma_20_int4.equals(Gamma_lprm) and
    Gamma_30_int4.equals(Gamma_lmrp) and
    Gamma_21_int4.equals(Gamma_lmrp) and
    Gamma_31_int4.equals(Gamma_lprm) and
    Gamma_32_int4.equals(0) and
    
    Gamma_10_int5.equals(Gamma_lprm) and
    Gamma_20_int5.equals(0) and
    Gamma_30_int5.equals(Gamma_lmrp) and
    Gamma_21_int5.equals(Gamma_lmrp) and
    Gamma_31_int5.equals(0) and
    Gamma_32_int5.equals(Gamma_lprm) and
    
    Gamma_10_int6.equals(Gamma_lprm) and
    Gamma_20_int6.equals(Gamma_lmrp) and
    Gamma_30_int6.equals(0) and
    Gamma_21_int6.equals(0) and
    Gamma_31_int6.equals(Gamma_lmrp) and
    Gamma_32_int6.equals(Gamma_lprm)
)

print(test_equivalence)



Gamma_lprm = sp.symbols('Gamma_lprm', real=True, positive=True)
Gamma_lmrp = sp.symbols('Gamma_lmrp', real=True, positive=True)


subs_dict_Gamma_int1 = {
    Gamma_10: Gamma_lmrp,
    Gamma_20: Gamma_lprm,
    Gamma_30: 0,
    Gamma_21: 0,
    Gamma_31: Gamma_lprm,
    Gamma_32: Gamma_lmrp
}

subs_dict_Gamma_int2 = {
    Gamma_10: Gamma_lmrp,
    Gamma_20: 0,
    Gamma_30: Gamma_lprm,
    Gamma_21: Gamma_lprm,
    Gamma_31: 0,
    Gamma_32: Gamma_lmrp
}

subs_dict_Gamma_int3 = {
    Gamma_10: 0,
    Gamma_20: Gamma_lmrp,
    Gamma_30: Gamma_lprm,
    Gamma_21: Gamma_lprm,
    Gamma_31: Gamma_lmrp,
    Gamma_32: 0
}

subs_dict_Gamma_int4 = {
    Gamma_10: 0,
    Gamma_20: Gamma_lprm,
    Gamma_30: Gamma_lmrp,
    Gamma_21: Gamma_lmrp,
    Gamma_31: Gamma_lprm,
    Gamma_32: 0
}

subs_dict_Gamma_int5 = {
    Gamma_10: Gamma_lprm,
    Gamma_20: 0,
    Gamma_30: Gamma_lmrp,
    Gamma_21: Gamma_lmrp,
    Gamma_31: 0,
    Gamma_32: Gamma_lprm
}

subs_dict_Gamma_int6 = {
    Gamma_10: Gamma_lprm,
    Gamma_20: Gamma_lmrp,
    Gamma_30: 0,
    Gamma_21: 0,
    Gamma_31: Gamma_lmrp,
    Gamma_32: Gamma_lprm
}







################################################



###
# int 1



D_db_int1 = substitute_expressions(
    expr_dict,
    subs_dict_U_int1,
    subs_dict_Gamma_int1,
    suffix=""
    )



###
# int 2




D_db_int2 = substitute_expressions(
    expr_dict,
    subs_dict_U_int2,
    subs_dict_Gamma_int2,
    suffix=""
    )




###
# int 3




D_db_int3 = substitute_expressions(
    expr_dict,
    subs_dict_U_int3,
    subs_dict_Gamma_int3,
    suffix=""
    )



###
# int 4


D_db_int4 = substitute_expressions(
    expr_dict,
    subs_dict_U_int4,
    subs_dict_Gamma_int4,
    suffix=""
    )



###
# int 5



D_db_int5 = substitute_expressions(
    expr_dict,
    subs_dict_U_int5,
    subs_dict_Gamma_int5,
    suffix=""
    )





####
# int 6


D_db_int6 = substitute_expressions(
    expr_dict,
    subs_dict_U_int6,
    subs_dict_Gamma_int6,
    suffix=""
    )


#############################














# Access example:
#D_db_cuda_int1_U00_p1["D_i23_db_cuda"]

############################



def display_dict(expr_dict, clear_opt=False):
    """
    Prints and displays all key-value pairs in a dictionary of SymPy expressions.
    
    Parameters:
        expr_dict (dict): Dictionary with keys (usually strings) and SymPy expressions as values.
    """
    
    g_p_sym = sp.symbols('\\gamma_{+}', real=True, positive=True)
    g_m_sym = sp.symbols('\\gamma_{-}', real=True, positive=True)
    
    Gamma_lprm_sym = sp.symbols('\\Gamma_{lp}', real=True, positive=True)
    Gamma_lmrp_sym = sp.symbols('\\Gamma_{lm}', real=True, positive=True)

    GammaLR0_sym = sp.symbols('\\Gamma_{LR0}', real=True, positive=True)
    
    subs_dict_latex = {
        g_p: g_p_sym,
        g_m: g_m_sym,
        Gamma_lprm: Gamma_lprm_sym,
        Gamma_lmrp: Gamma_lmrp_sym,
        GammaLR0: GammaLR0_sym
    }
    
    

    if clear_opt == True:
        # Clear IPython console (works in Spyder's IPython Console)
        ipython = IPython.get_ipython()
        if ipython is not None:
            ipython.run_line_magic('clear', '')
        
        
    sp.init_printing(use_latex='mathjax')  # Ensure LaTeX rendering in IPython environments
    
    for key, expr in expr_dict.items():
        #print(f"{key} = {expr}")
        display(sp.Eq(sp.Symbol(key), expr.subs(subs_dict_latex)))
        print()
        print("--------------------")
        print()




def print_dict(expr_dict):
     """
     Prints and displays all key-value pairs in a dictionary of SymPy expressions.
     
     Parameters:
         expr_dict (dict): Dictionary with keys (usually strings) and SymPy expressions as values.
     """
     
     # Clear IPython console (works in Spyder's IPython Console)
     ipython = IPython.get_ipython()
     if ipython is not None:
         ipython.run_line_magic('clear', '')
         
         
     sp.init_printing(use_latex='mathjax')  # Ensure LaTeX rendering in IPython environments
     
     for key, expr in expr_dict.items():
         print(f"drho_out_{key} = {expr};")
         #display(sp.Eq(sp.Symbol(key), expr))
         print()
         print()
       


def print_dict_cuda_style(expr_dict, for_python_check_opt = False):
    """
    Prints the expressions in CUDA-style C code with individual terms on separate lines.

    Parameters:
        expr_dict (dict): Dictionary with keys (e.g., 'D_r00') and SymPy expressions as values.
    """

    # Clear IPython console (works in Spyder's IPython Console)
    ipython = IPython.get_ipython()
    if ipython is not None:
        ipython.run_line_magic('clear', '')

    for key, expr in expr_dict.items():
        
        if for_python_check_opt == False:
            print("tmp = 0.0f;")
        elif for_python_check_opt == True:
            print("tmp = 0")

        # Convert to sympy expression if it's not already
        expr = sp.sympify(expr)

        # If it's an addition of terms, split them
        if isinstance(expr, sp.Add):
            terms = expr.args
        else:
            terms = [expr]

        first_term = True
        for term in terms:
            # Convert ** to * * for C-style syntax
            #term_str = str(term).replace("**", "* *")
            term_str = str(term).replace("*", " * ")

            if first_term:
                print(f"tmp = {term_str};")
                first_term = False
            else:
                print(f"tmp += {term_str};")
        
        
        
        if for_python_check_opt == False:
            print("tmp *= 0.5f;")
            print(f"drho_out_{key} += tmp;")
            print(f"d_log_buffer[t_idx_substep].drho_out_{key} = tmp;\n\n")
        elif for_python_check_opt == True:
            print("tmp *= half")
            print(f"drho_out_{key} = tmp\n\n")






checks = (check_is_hermitian_D_sum_db
         and check_D_sum_db_im_00_is_zero
         and check_D_sum_db_im_11_is_zero
         and check_D_sum_db_im_22_is_zero
         and check_D_sum_db_im_33_is_zero
         and check_column_invariance
         and check_Gamma_equivalence1
         and check_Gamma_equivalence2
         and test_equivalence)

if not checks:
    raise ValueError("Checks not passed")
else:
    print("Checks passed successfully.")


'''
print_dict(D_db_int1)
print_dict(D_db_int2)
print_dict(D_db_int3)
print_dict(D_db_int4)
print_dict(D_db_int5)
print_dict(D_db_int6)
'''





























D_db_int1_x2 = {key: Integer(2) * expr for key, expr in D_db_int1.items()}
D_db_int2_x2 = {key: Integer(2) * expr for key, expr in D_db_int2.items()}
D_db_int3_x2 = {key: Integer(2) * expr for key, expr in D_db_int3.items()}
D_db_int4_x2 = {key: Integer(2) * expr for key, expr in D_db_int4.items()}
D_db_int5_x2 = {key: Integer(2) * expr for key, expr in D_db_int5.items()}
D_db_int6_x2 = {key: Integer(2) * expr for key, expr in D_db_int6.items()}



gp_gm = sp.symbols('gp_gm', real=True, positive=True)



'''


def simplify_dict_opt1(expr_dict):
    """
    Groups each expression in the dictionary by products of Gamma_ij * r_mn or Gamma_ij * i_mn,
    and returns a new dictionary with grouped expressions.

    Parameters:
    - expr_dict (dict): Original dictionary of symbolic expressions.

    Returns:
    - grouped_dict (dict): New dictionary with grouped expressions.
    """
    
    
    ## Step 1: List of predefined Gamma symbols (assumes they exist in the current scope)
    #gamma_symbols = [
    #    #Gamma_lprm, Gamma_lmrp
    #    sp.Integer(1), sp.Integer(2), g_p*g_m, sp.Integer(2)*g_p*g_m, sp.Integer(4)*g_p*g_m
    #]

    ## Step 2: List of all defined r_ij and i_ij symbols
    #rho_symbols = [
    #    r00, r11, r22, r33,
    #    r01, i01, r02, i02,
    #    r03, i03, r12, i12,
    #    r13, i13, r23, i23
    #]
    #
    ## Step 3: Generate all Gamma*r and Gamma*i products
    #collect_terms = []
    #for G in gamma_symbols:
    #    collect_terms.extend([G * rho for rho in rho_symbols])
    #
    # Step 4: Group each expression and build a new dictionary
    #grouped_dict = {}
    #
    #for key, expr in expr_dict.items():
    #    expr = sp.expand(expr)
    #    grouped_expr = sp.collect(expr, collect_terms, evaluate=False)
    #    regrouped_expr = sum(term * grouped_expr[term] for term in collect_terms if term in grouped_expr)
    #    grouped_dict[key] = regrouped_expr
    
    
    # Step 1: List of predefined Gamma symbols (assumes they exist in the current scope)
    coeffs = [
        #Gamma_lprm, Gamma_lmrp
        sp.Integer(1), sp.Integer(2), sp.Integer(4)
    ]

    # Step 2: List of all defined r_ij and i_ij symbols
    rho_symbols = [
        r00, r11, r22, r33,
        r01, i01, r02, i02,
        r03, i03, r12, i12,
        r13, i13, r23, i23
    ]

    collect_terms_c_rho = []
    for rho in rho_symbols:
        for c in coeffs:
            collect_terms_c_rho.append(c * rho)
    
    
    subs_dict_equality1 = {
        Gamma_lprm + Gamma_lmrp: GammaLR0,
        g_p*g_m: gp_gm
    }

    subs_dict_equality2 = {
        g_p**2: gp_sqr,
        g_m**2: gm_sqr
    }
    
    result_dict = {}
    
    for key, expr in expr_dict.items():
        simplified = sp.expand(sp.sympify(expr))
        simplified = simplified.subs(subs_dict_equality1)
        simplified = simplified.subs(subs_dict_equality2)
        
        simplified = sp.collect(simplified, collect_terms_c_rho, evaluate=True)
        
        result_dict[key] = simplified
    
    return result_dict
    





D_db_int1_x2_simpl_opt1 = simplify_dict_opt1(D_db_int1_x2)
D_db_int2_x2_simpl_opt1 = simplify_dict_opt1(D_db_int2_x2)
D_db_int3_x2_simpl_opt1 = simplify_dict_opt1(D_db_int3_x2)
D_db_int4_x2_simpl_opt1 = simplify_dict_opt1(D_db_int4_x2)
D_db_int5_x2_simpl_opt1 = simplify_dict_opt1(D_db_int5_x2)
D_db_int6_x2_simpl_opt1 = simplify_dict_opt1(D_db_int6_x2)



'''







'''
subs_dict_equality1 = {
    #g_p**2 + g_m**2: sp.Integer(1),
    
    
    #Gamma_lprm + Gamma_lmrp: GammaLR0,
        
    g_p*g_m: gp_gm

}

subs_dict_equality2 = {
    #g_p**2 + g_m**2: sp.Integer(1),
    g_p**2: gp_sqr,
    g_m**2: gm_sqr
}

subs_dict_equality3 = {
      Gamma_lmrp*gm_sqr: Gamma_lmrp_gm_sqr
}

subs_dict_equality2 = {
      Gamma_lmrp*gm_sqr + Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr + Gamma_lprm*gp_sqr: GammaLR0,
    - Gamma_lmrp*gm_sqr - Gamma_lmrp*gp_sqr - Gamma_lprm*gm_sqr - Gamma_lprm*gp_sqr: -GammaLR0
}
'''





Gamma_lprm = sp.symbols('Gamma_lprm', real=True, positive=True)
Gamma_lmrp = sp.symbols('Gamma_lmrp', real=True, positive=True)
GammaLR0   = sp.symbols('GammaLR0', real=True, positive=True)




def simplify_dict_opt2(expr_dict):

    new_dict = {}

    for key, expr in expr_dict.items():
        expr1 = sp.sympify(expr)


        # Intermediate simplification (optional)
        #expr = sp.expand(expr)

        expr2 = expr1.subs({g_p*g_m: gp_gm})
        expr3 = expr2.subs({g_p**2: gp_sqr, g_m**2: gm_sqr})
        expr4 = expand(expr3)
        

        
        
        # Step 1: List of predefined Gamma symbols (assumes they exist in the current scope)
        gamma_symbols = [
            Gamma_lprm, Gamma_lmrp
            #sp.Integer(1), sp.Integer(2), g_p*g_m, sp.Integer(2)*g_p*g_m, sp.Integer(4)*g_p*g_m
        ]
        
        
        # Step 2: List of all defined r_ij and i_ij symbols
        rho_symbols = [
            r00, r11, r22, r33,
            r01, i01, r02, i02,
            r03, i03, r12, i12,
            r13, i13, r23, i23
        ]
        
        # Step 3: Scalar multipliers
        coeff = [sp.Integer(1), sp.Integer(2), sp.Integer(4)]
        
        # Step 4: Generate all scalar * Gamma * rho terms
        collect_terms_c_G_rho = []
        collect_terms_c_rho = []
        
        for rho in rho_symbols:
            for c in coeff:
                collect_terms_c_rho.append(c * rho)
                for G in gamma_symbols:
                    collect_terms_c_G_rho.append(c * G * rho)
        
        

        expr5 = sp.collect(expr4, collect_terms_c_G_rho)
        

        expr6 = expr5.subs({gp_sqr + gm_sqr: sp.Integer(1)})
        
        expr7 = expand(expr6)
        
        expr8 = sp.collect(expr7, collect_terms_c_rho)
        
        
        expr9 = expr8.subs({Gamma_lprm + Gamma_lmrp: GammaLR0})
        

        
        '''
        collected = sp.collect(expr4, i23, evaluate=False)
        coeff = collected.get(i23, sp.sympify(0))
        
        print(coeff)
        
        # Substitute inside the coefficient of i23
        new_coeff = coeff.subs({Gamma_lmrp * gm_sqr: Gamma_lmrp_gm_sqr})
        
        print(new_coeff)
        print()
        
        # Rebuild expression
        expr5 = i23 * new_coeff + collected.get(1, sp.sympify(0))
        '''



        new_dict[key] = expr9

    return new_dict








D_db_int1_x2_simpl_opt2 = simplify_dict_opt2(D_db_int1_x2)
D_db_int2_x2_simpl_opt2 = simplify_dict_opt2(D_db_int2_x2)
D_db_int3_x2_simpl_opt2 = simplify_dict_opt2(D_db_int3_x2)
D_db_int4_x2_simpl_opt2 = simplify_dict_opt2(D_db_int4_x2)
D_db_int5_x2_simpl_opt2 = simplify_dict_opt2(D_db_int5_x2)
D_db_int6_x2_simpl_opt2 = simplify_dict_opt2(D_db_int6_x2)




#print(D_db_int1_x2_simpl_opt2['D_r00'])

#print(D_db_int1_x2_simpl_opt1['D_r00'])

#print(expand(D_db_int1_x2_simpl_opt1['D_r00'] - D_db_int1_x2_simpl_opt2['D_r00']))


##################################################################
# printing


#display_dict(D_db_int1_x2_grp_subs, True)

#display_dict(D_db_int4_x2_grp_subs)


print_dict_cuda_style(D_db_int1_x2_simpl_opt2, for_python_check_opt = True)  

print_dict_cuda_style(D_db_int2_x2_simpl_opt2, for_python_check_opt = True)  

print_dict_cuda_style(D_db_int3_x2_simpl_opt2, for_python_check_opt = True)  

print_dict_cuda_style(D_db_int4_x2_simpl_opt2, for_python_check_opt = True)  

print_dict_cuda_style(D_db_int5_x2_simpl_opt2, for_python_check_opt = True)  

print_dict_cuda_style(D_db_int6_x2_simpl_opt2, for_python_check_opt = True)  







# final expression for cuda program

print_dict_cuda_style(D_db_int1_x2_simpl_opt2)

print_dict_cuda_style(D_db_int2_x2_simpl_opt2)

print_dict_cuda_style(D_db_int3_x2_simpl_opt2)

print_dict_cuda_style(D_db_int4_x2_simpl_opt2)

print_dict_cuda_style(D_db_int5_x2_simpl_opt2)

print_dict_cuda_style(D_db_int6_x2_simpl_opt2)


# end of printing
###########################################################





def compare_sympy_dicts(dict1, dict2, verbose=False):
    """
    Compare two dictionaries with SymPy expressions as values.
    Returns True if they are mathematically equivalent.
    """
    if dict1.keys() != dict2.keys():
        if verbose:
            missing1 = dict1.keys() - dict2.keys()
            missing2 = dict2.keys() - dict1.keys()
            if missing1:
                print("Keys only in dict1:", missing1)
            if missing2:
                print("Keys only in dict2:", missing2)
        return False

    all_equal = True

    for key in dict1:
        expr1 = sp.expand(sp.simplify(dict1[key]))
        expr2 = sp.expand(sp.simplify(dict2[key]))
        
        expr1_modif1 = sp.expand(expr1.subs({gm_sqr: sp.Integer(1) - gp_sqr}))
        expr2_modif1 = sp.expand(expr2.subs({gm_sqr: sp.Integer(1) - gp_sqr}))
        
        expr1_modif2 = sp.expand(expr1_modif1.subs({Gamma_lmrp: GammaLR0 - Gamma_lprm}))
        expr2_modif2 = sp.expand(expr2_modif1.subs({Gamma_lmrp: GammaLR0 - Gamma_lprm}))
        
        #expr1_modif3 = sp.expand(expr1_modif2.subs({gp_sqr: g_p**2, gm_sqr: g_m**2}))
        #expr2_modif3 = sp.expand(expr2_modif2.subs({gp_sqr: g_p**2, gm_sqr: g_m**2}))
        
        #expr1_modif4 = sp.expand(expr1_modif3.subs({Gamma_lprm: GammaL0*g_p**2 + GammaR0*g_m**2, Gamma_lmrp: GammaL0*g_m**2 + GammaR0*g_p**2}))
        #expr2_modif4 = sp.expand(expr2_modif3.subs({Gamma_lprm: GammaL0*g_p**2 + GammaR0*g_m**2, Gamma_lmrp: GammaL0*g_m**2 + GammaR0*g_p**2}))
        
        #expr1_modif5 = sp.expand(expr1_modif4)
        #expr2_modif5 = sp.expand(expr2_modif4)
        
        
        if not sp.simplify(expr1_modif2 - expr2_modif2) == 0:
            if verbose:
                print()
                print("-------------------------")
                print(f"Difference at key: {key}")
                print("dict1:", expr1_modif2)
                print()
                print("dict2:", expr2_modif2)
                print()
                print("Difference:", sp.simplify(expr1_modif2 - expr2_modif2))
                print()
                print()
            all_equal = False

    return all_equal




'''
D_db_int1_x2_simpl_opt1_M1 = D_db_int1_x2_simpl_opt1
D_db_int2_x2_simpl_opt1_M1 = D_db_int2_x2_simpl_opt1
D_db_int3_x2_simpl_opt1_M1 = D_db_int3_x2_simpl_opt1
D_db_int4_x2_simpl_opt1_M1 = D_db_int4_x2_simpl_opt1
D_db_int5_x2_simpl_opt1_M1 = D_db_int5_x2_simpl_opt1
D_db_int6_x2_simpl_opt1_M1 = D_db_int6_x2_simpl_opt1


D_db_int1_x2_simpl_opt2_M1 = D_db_int1_x2_simpl_opt2
D_db_int2_x2_simpl_opt2_M1 = D_db_int2_x2_simpl_opt2
D_db_int3_x2_simpl_opt2_M1 = D_db_int3_x2_simpl_opt2
D_db_int4_x2_simpl_opt2_M1 = D_db_int4_x2_simpl_opt2
D_db_int5_x2_simpl_opt2_M1 = D_db_int5_x2_simpl_opt2
D_db_int6_x2_simpl_opt2_M1 = D_db_int6_x2_simpl_opt2

'''

'''
D_db_int1_x2_simpl_opt1_M2 = D_db_int1_x2_simpl_opt1
D_db_int2_x2_simpl_opt1_M2 = D_db_int2_x2_simpl_opt1
D_db_int3_x2_simpl_opt1_M2 = D_db_int3_x2_simpl_opt1
D_db_int4_x2_simpl_opt1_M2 = D_db_int4_x2_simpl_opt1
D_db_int5_x2_simpl_opt1_M2 = D_db_int5_x2_simpl_opt1
D_db_int6_x2_simpl_opt1_M2 = D_db_int6_x2_simpl_opt1

D_db_int1_x2_simpl_opt2_M2 = D_db_int1_x2_simpl_opt2
D_db_int2_x2_simpl_opt2_M2 = D_db_int2_x2_simpl_opt2
D_db_int3_x2_simpl_opt2_M2 = D_db_int3_x2_simpl_opt2
D_db_int4_x2_simpl_opt2_M2 = D_db_int4_x2_simpl_opt2
D_db_int5_x2_simpl_opt2_M2 = D_db_int5_x2_simpl_opt2
D_db_int6_x2_simpl_opt2_M2 = D_db_int6_x2_simpl_opt2
'''


#winner: M1 + opt2
'''
bool_int1_opt1 = compare_sympy_dicts(D_db_int1_x2_simpl_opt1_M2, D_db_int1_x2_simpl_opt1_M1)
bool_int2_opt1 = compare_sympy_dicts(D_db_int2_x2_simpl_opt1_M2, D_db_int2_x2_simpl_opt1_M1)
bool_int3_opt1 = compare_sympy_dicts(D_db_int3_x2_simpl_opt1_M2, D_db_int3_x2_simpl_opt1_M1)
bool_int4_opt1 = compare_sympy_dicts(D_db_int4_x2_simpl_opt1_M2, D_db_int4_x2_simpl_opt1_M1)
bool_int5_opt1 = compare_sympy_dicts(D_db_int5_x2_simpl_opt1_M2, D_db_int5_x2_simpl_opt1_M1)
bool_int6_opt1 = compare_sympy_dicts(D_db_int6_x2_simpl_opt1_M2, D_db_int6_x2_simpl_opt1_M1)

bool_int1_opt2 = compare_sympy_dicts(D_db_int1_x2_simpl_opt2_M2, D_db_int1_x2_simpl_opt2_M1)
bool_int2_opt2 = compare_sympy_dicts(D_db_int2_x2_simpl_opt2_M2, D_db_int2_x2_simpl_opt2_M1)
bool_int3_opt2 = compare_sympy_dicts(D_db_int3_x2_simpl_opt2_M2, D_db_int3_x2_simpl_opt2_M1)
bool_int4_opt2 = compare_sympy_dicts(D_db_int4_x2_simpl_opt2_M2, D_db_int4_x2_simpl_opt2_M1)
bool_int5_opt2 = compare_sympy_dicts(D_db_int5_x2_simpl_opt2_M2, D_db_int5_x2_simpl_opt2_M1)
bool_int6_opt2 = compare_sympy_dicts(D_db_int6_x2_simpl_opt2_M2, D_db_int6_x2_simpl_opt2_M1)



bool_int1_M1 = compare_sympy_dicts(D_db_int1_x2_simpl_opt1_M1, D_db_int1_x2_simpl_opt2_M1)
bool_int2_M1 = compare_sympy_dicts(D_db_int2_x2_simpl_opt1_M1, D_db_int2_x2_simpl_opt2_M1)
bool_int3_M1 = compare_sympy_dicts(D_db_int3_x2_simpl_opt1_M1, D_db_int3_x2_simpl_opt2_M1)
bool_int4_M1 = compare_sympy_dicts(D_db_int4_x2_simpl_opt1_M1, D_db_int4_x2_simpl_opt2_M1)
bool_int5_M1 = compare_sympy_dicts(D_db_int5_x2_simpl_opt1_M1, D_db_int5_x2_simpl_opt2_M1)
bool_int6_M1 = compare_sympy_dicts(D_db_int6_x2_simpl_opt1_M1, D_db_int6_x2_simpl_opt2_M1)


bool_int1_M2 = compare_sympy_dicts(D_db_int1_x2_simpl_opt1_M2, D_db_int1_x2_simpl_opt2_M2)
bool_int2_M2 = compare_sympy_dicts(D_db_int2_x2_simpl_opt1_M2, D_db_int2_x2_simpl_opt2_M2)
bool_int3_M2 = compare_sympy_dicts(D_db_int3_x2_simpl_opt1_M2, D_db_int3_x2_simpl_opt2_M2)
bool_int4_M2 = compare_sympy_dicts(D_db_int4_x2_simpl_opt1_M2, D_db_int4_x2_simpl_opt2_M2)
bool_int5_M2 = compare_sympy_dicts(D_db_int5_x2_simpl_opt1_M2, D_db_int5_x2_simpl_opt2_M2)
bool_int6_M2 = compare_sympy_dicts(D_db_int6_x2_simpl_opt1_M2, D_db_int6_x2_simpl_opt2_M2)



test_all = all([
    # M1 vs M2 (opt1)
    bool_int1_opt1, bool_int2_opt1, bool_int3_opt1, bool_int4_opt1, bool_int5_opt1, bool_int6_opt1,

    # M1 vs M2 (opt2)
    bool_int1_opt2, bool_int2_opt2, bool_int3_opt2, bool_int4_opt2, bool_int5_opt2, bool_int6_opt2,

    # opt1 vs opt2 (M1)
    bool_int1_M1, bool_int2_M1, bool_int3_M1, bool_int4_M1, bool_int5_M1, bool_int6_M1,

    # opt1 vs opt2 (M2)
    bool_int1_M2, bool_int2_M2, bool_int3_M2, bool_int4_M2, bool_int5_M2, bool_int6_M2,
])

'''






########################
# analyzing single row


expr = D_db_int1_x2['D_r00']

expr1 = sp.sympify(expr)


# Intermediate simplification (optional)
#expr = sp.expand(expr)

expr2 = expr1.subs({g_p*g_m: gp_gm})
expr3 = expr2.subs({g_p**2: gp_sqr, g_m**2: gm_sqr})
expr4 = expand(expr3)







# Step 1: List of predefined Gamma symbols (assumes they exist in the current scope)
gamma_symbols = [
Gamma_lprm, Gamma_lmrp
#sp.Integer(1), sp.Integer(2), g_p*g_m, sp.Integer(2)*g_p*g_m, sp.Integer(4)*g_p*g_m
]


# Step 2: List of all defined r_ij and i_ij symbols
rho_symbols = [
r00, r11, r22, r33,
r01, i01, r02, i02,
r03, i03, r12, i12,
r13, i13, r23, i23
]

# Step 3: Scalar multipliers
coeff = [
    sp.Integer(1),
    sp.Integer(2),
    sp.Integer(4),
]


# Step 4: Generate all scalar * Gamma * rho terms
collect_terms_c_G_rho = []
collect_terms_c_rho = []

for rho in rho_symbols:
    for c in coeff:
        collect_terms_c_rho.append(c * rho)
        for G in gamma_symbols:
            collect_terms_c_G_rho.append(c * G * rho)




expr5 = sp.collect(expr4, collect_terms_c_G_rho)


expr6 = expr5.subs({gp_sqr + gm_sqr: sp.Integer(1)})

expr7 = expand(expr6)

expr8 = sp.collect(expr7, collect_terms_c_rho)


expr9 = expr8.subs({Gamma_lprm + Gamma_lmrp: GammaLR0})


        





######################



















       
        











 



















#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################

























'''
D_sum_adbx2 = sp.Integer(2)*D_sum_adb


D00 = D_sum_adbx2[0,0]
D11 = D_sum_adbx2[1,1]
D22 = D_sum_adbx2[2,2]
D33 = D_sum_adbx2[3,3]

D01_re = re(D_sum_adbx2[0,1])
D01_im = im(D_sum_adbx2[0,1])

D02_re = re(D_sum_adbx2[0,2])
D02_im = im(D_sum_adbx2[0,2])

D03_re = re(D_sum_adbx2[0,3])
D03_im = im(D_sum_adbx2[0,3])


D12_re = re(D_sum_adbx2[1,2])
D12_im = im(D_sum_adbx2[1,2])

D13_re = re(D_sum_adbx2[1,3])
D13_im = im(D_sum_adbx2[1,3])

D23_re = re(D_sum_adbx2[2,3])
D23_im = im(D_sum_adbx2[2,3])
'''




'''
D00, D01, D02, D03 = sp.symbols('D00 D01 D02 D03', real=True)
D11, D12, D13 = sp.symbols('D11 D12 D13', real=True)
D22, D23 = sp.symbols('D22 D23', real=True)
D33 = sp.symbols('D33', real=True)



D_sum_adbx2_00 = 2*Gamma_10*r11 + 2*Gamma_20*r22 + 2*Gamma_30*r33
D_sum_adbx2_11 = -2*Gamma_10*r11 + 2*Gamma_21*r22 + 2*Gamma_31*r33
D_sum_adbx2_22 = -2*Gamma_20*r22 - 2*Gamma_21*r22 + 2*Gamma_32*r33
D_sum_adbx2_33 = 2*r33*(-Gamma_30 - Gamma_31 - Gamma_32)

D_sum_adbx2_01_re = -Gamma_10*re01
D_sum_adbx2_01_im = -Gamma_10*im01

D_sum_adbx2_02_re = re02*(-Gamma_20 - Gamma_21)
D_sum_adbx2_02_im = im02*(-Gamma_20 - Gamma_21)

D_sum_adbx2_03_re = re03*(-Gamma_30 - Gamma_31 - Gamma_32)
D_sum_adbx2_03_im = im03*(-Gamma_30 - Gamma_31 - Gamma_32)

D_sum_adbx2_12_re = re12*(-Gamma_10 - Gamma_20 - Gamma_21)
D_sum_adbx2_12_im = im12*(-Gamma_10 - Gamma_20 - Gamma_21)

D_sum_adbx2_13_re = re13*(-Gamma_10 - Gamma_30 - Gamma_31 - Gamma_32)
D_sum_adbx2_13_im = im13*(-Gamma_10 - Gamma_30 - Gamma_31 - Gamma_32)

D_sum_adbx2_23_re = re23*(-Gamma_20 - Gamma_21 - Gamma_30 - Gamma_31 - Gamma_32)
D_sum_adbx2_23_im = im23*(-Gamma_20 - Gamma_21 - Gamma_30 - Gamma_31 - Gamma_32)
'''





'''
D_sum_adbx2_00 = 2*Gamma_10*r11 + 2*Gamma_20*r22 + 2*Gamma_30*r33
D_sum_adbx2_11 = -2*Gamma_10*r11 + 2*Gamma_21*r22 + 2*Gamma_31*r33
D_sum_adbx2_22 = -2*Gamma_20*r22 - 2*Gamma_21*r22 + 2*Gamma_32*r33
D_sum_adbx2_33 = 2*r33*(-Gamma_30 - Gamma_31 - Gamma_32)


G01 = Gamma_10

G02 = Gamma_20 + Gamma_21

G03 = Gamma_30 + Gamma_31 + Gamma_32

G12 = Gamma_10 + Gamma_20 + Gamma_21

G13 = Gamma_10 + Gamma_30 + Gamma_31 + Gamma_32

G23 = Gamma_20 + Gamma_21 + Gamma_30 + Gamma_31 + Gamma_32




D_sum_adbx2_01_re = -re01*G01
D_sum_adbx2_01_im = -im01*G01

D_sum_adbx2_02_re = -re02*G02
D_sum_adbx2_02_im = -im02*G02

D_sum_adbx2_03_re = -re03*G03
D_sum_adbx2_03_im = -im03*G03

D_sum_adbx2_12_re = -re12*G12
D_sum_adbx2_12_im = -im12*G12

D_sum_adbx2_13_re = -re13*G13
D_sum_adbx2_13_im = -im13*G13

D_sum_adbx2_23_re = -re23*G23
D_sum_adbx2_23_im = -im23*G23





D_sum_adbx2_2 = - sp.Matrix([
    [-2*Gamma_10*r11 - 2*Gamma_20*r22 - 2*Gamma_30*r33, G01*(re01 + I*im01),                              G02*(re02 + I*im02),                               G03*(re03 + I*im03)                   ],
    [G01*(re01 - I*im01),                               2*Gamma_10*r11 - 2*Gamma_21*r22 - 2*Gamma_31*r33, G12*(re12 + I*im12),                               G13*(re13 + I*im13)                   ],
    [G02*(re02 - I*im02),                               G12*(re12 - I*im12),                              2*Gamma_20*r22 + 2*Gamma_21*r22 - 2*Gamma_32*r33,  G23*(re23 + I*im23)                   ],
    [G03*(re03 - I*im03),                               G13*(re13 - I*im13),                              G23*(re23 - I*im23),                               2*r33*(Gamma_30 + Gamma_31 + Gamma_32)]
])


print(sp.simplify(D_sum_adbx2) == sp.simplify(D_sum_adbx2_2))


subs_dict = {
    Gamma_10: 0,
    Gamma_32: 0
}



D_sum_adbx2_sub = D_sum_adbx2_2.subs(subs_dict)












G_01, G_02, G_03 = sp.symbols('G_01 G_02 G_03', positive=True, real=True)
G_12, G_13, G_23 = sp.symbols('G_12 G_13 G_23', positive=True, real=True)


D_sum_adbx2_3 = - sp.Matrix([
    [-2*Gamma_10*r11 - 2*Gamma_20*r22 - 2*Gamma_30*r33,  G_01*(re01 + I*im01),                             G_02*(re02 + I*im02),                              G_03*(re03 + I*im03)                   ],
    [G_01*(re01 - I*im01),                               2*Gamma_10*r11 - 2*Gamma_21*r22 - 2*Gamma_31*r33, G_12*(re12 + I*im12),                              G_13*(re13 + I*im13)                   ],
    [G_02*(re02 - I*im02),                               G_12*(re12 - I*im12),                             2*Gamma_20*r22 + 2*Gamma_21*r22 - 2*Gamma_32*r33,  G_23*(re23 + I*im23)                   ],
    [G_03*(re03 - I*im03),                               G_13*(re13 - I*im13),                             G_23*(re23 - I*im23),                              2*r33*(Gamma_30 + Gamma_31 + Gamma_32)]
])


D_sum_adbx2_3_re = re(D_sum_adbx2_3)
D_sum_adbx2_3_im = im(D_sum_adbx2_3)


D_dbx2 = U_matrix*D_sum_adbx2_3*U_matrix.T

D_dbx2 = U_matrix*D_sum_adbx2_3_re*U_matrix.T
D_dbx2_im = U_matrix*D_sum_adbx2_3_im*U_matrix.T










# rho entries
D00, D11, D22, D33 = sp.symbols('D00 D11 D22 D33', real=True)
D01re, D01im = sp.symbols('D01re D01im', real=True)
D02re, D02im = sp.symbols('D02re D02im', real=True)
D03re, D03im = sp.symbols('D03re D03im', real=True)
D12re, D12im = sp.symbols('D12re D12im', real=True)
D13re, D13im = sp.symbols('D13re D13im', real=True)
D23re, D23im = sp.symbols('D23re D23im', real=True)



# dissipator in adiabatic basis

D_adb = sp.Matrix([
    [D00,               D01re + I*D01im,  D02re + I*D02im,  D03re + I*D03im],
    [D01re - I*D01im,   D11,              D12re + I*D12im,  D13re + I*D13im],
    [D02re - I*D02im,   D12re - I*D12im,  D22,              D23re + I*D23im],
    [D03re - I*D03im,   D13re - I*D13im,  D23re - I*D23im,  D33]
])


# Split D_adb into real and imaginary parts explicitly
D_adb_re = re(D_adb)
D_adb_im = im(D_adb)

# Then transform each part separately
D_db_re = U_matrix * D_adb_re * U_matrix.T
D_db_im = U_matrix * D_adb_im * U_matrix.T

D_db = U_matrix*D_adb*U_matrix.T


#D_db_re = re(D_db)
#D_db_im = im(D_db)

print(expand(D_db_re - re(D_db)) == sp.zeros(4, 4))
print(expand(D_db_im - im(D_db)) == sp.zeros(4, 4))




print(expand(D_db_im[0,0]))
print(expand(D_db_im[1,1]))
print(expand(D_db_im[2,2]))
print(expand(D_db_im[3,3]))



print()
print("drho_out_0 +=", expand(D_db_re[0,0]))
print()
print("drho_out_1 +=", expand(D_db_re[1,1]))
print()
print("drho_out_2 +=", expand(D_db_re[2,2]))
print()
print("drho_out_3 +=", expand(D_db_re[3,3]))
print()

print("drho_out_4 +=", expand(D_db_re[0,1]))
print()
print("drho_out_5 +=", expand(D_db_im[0,1]))
print()

print("drho_out_6 +=", expand(D_db_re[0,2]))
print()
print("drho_out_7 +=", expand(D_db_im[0,2]))
print()

print("drho_out_8 +=", expand(D_db_re[0,3]))
print()
print("drho_out_9 +=", expand(D_db_im[0,3]))
print()

print("drho_out_10 +=", expand(D_db_re[1,2]))
print()
print("drho_out_11 +=", expand(D_db_im[1,2]))
print()

print("drho_out_12 +=", expand(D_db_re[1,3]))
print()
print("drho_out_13 +=", expand(D_db_im[1,3]))
print()

print("drho_out_14 +=", expand(D_db_re[2,3]))
print()
print("drho_out_15 +=", expand(D_db_im[2,3]))
print()
'''











'''
with open("L_and_D_results.txt", "w") as fout:
    # Energy-type dissipators
    for i, j, Gamma in energy_pairs:
        # L_adb WITHOUT sqrt(Gamma)
        L_adb = L_adb_en(i, j, 1)  # unit coefficient
        
        # L_db WITHOUT sqrt(Gamma)
        L_db_no_coeff = U_matrix * L_adb * U_matrix.H
        
        # Dissipator WITHOUT Gamma factor
        D_no_coeff = dissipator(L_db_no_coeff, rho)
        
        # Print L_adb
        fout.write(f"L_en_{i}{j}_adb = {sp.simplify(L_adb)}\n\n")
        
        # Print L_db with sqrt(Gamma) factored out
        fout.write(f"L_en_{i}{j}_db = sqrt({Gamma}) * Matrix([\n")
        for row in L_db_no_coeff.tolist():
            fout.write(f"    {row},\n")
        fout.write("])\n\n")
        

        # Print dissipator with Gamma factored out
        fout.write(f"D_en_{i}{j} = {Gamma} * Matrix([\n")
        for row in D_no_coeff.tolist():
            simplified_row = [sp.simplify(x) for x in row]
            fout.write(f"    {simplified_row},\n")
        fout.write("])\n")
        fout.write("\n" + "="*60 + "\n\n")

    # Phase-type dissipators
    for i, j, Gamma in phase_pairs:
        L_adb = L_adb_phi(i, j, 1)  # no Gamma coeff
        
        L_db_no_coeff = U_matrix * L_adb * U_matrix.H
        D_no_coeff = dissipator(L_db_no_coeff, rho)
        
        fout.write(f"L_phi_{i}{j}_adb = {sp.simplify(L_adb)}\n\n")
        
        fout.write(f"L_phi_{i}{j}_db = sqrt({Gamma}) * Matrix([\n")
        for row in L_db_no_coeff.tolist():
            fout.write(f"    {row},\n")
        fout.write("])\n\n")
        
        fout.write(f"D_phi_{i}{j} = {Gamma} * Matrix([\n")
        for row in D_no_coeff.tolist():
            simplified_row = [sp.simplify(x) for x in row]
            fout.write(f"    {simplified_row},\n")
        fout.write("])\n")
        fout.write("\n" + "="*60 + "\n\n")
'''