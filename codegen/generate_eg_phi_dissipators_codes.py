########################################
# codegen/generate_eg_phi_dissipators_codes.py
########################################

import sympy as sp
from sympy import Integer, Rational, re, im, simplify, expand
import IPython


#sp.init_printing(line_width=120)



# Define symbols (assume real for simplicity)
#gamma_plus, gamma_minus = sp.symbols('gamma_plus gamma_minus', real=True, positive=True)

I = sp.I

# rho entries
r00, r11, r22, r33 = sp.symbols('r00 r11 r22 r33', real=True)
r01, i01 = sp.symbols('r01 i01', real=True)
r02, i02 = sp.symbols('r02 i02', real=True)
r03, i03 = sp.symbols('r03 i03', real=True)
r12, i12 = sp.symbols('r12 i12', real=True)
r13, i13 = sp.symbols('r13 i13', real=True)
r23, i23 = sp.symbols('r23 i23', real=True)

# rho density matrix (Hermitian) in the diabatic basis
rho = sp.Matrix([
    [r00,           r01 + I*i01, r02 + I*i02, r03 + I*i03],
    [r01 - I*i01, r11,           r12 + I*i12, r13 + I*i13],
    [r02 - I*i02, r12 - I*i12, r22,           r23 + I*i23],
    [r03 - I*i03, r13 - I*i13, r23 - I*i23, r33]
])



g_p = sp.symbols('g_p', real=True, positive=True)
g_m = sp.symbols('g_m', real=True, positive=True)


# Define operator O (4x4 matrix)
O = sp.Matrix([
    [0, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
])

# Define |g> and |e>
g = sp.Matrix([0, g_p, g_m, 0])
e = sp.Matrix([0, -g_m, g_p, 0])

# Calculate O|e>
O_e = O * e

# Calculate matrix element <g| O |e> = g.H * (O|e>)
# .H is Hermitian transpose, for real vectors it's just transpose
M = g.T * O_e

# Simplify and display the result
M_simplified = sp.simplify(M[0])
print(f"M(ε) = {M_simplified}")






###############
#

Gamma_eg = sp.symbols('Gamma_eg', real=True, positive=True)
Gamma_phi = sp.symbols('Gamma_phi', real=True, positive=True)


L_eg_db = sp.sqrt(Gamma_eg) * g * e.T

L_phi_db = sp.simplify(sp.sqrt(sp.Rational(1,2)*Gamma_phi) * (e*e.T - g*g.T))






####################





def dissipator_db(L_db):
    

    M_db = L_db.T*L_db
    
    
    dissipator_db = L_db * rho * L_db.T - sp.Rational(1,2) * (M_db * rho + rho * M_db)
    
    dissipator_db = sp.expand(dissipator_db)
    
    return dissipator_db


D_eg = dissipator_db(L_eg_db)

D_phi = dissipator_db(L_phi_db)




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



check_is_hermitian_D_eg = is_hermitian_matrix_elementwise(D_eg)
check_is_hermitian_D_phi = is_hermitian_matrix_elementwise(D_phi)
print(check_is_hermitian_D_eg)
print(check_is_hermitian_D_phi)




D_eg_x2 = Integer(2)*D_eg

D_eg_x2_re = re(D_eg_x2)
D_eg_x2_im = im(D_eg_x2)


print(D_eg_x2_im[0,0])
print(D_eg_x2_im[1,1])
print(D_eg_x2_im[2,2])
print(D_eg_x2_im[3,3])


check_D_eg_x2_im_00_is_zero = (D_eg_x2_im[0,0] == 0)
check_D_eg_x2_im_11_is_zero = (D_eg_x2_im[1,1] == 0)
check_D_eg_x2_im_22_is_zero = (D_eg_x2_im[2,2] == 0)
check_D_eg_x2_im_33_is_zero = (D_eg_x2_im[3,3] == 0)




D_eg_x2_dict = {
    "D_r00": D_eg_x2_re[0,0],
    "D_r11": D_eg_x2_re[1,1],
    "D_r22": D_eg_x2_re[2,2],
    "D_r33": D_eg_x2_re[3,3],
    "D_r01": D_eg_x2_re[0,1],
    "D_i01": D_eg_x2_im[0,1],
    "D_r02": D_eg_x2_re[0,2],
    "D_i02": D_eg_x2_im[0,2],
    "D_r03": D_eg_x2_re[0,3],
    "D_i03": D_eg_x2_im[0,3],
    "D_r12": D_eg_x2_re[1,2],
    "D_i12": D_eg_x2_im[1,2],
    "D_r13": D_eg_x2_re[1,3],
    "D_i13": D_eg_x2_im[1,3],
    "D_r23": D_eg_x2_re[2,3],
    "D_i23": D_eg_x2_im[2,3],
}






D_phi_x4 = Integer(4)*D_phi

D_phi_x4_re = re(D_phi_x4)
D_phi_x4_im = im(D_phi_x4)


print(D_phi_x4_im[0,0])
print(D_phi_x4_im[1,1])
print(D_phi_x4_im[2,2])
print(D_phi_x4_im[3,3])


check_D_phi_x4_im_00_is_zero = (D_phi_x4_im[0,0] == 0)
check_D_phi_x4_im_11_is_zero = (D_phi_x4_im[1,1] == 0)
check_D_phi_x4_im_22_is_zero = (D_phi_x4_im[2,2] == 0)
check_D_phi_x4_im_33_is_zero = (D_phi_x4_im[3,3] == 0)




D_phi_x4_dict = {
    "D_r00": D_phi_x4_re[0,0],
    "D_r11": D_phi_x4_re[1,1],
    "D_r22": D_phi_x4_re[2,2],
    "D_r33": D_phi_x4_re[3,3],
    "D_r01": D_phi_x4_re[0,1],
    "D_i01": D_phi_x4_im[0,1],
    "D_r02": D_phi_x4_re[0,2],
    "D_i02": D_phi_x4_im[0,2],
    "D_r03": D_phi_x4_re[0,3],
    "D_i03": D_phi_x4_im[0,3],
    "D_r12": D_phi_x4_re[1,2],
    "D_i12": D_phi_x4_im[1,2],
    "D_r13": D_phi_x4_re[1,3],
    "D_i13": D_phi_x4_im[1,3],
    "D_r23": D_phi_x4_re[2,3],
    "D_i23": D_phi_x4_im[2,3],
}



gp_sqr = sp.symbols('gp_sqr', real=True, positive=True)
gm_sqr = sp.symbols('gm_sqr', real=True, positive=True)
gp_gm  = sp.symbols('gp_gm', real=True, positive=True)




def simplify_dict_eg(expr_dict):

    new_dict = {}

    for key, expr in expr_dict.items():
        expr1 = sp.sympify(expr)

        
        if key in ('D_r11', 'D_r22'):
            collect_terms = [
                Integer(2)*Gamma_eg*g_p*g_m*r12
            ]
        elif key in ('D_r01'):
            collect_terms = [
                Gamma_eg*r01*g_m**2,
                Gamma_eg*r02*g_p*g_m
            ]
        elif key in ('D_i01'):
            collect_terms = [
                Gamma_eg*i01*g_m**2,
                Gamma_eg*i02*g_p*g_m
            ]
        elif key in ('D_r02'):
            collect_terms = [
                Gamma_eg*r01*g_p*g_m,
                Gamma_eg*r02*g_p**2
            ]
        elif key in ('D_i02'):
            collect_terms = [
                Gamma_eg*i01*g_p*g_m,
                Gamma_eg*i02*g_p**2
            ]
        elif key in ('D_r12'):
            collect_terms = [
                Gamma_eg*r11*g_p*g_m,
                -Gamma_eg*r12,
                Gamma_eg*r22*g_p*g_m
            ]
        elif key in ('D_i12'):
            collect_terms = [
                -Gamma_eg*i12
            ]
        elif key in ('D_r13'):
            collect_terms = [
                Gamma_eg*r13*g_m**2,
                Gamma_eg*r23*g_p*g_m
            ]
        elif key in ('D_i13'):
            collect_terms = [
                Gamma_eg*i13*g_m**2,
                Gamma_eg*i23*g_p*g_m
            ]
        elif key in ('D_r23'):
            collect_terms = [
                Gamma_eg*r13*g_p*g_m,
                Gamma_eg*r23*g_p**2
            ]
        elif key in ('D_i23'):
            collect_terms = [
                Gamma_eg*i13*g_p*g_m,
                Gamma_eg*i23*g_p**2
            ]
        else:
            collect_terms = []
            

        expr5 = sp.collect(expr1, collect_terms, evaluate=True)
        
        expr6 = expr5.subs({g_p*g_m: gp_gm}).subs({g_p**2: gp_sqr, g_m**2: gm_sqr})

        #expr7 = expr6.subs({g_p**2 + g_m**2: Integer(1)})
        expr7 = expr6.subs({gp_sqr + gm_sqr: Integer(1)})

        if key in ('D_r12'):
            expr8 = expr7.subs({ gp_sqr**2 + 6*gp_gm**2 + gm_sqr**2: sp.Integer(1) + Integer(4)*gp_gm**2 })
            expr9 = expr8.subs({ 3*gp_sqr + gm_sqr: Integer(1) + Integer(2)*gp_sqr }).subs({ gp_sqr + 3*gm_sqr: sp.Integer(1) + Integer(2)*gm_sqr })
        elif key in ('D_i12'):
            expr9 = expr7.subs({ gp_sqr**2 + 2*gp_gm**2 + gm_sqr**2: sp.Integer(1) })
        else:
            expr9 = expr7
            


        new_dict[key] = expr9

    return new_dict









def simplify_dict_phi(expr_dict):

    new_dict = {}

    for key, expr in expr_dict.items():
        expr1 = sp.sympify(expr)
      
        collect_terms = [
            Gamma_phi*r01,
            Gamma_phi*i01,
            Gamma_phi*r02,
            Gamma_phi*i02,
            
            Integer(4)*Gamma_phi*r12,
            Integer(4)*Gamma_phi*r11*g_m*g_p,
            Integer(4)*Gamma_phi*r22*g_m*g_p,
            
            Gamma_phi*i12*Integer(4),
            Gamma_phi*r13,
            Gamma_phi*i13,
            Gamma_phi*r23,
            Gamma_phi*i23
        ]
        
        
        expr5 = sp.collect(expr1, collect_terms, evaluate=True)
        
        expr6 = expr5.subs({g_p*g_m: gp_gm}).subs({g_p**2: gp_sqr, g_m**2: gm_sqr})
        
        expr7 = expr6.subs({ gp_sqr**2 + 2*gp_gm**2 + gm_sqr**2: sp.Integer(1) })
        #expr7 = expr6.subs({ g_m**4 + 2*g_m**2*g_p**2 + g_p**4: sp.Integer(1)})


        new_dict[key] = expr7

    return new_dict




D_eg_x2_dict_simpl  = simplify_dict_eg(D_eg_x2_dict)


D_phi_x4_dict_simpl = simplify_dict_phi(D_phi_x4_dict)




def print_dict_cuda_style(expr_dict, divided_by, for_python_check_opt = False):
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
            print("tmp = 0")
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
            term_str = str(term).replace("*", " * ").replace(" *  * 2", "**2")
            
            if for_python_check_opt == False:
                term_str = term_str.replace("gp_sqr**2", "gp_sqr * gp_sqr")
                term_str = term_str.replace("gm_sqr**2", "gm_sqr * gm_sqr")
                term_str = term_str.replace("gp_gm**2", "gp_gm * gp_gm")
                
            if first_term:
                print(f"tmp = {term_str};")
                first_term = False
            else:
                print(f"tmp += {term_str};")
        
        
        
        if divided_by == 2:
        
            if for_python_check_opt == False:
                print("tmp *= 0.5f;")
                print(f"drho_out_{key} += tmp;")
                print(f"d_log_buffer[t_idx_substep].drho_out_{key} = tmp;\n\n")
            elif for_python_check_opt == True:
                print("tmp *= half")
                print(f"drho_out_{key} = tmp\n\n")

        elif divided_by == 4:
            
            if for_python_check_opt == False:
                print("tmp *= 0.25f;")
                print(f"drho_out_{key} += tmp;")
                print(f"d_log_buffer[t_idx_substep].drho_out_{key} = tmp;\n\n")
            elif for_python_check_opt == True:
                print("tmp *= quarter")
                print(f"drho_out_{key} = tmp\n\n")




#print_dict_cuda_style(D_eg_x2_dict_simpl, divided_by=2, for_python_check_opt = True)  

#print_dict_cuda_style(D_phi_x4_dict_simpl, divided_by=4, for_python_check_opt = True)  



#print_dict_cuda_style(D_eg_x2_dict_simpl, divided_by=2, for_python_check_opt = False)  

#print_dict_cuda_style(D_phi_x4_dict_simpl, divided_by=4, for_python_check_opt = False)  











'''
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"


from sympy import symbols, expand, init_printing

x = symbols('x')
expr = expand((x + 1)**20)

init_printing(use_latex=True, num_columns=600)

print(expr)  # This will now show on a single line (if short enough)
'''


###############################################
# checking cuda program:
    
    
Gamma_eg_loc = Gamma_eg
Gamma_eg_half_loc = Rational(1,2) * Gamma_eg_loc


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.5f;
# //drho_out_D_eg_r00 += tmp;


tmp = 0
tmp = -gm_sqr * gm_sqr * r11;
tmp += gp_sqr * gp_sqr * r22;
tmp += gp_gm * r12 * gm_sqr;
tmp += -gp_gm * r12 * gp_sqr;
tmp *= 2 * Gamma_eg_half_loc;
drho_out_D_eg_r11 = tmp;


tmp = 0
tmp = -gp_sqr * gp_sqr * r22;
tmp += gm_sqr * gm_sqr * r11;
tmp += -gp_gm * r12 * gm_sqr;
tmp += gp_gm * r12 * gp_sqr;
tmp *= 2 * Gamma_eg_half_loc;
drho_out_D_eg_r22 = tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.5f;
# //drho_out_D_eg_r33 += tmp;


tmp = 0
tmp = gp_gm * r02;
tmp += -gm_sqr * r01;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_r01 = tmp;


tmp = 0
tmp = gp_gm * i02;
tmp += -gm_sqr * i01;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_i01 = tmp;


tmp = 0
tmp = gp_gm * r01;
tmp += -gp_sqr * r02;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_r02 = tmp;


tmp = 0
tmp = gp_gm * i01;
tmp += -gp_sqr * i02;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_i02 = tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.5f;
# //drho_out_D_eg_r03 += tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.5f;
# //drho_out_D_eg_i03 += tmp;


tmp = 0
tmp = -4 * r12 * gp_gm * gp_gm;
tmp += -r12;
tmp += 2 * gp_gm * r11 * gm_sqr;
tmp += gp_gm * r11;
tmp += 2 * gp_gm * r22 * gp_sqr;
tmp += gp_gm * r22;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_r12 = tmp;


tmp = 0
tmp = -i12;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_i12 = tmp;


tmp = 0
tmp = gp_gm * r23;
tmp += -gm_sqr * r13;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_r13 = tmp;


tmp = 0
tmp = gp_gm * i23;
tmp += -gm_sqr * i13;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_i13 = tmp;


tmp = 0
tmp = gp_gm * r13;
tmp += -gp_sqr * r23;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_r23 = tmp;


tmp = 0
tmp = gp_gm * i13;
tmp += -gp_sqr * i23;
tmp *= Gamma_eg_half_loc;
drho_out_D_eg_i23 = tmp;













Gamma_phi_loc = Gamma_phi


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.25f;
# //drho_out_D_phi_r00 += tmp;


tmp = 0
tmp = -gp_gm * gp_gm * r11;
tmp += -r12 * gm_sqr * gp_gm;
tmp += r12 * gp_gm * gp_sqr;
tmp += gp_gm * gp_gm * r22;
tmp *= 2 * Gamma_phi_loc;
drho_out_D_phi_r11 = tmp;


tmp = 0
tmp = -gp_gm * gp_gm * r22;
tmp += r12 * gm_sqr * gp_gm;
tmp += -r12 * gp_gm * gp_sqr;
tmp += gp_gm * gp_gm * r11;
tmp *= 2 * Gamma_phi_loc;
drho_out_D_phi_r22 = tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.25f;
# //drho_out_D_phi_r33 += tmp;


tmp = 0
tmp = -Gamma_phi_loc * r01;
tmp *= 0.25
drho_out_D_phi_r01 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * i01;
tmp *= 0.25
drho_out_D_phi_i01 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * r02;
tmp *= 0.25
drho_out_D_phi_r02 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * i02;
tmp *= 0.25
drho_out_D_phi_i02 = tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.25f;
# //drho_out_D_phi_r03 += tmp;


# //tmp = 0.0f;
# //tmp = 0;
# //tmp *= 0.25f;
# //drho_out_D_phi_i03 += tmp;


tmp = 0
tmp = -r12 * gm_sqr * gm_sqr;
tmp += 2 * r12 * gp_gm * gp_gm;
tmp += -r12 * gp_sqr * gp_sqr;
tmp += -gp_gm * r11 * gm_sqr;
tmp += gp_gm * r11 * gp_sqr;
tmp += gp_gm * r22 * gm_sqr;
tmp += -gp_gm * r22 * gp_sqr;
tmp *= Gamma_phi_loc;
drho_out_D_phi_r12 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * i12;
drho_out_D_phi_i12 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * r13;
tmp *= 0.25
drho_out_D_phi_r13 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * i13;
tmp *= 0.25
drho_out_D_phi_i13 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * r23;
tmp *= 0.25
drho_out_D_phi_r23 = tmp;


tmp = 0
tmp = -Gamma_phi_loc * i23;
tmp *= 0.25
drho_out_D_phi_i23 = tmp;











drho_out_D_eg = {
    'D_r00': Integer(0),
    'D_r11': drho_out_D_eg_r11,
    'D_r22': drho_out_D_eg_r22,
    'D_r33': Integer(0),
    'D_r01': drho_out_D_eg_r01,
    'D_i01': drho_out_D_eg_i01,
    'D_r02': drho_out_D_eg_r02,
    'D_i02': drho_out_D_eg_i02,
    'D_r03': Integer(0),
    'D_i03': Integer(0),
    'D_r12': drho_out_D_eg_r12,
    'D_i12': drho_out_D_eg_i12,
    'D_r13': drho_out_D_eg_r13,
    'D_i13': drho_out_D_eg_i13,
    'D_r23': drho_out_D_eg_r23,
    'D_i23': drho_out_D_eg_i23
}


drho_out_D_phi = {
    'D_r00': Integer(0),
    'D_r11': drho_out_D_phi_r11,
    'D_r22': drho_out_D_phi_r22,
    'D_r33': Integer(0),
    'D_r01': drho_out_D_phi_r01,
    'D_i01': drho_out_D_phi_i01,
    'D_r02': drho_out_D_phi_r02,
    'D_i02': drho_out_D_phi_i02,
    'D_r03': Integer(0),
    'D_i03': Integer(0),
    'D_r12': drho_out_D_phi_r12,
    'D_i12': drho_out_D_phi_i12,
    'D_r13': drho_out_D_phi_r13,
    'D_i13': drho_out_D_phi_i13,
    'D_r23': drho_out_D_phi_r23,
    'D_i23': drho_out_D_phi_i23
}



print()

for key in D_eg_x2_dict_simpl:
    
    
    difference = sp.simplify(D_eg_x2_dict_simpl[key] - drho_out_D_eg[key]*Integer(2))
    if difference == 0:
        print(f"{key} matches.")
    else:
        print(f"{key} DOES NOT match.")
        print(f"Difference: {difference}\n")


print()

for key in D_phi_x4_dict_simpl:
    
    
    difference = sp.simplify(D_phi_x4_dict_simpl[key] - drho_out_D_phi[key]*Integer(4))
    if difference == 0:
        print(f"{key} matches.")
    else:
        print(f"{key} DOES NOT match.")
        print(f"Difference: {difference}\n")




