import tkinter as tk
from tkinter import *
import time
import fractions
import random
import webbrowser





π = 3.141592653589793238462643383279
e = 2.718281828459045235360287471352
radian = 180 / π





#Removes the .0 off of a number if it has one to turn it into an integer
def un_dec(x):
    try:
        if int(x) == x:
            x = int(x)
    except:
        pass
    return(x)








# FACTORIAL FUNCTION

#Decreases the number by one and multiplies it by the total until the number reaches 1
def fac(n):
    n = int(n)
    if n == 0:
        number = 1
    else:
        number = n
        while n >= 2:
            n -= 1
            number = number * n
    return number








# MISCELLANEOUS FUNCTIONS

#Makes the decimal of the number in the brackets become recurring
#Works by finding what the fractions of the recurring number would be then evaluating the fraction
def rec(number):
    negative = False
    if number < 0:
        negative = True
        number = number*(-1)
    numerator = number
    number = list(str(number))
    decimal_place = number.index('.')
    multiplier = 10**(len(number) - decimal_place - 1)
    numbers_decimals = numerator - int(numerator)
    if numbers_decimals < 0:
        numbers_decimals += 1
    numerator = multiplier * numerator + numbers_decimals - numerator
    denominator = multiplier - 1
    if not negative:
        fraction = fractions.Fraction(numerator/denominator).limit_denominator()
    else:
        fraction = - fractions.Fraction(numerator/denominator).limit_denominator()
    fraction_str = str(fraction.numerator) + "/" + str(fraction.denominator)
    decimal = eval(fraction_str)
    return decimal
    

#Finds a random integer between 1 and 1000
def ran_int():
    random_integer = random.randint(0,1000)
    return random_integer

#Finds a random integer between two numbers
def ran(x,y):
    random_number = random.randint(x,y)
    return random_number

#Converts a percentage to a decimal
def per(x):
    x = x / 100
    return x

#Finds the absolute value of a number by either pythagoras if it is a complex number
#or by multiplying it by -1 if the number is negative
def Abs(x):
    if isinstance(x, complex):
        x = ((x.real)**2 + (x.imag)**2)**0.5
    elif isinstance(x, list):
        x = vector_abs(x)
    elif x < 0:
        x = x * -1
    return x








# SQUARE ROOT WITH SURDS FUNCTION


#Square root function that can output a surd under certain conditions
def sqrt(x):

    decimalise = single_root.get()
    rooted_factors = []
    imaginary = ''
    if x < 0:
        imaginary = '*j'
        x = x * -1
    target_number = x
    intx = int(x)
    #Turns number into a fraction if it is not an integer
    if not intx == x:
        fraction1 = str(fractions.Fraction(x).limit_denominator())
        fraction1 = fraction1.split('/')
        fraction1[0] = int(fraction1[0]) * int(fraction1[1])
        target_number = fraction1[0]
    else:
        x = intx
       
    if len(str(target_number)) > 10 or int(x ** 0.5) == x ** 0.5 or not decimalise or x < 0:
        #Calculates the number not as surd
        surd = x ** 0.5
        if int(surd) == surd:
            surd = int(surd)
        if imaginary == '*j':
            surd = surd * 1j
            
    else:
        #Finds the factors of the number that are square numbers
        place = int(round((target_number ** 0.5),0) + 1)
        remaining = target_number
        while place > 1:
            if remaining % place == 0:
                second_factor = False
                z = remaining / place
                if remaining % z == 0 and not z == place:
                    z = z ** 0.5
                    second_factor = True
                y = place ** 0.5
                if target_number % y == 0:
                    rooted_factors.append(y)
                    remaining = remaining / (y ** 2)
                if target_number % z == 0 and second_factor:
                    rooted_factors.append(z)
                    remaining = remaining / (z ** 2)
            place -=1
            
        rooted = 1
        left = target_number
        if int(left) == left:
            left = int(left)
        #Finds which numbers need to go within and outside of the surd
        for i in rooted_factors:
            rooted = int(rooted * i)
            left = int(left / (i ** 2))

        #Outputs the surd which is a fraction if the input number is not an integer   
        if intx == x:
            surd = "√(" + str(left) + ")" + imaginary
            if not rooted == 1:
                surd = str(rooted) + surd
                
        else:
            fraction2 = str(fractions.Fraction(rooted / int(fraction1[1])).limit_denominator())
            fraction2 = fraction2.split('/')
            if not left == 1:
                surd = "√(" + str(left) + ")/" + fraction2[1] + imaginary
            else:
                surd = '/' + fraction2[1]
            if not int(fraction2[0]) == 1:
                surd = fraction2[0] + surd
                
    return surd








# BINOMIAL FUNCTIONS

#The number of different, unordered combinations of r objects from a set of n objects.
def nCr(n,r):
    n = int(n)
    r = int(r)
    pascal = fac(n) / ( fac(r) * fac(n - r) )
    return pascal

#The number of different, ordered sets of r objects that can be chosen from a total of n objects.
def nPr(n,r):
    n = int(n)
    r = int(r)
    probability = fac(n) / fac(n -r)
    return probability


#The probability of choosing X items from a sample of N with a probability of p
def binomialP(X,N,p):
    X = int(X)
    N = int(N)
    p = float(p)
    result = nCr(N,X) * (p**X) * ((1-p)**(N-X))
    return result


#Calculate the probability of choosing up to X items from a sample of N with a probability of p
def binomialC(X,N,p):
    X = int(X)
    N = int(N)
    p = float(p)
    r = 0
    result = 0
    while r <= X:
        result += nCr(N,r) * (p**r) * ((1-p)**(N-r))
        r += 1
    return result








# TRIGONOMETRIC FUNCTIONS

def sin(x):
    degrees = not_radians.get()
    if degrees:
        x = x / radian
    #Brings the angle into the range between -360 and 360 degrees
    while x > 2*π:
        x -= 2*π
    while x < -2*π:
        x += 2*π
    if x > π and x <= 2*π:
        x = -(π - (x - π))
    if x < -π and x >= -2*π:
        x = -(-π - (x + π))
    n = 1
    angle = x
    #Uses the maclaurin series for sine to approximate sine
    while n < 50:
        angle += (((-1)**n) / fac(2*n+1)) * (x**(2*n+1))
        n += 1
    return round(angle,15)
        

def cos(x):
    #Uses the same idea as with sine
    degrees = not_radians.get()
    if degrees:
        x = x / radian
    while x > 2*π:
        x -= 2*π
    while x < -2*π:
        x += 2*π
    if x > π and x <= 2*π:
        x = π - (x - π)
    if x < -π and x >= -2*π:
        x = -π - (x + π)
    n = 1
    angle = 1
    while n < 50:
        angle += (((-1)**n) / fac(2*n)) * (x**(2*n))
        n += 1
    return round(angle,15)


def tan(x):
    #Uses the identity of tangent
    angle = sin(x) / cos(x)
    return round(angle,15)



def arcsin(x):
    degrees = not_radians.get()
    #Arcsine cannot be well approximated at values for sine that approach 1
    #so a higher degree of terms in the series is used for these values but this makes the code run slower
    if x < 1 and x > -1:
        if x > 0.95 or x < -0.95:
            n_max = 500
        else:
            n_max = 100
        n = 1
        angle = x
        while n < n_max :
            angle += (fac(2*n) / ((4**n) * ((fac(n))**2) * (2*n+1))) * (x**(2*n+1))
            n += 1
    elif x == 1:
        angle = π / 2
    elif x == -1:
        angle = -π / 2
    else:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE ANGLE")
        raise Exception("IMPOSSIBLE ANGLE")
    if degrees:
        angle = angle * radian
    return round(angle,15)


def arccos(x):
    degrees = not_radians.get()
    if degrees:
        angle = 90 - arcsin(x)
    else:
        angle = π / 2 - arcsin(x)
    return round(angle,15)


def arctan(x):
    angle = arcsin(x / ((1+(x**2))**0.5))
    return round(angle,15)
        

    





# LOGARITHMIC FUNCTIONS

#Calculated by using an iterative trial and error function that approximates the using the logic:
#if log10(x) = y 10^y = x and so keeps on approximating y until 10^y = x to 15 decimal places
def log_10(x):

    n = 0
    test = 10**n
    bounded = False
    going_upwards = False
    if test < x:
        going_upwards = True
    while not bounded:
    #Finds the first n for 10^n that is either bigger or smaller than x depending on whether x<1 or x>1
        if round(test,15) == x:
            log_x = n
            low_bound = n - 1
            bounded = True
        if going_upwards:
            n += 1
            test = 10**n
            if test > x:
                low_bound = n - 1
                bounded = True
        elif not going_upwards:
            n -= 1
            test = 10**n
            if test < x:
                low_bound = n
                bounded = True
                
    numerator = 1
    denominator = 2
    test = 10**(low_bound + (numerator / denominator))
    test1 = 1
    while not round(test,15) == x:
        #Starts by finding 10^(n+1/2) and changes the fraction of 1/2 to get the result closer to x
        #the value for the index that gets 10^index = x to 15 decimal place is taken as the answer
        if test < x:
            numerator = 2 * numerator + 1
            denominator = 2 * denominator
        elif test > x:
            numerator = 2 * numerator - 1
            denominator = 2 * denominator
        test = 10**(low_bound + (numerator / denominator))
        if test == test1:
            break
        test1 = test
    log_x = round(low_bound + (numerator / denominator),15)
    if int(log_x) == log_x:
        log_x = int(log_x)
    return log_x
   
def log(x,y):
    if x == 1 or x<= 0 or y <= 0:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE LOGARITHM INPUT")
        raise Exception("IMPOSSIBLE LOGARITHM INPUT")
    log_x = log_10(x)
    log_y = log_10(y)
    #Uses the identity logx(y) = log10(y) / log10(x)
    answer = round((log_y / log_x),15)
    if int(answer) == answer:
        answer = int(answer)
    return answer

def ln(x):
    answer = log(e,x)
    return answer








# INTEGRATION FUNCTIONS

#Finds the output of all the possible combinations of + and - when used in conjunction
def combine_operations(operation1,operation2):
    if operation1 == '-' and operation2 == '-':
        net_operation = "+"
    elif operation1 == '+' and operation2 == '+':
        net_operation = "+"
    else:
        net_operation = "-"
    return net_operation

    
def integrate(a,b,equation):
    
    #Splits the equation into all the terms and the operations that join them
    equation = list(equation)
    while " " in equation:
        equation.remove(" ")
    equation = "".join(equation)
    while '**-' in equation:
        equation = equation.replace('**-', '**a')
    while '/-' in equation:
        equation = equation.replace('/-', '/a')
    while '(-' in equation:
        equation = equation.replace('(-', '(a')
    net_operation = 0
    while equation[0] == '-' or equation[0] == '+':
        if equation [0] == '-':
            net_operation += 1
        equation = equation[1:]
    if not net_operation % 2 == 0:
        equation = "a" + equation
    operations = [x for x in equation if x == '+' or x == '-']
    while '-' in equation:
        equation = equation.replace('-', '+')
    while 'a' in equation:
        equation = equation.replace('a', '-')
    while equation[0] == '+':
        equation = equation[1:]
    equation = equation.split('+')
    while "" in equation:
        index = equation.index("")
        equation.remove("")
        operations[index] = combine_operations(operations[index-1],operations[index])
        operations.pop(index-1)

    #Puts all the terms into a common form regardless of how they are entered into the calculator
    integrated_terms = []
    power = False
    for i in equation:
        if '**' in i:
            power = True
        i = list(i)
        i = list(filter(None, i))
        while ' ' in i:
            i.remove(' ')
        while '*' in i:
            i.remove('*')
        if not power and 'x' in i:
            i.append('1')
            power = True
        if not power and not 'x' in i:
            i.append('x0')
            power = True
        if i[0] == 'x':
            i.insert(0, '1')
        i = ''.join(i)
        i = i.split('x')

        #Changes the coeffiecients and the indices to get the integrated term 
        i[1] = "(" + str(fractions.Fraction(eval(i[1]) + 1).limit_denominator()) + ")"
        i[0] = "(" + i[0] + "/" + i[1] + ")"
        power_n = i.pop()
        i.append("*")
        i.append("x")
        i.append("**")
        i.append(power_n)
        i = "".join(i)
        integrated_terms.append(i)
        power = False

    #Combines the terms and operations to get the integrated equation
    integrated = ""
    n = 0
    while n < (len(integrated_terms) - 1):
        integrated += integrated_terms[n] + operations[n]
        n += 1
    integrated += integrated_terms[n]

    #Calculates the output when x is equal to the upper and the lower bounds and finds their difference
    result = 0
    x = float(b)
    result += eval(integrated)
    x = float(a)
    result -= eval(integrated)
    if isinstance(result, complex):
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "CANNOT INTEGRATE COMPLEX")
        raise Exception("CANNOT INTEGRATE COMPLEX")
    if int(result) == result:
        result = int(result)
    return result








# DIFFERENTIATION FUNCTION

def differentiate(a,equation):

    #Similar process to integration
    equation = list(equation)
    while " " in equation:
        equation.remove(" ")
    equation = "".join(equation)
    while '**-' in equation:
        equation = equation.replace('**-', '**a')
    while '/-' in equation:
        equation = equation.replace('/-', '/a')
    while '(-' in equation:
        equation = equation.replace('(-', '(a')
    net_operation = 0
    while equation[0] == '-' or equation[0] == '+':
        if equation [0] == '-':
            net_operation += 1
        equation = equation[1:]
    if not net_operation % 2 == 0:
        equation = "a" + equation
    operations = [x for x in equation if x == '+' or x == '-']
    while '-' in equation:
        equation = equation.replace('-', '+')
    while 'a' in equation:
        equation = equation.replace('a', '-')
    equation = equation.split('+')
    while "" in equation:
        index = equation.index("")
        equation.remove("")
        operations[index] = combine_operations(operations[index-1],operations[index])
        operations.pop(index-1)

    differentiated_terms = []
    power = False
    for i in equation:
        if '**' in i:
            power = True
        i = list(i)
        i = list(filter(None, i))
        while ' ' in i:
            i.remove(' ')
        while '*' in i:
            i.remove('*')
        if not power and 'x' in i:
            i.append('1')
            power = True
        if not power and not 'x' in i:
            i.append('x0')
            power = True
        if i[0] == 'x':
            i.insert(0, '1')
        i = ''.join(i)
        i = i.split('x')

        index_1 = i[1]
        i[1] = "(" + str(fractions.Fraction(eval(i[1]) - 1).limit_denominator()) + ")"
        i[0] = "(" + i[0] + "*" + index_1 + ")"
        power_n = i.pop()
        i.append("*")
        i.append("x")
        i.append("**")
        i.append(power_n)
        i = "".join(i)
        differentiated_terms.append(i)
        power = False

    differentiated = ""
    n = 0
    while n < (len(differentiated_terms) - 1):
        differentiated += differentiated_terms[n] + operations[n]
        n += 1
    differentiated += differentiated_terms[n]

    result = 0
    x = float(a)
    result += eval(differentiated)
    if isinstance(result, complex):
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "CANNOT DIFFERENTIATE COMPLEX")
        raise Exception("CANNOT DIFFERENTIATE COMPLEX")
    if int(result) == result:
        result = int(result)
    return result








# SUMMATION FUNCTION

def summate(r,n,equation):

    #Puts all the terms into a common form regardless of how they are entered into the calculator
    equation = list(equation)
    while " " in equation:
        equation.remove(" ")
    syntaxed_equation = ""
    for index, i in enumerate(equation):
        try:
            if not index == 0 and i == 'x':
                a = int(equation[index-1])
                syntaxed_equation += "*"
        except:
            pass
        syntaxed_equation += i
        
    total = 0
    r = float(r)
    n = float(n)
    if r < 0 or n < 0 or not r == int(r) or not n == int(n):
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE ARGUMENT")
        raise Exception("IMPOSSIBLE ARGUMENT")
    if n < r:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE ARGUMENT")
        raise Exception("IMPOSSIBLE ARGUMENT")
    #Calculates the sum of the outputs of the equation when x is equal to all the numbers from r up to n
    while r < n+1:
        x = r
        total += eval(syntaxed_equation)
        r += 1
    return total








# MATRIX FUNCTIONS

identity_2 = [[1,0],
              [0,1]]
identity_3 = [[1,0,0],
              [0,1,0],
              [0,0,1]]

def check_if_square(A):
    if not (len(A) == 2 and len(A[0]) == 2) and not (len(A) == 3 and len(A[0]) == 3):
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
        raise Exception("IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")

#Adds the numbers at the corresponding index of two two dimentional lists or 'Matrices'
def matrix_add(A,B):
    C = []
    for index_y, i in enumerate(A):
        D = []
        for index_x, j in enumerate(i):
            element = j + B[index_y][index_x]
            element = un_dec(element)
            D.append(element)
        C.append(D)
    return C

#Minuses the numbers at the corresponding index of two two dimentional lists or 'Matrices'
def matrix_minus(A,B):
    C = []
    for index_y, i in enumerate(A):
        D = []
        for index_x, j in enumerate(i):
            element = j - B[index_y][index_x]
            element = un_dec(element)
            D.append(element)
        C.append(D)
    return C

#Multiplies all the numbers in the matrix by the same number
def matrix_multiply_scalar(b,A):
    B = []
    for i in A:
        C = []
        for j in i:
            element = round(j * b,5)
            element = un_dec(element)
            C.append(element)
        B.append(C)
    return B

#Multiplies the rows of the first matrix by the collumns of the next
#using the rule of matrix multiplication
def matrix_multiply(A,B):
    if not len(A[0]) == len(B):
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
        raise Exception("IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
    C = []
    for index_y, i in enumerate(A):
        D = []
        for k in range(len(B[0])):
            total = 0
            for index_x, j in enumerate(i):
                total += j * B[index_x][k]
                total = round(un_dec(total),5)
            D.append(total)
        C.append(D)
    return C

#Finds the matrix of minors for a 3*3 matrix
#by finding the minor matrix of each number in the matrix then finding it's determinant
def minors(A):
    check_if_square(A)
    B = []
    if len(A) == 3:
        for index_y, i in enumerate(A):
            C = []
            for index_x, i in enumerate(i):
                D = [x[:] for x in A]
                D.pop(index_y)
                for k in range(len(D)):
                    D[k].pop(index_x)
                C.append(D)
            B.append(C)
        E = []
        for i in B:
            F = []
            for j in i:
                minor = (j[0][0] * j[1][1]) - (j[0][1] * j[1][0])
                minor = un_dec(minor)
                F.append(minor)
            E.append(F)
    else:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
        raise Exception("IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
    return E

#Finds the determinant of a 2*2 or a 3*3 matrix
def det(A):
    check_if_square(A)
    if len(A) == 2 and len(A[0]) == 2:
        determinant = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])
    elif len(A) == 3 and len(A[0]) == 3:
        B = minors(A)
        determinant = ((A[0][0] * B[0][0]) - (A[0][1] * B[0][1]) +
                       (A[0][2] * B[0][2]))
    determinant = un_dec(determinant)
    return determinant

#Inverses the signs on the correct numbers in the matrix to make the matrix of cofactors
def cofactors(A):
    if len(A) == 3 and len(A[0]) == 3:
        B = []
        for index_y, i in enumerate(A):
            C = []
            for index_x, j in enumerate(i):
                if (index_y == 0 and index_x == 1 or index_y == 1 and index_x == 0 or
                index_y == 1 and index_x == 2 or index_y == 2 and index_x == 1):
                    j = j * -1
                C.append(j)
            B.append(C)
    else:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
        raise Exception("IMPOSSIBLE MATRIX DIMENSIONS FOR THIS FUNCTION")
    return B

#Transposes the matrix by switching it's rows and collumns
def transpose(A):
    check_if_square(A)
    B = []
    for index_y, i in enumerate(A):
        C = []
        for index_x, j in enumerate(i):
            j = A[index_x][index_y]
            C.append(j)
        B.append(C)
    return B

#Uses the rules of inversion and the previous functions to inverse 2*2 and 3*3 matrices
def matrix_inverse(A):
    determinant = det(A)
    b = 1 / determinant
    if len(A) == 2 and len(A[0]) == 2:
        A = [[A[1][1], A[0][1] * -1], [A[1][0] * -1, A[0][0]]]
        inverse = matrix_multiply_scalar(b,A)
    elif len(A) == 3 and len(A[0]) == 3:
        matrix_of_minors = minors(A)
        matrix_of_cofactors = cofactors(matrix_of_minors)
        transposed_matrix = transpose(matrix_of_cofactors)
        inverse = matrix_multiply_scalar(b,transposed_matrix)
    return inverse
        

#The function that calculates a matrix equation
def matrix_calc(equation,MatA,MatB,MatC,MatD,MatAns):

    defined_matrices = ['MatA', MatA,'MatB', MatB,'MatC', MatC, 'MatD', MatD,'MatAns', MatAns]
    #Splits the equation into all the numbers and matrices and the operations that join them
    while " " in equation:
        equation = equation.replace(" ","")
    while '**-' in equation:
        equation = equation.replace('**-', '**b')
    while '/-' in equation:
        equation = equation.replace('/-', '/b')
    while '*-' in equation:
        equation = equation.replace('*-', '*b')
    if equation[0] == '-':
        equation = equation[1:]
        equation = "b" + equation
    while '**' in equation:
        equation = equation.replace('**', 'd')
    operations = [x for x in equation if x == '+' or x == '-' or x == '*' or x == 'd']
    operations_a = []
    for  i in operations:
        if i == 'd':
            i = "**"
        operations_a.append(i)
    operations = operations_a
    while '-' in equation:
        equation = equation.replace('-', '+')
    while '*' in equation:
        equation = equation.replace('*', '+')
    while 'd' in equation:
        equation = equation.replace('d', '+')
    while 'b' in equation:
        equation = equation.replace('b', '-')
    equation = equation.split('+')
    #If the first matrix is negative it is multiplied by the scalar -1
    if '-' in equation[0]:
        try:
            eval(equation[0])
        except:
            first_matrix_letter = equation[0].replace("-", "")
            first_matrix_index = defined_matrices.index(first_matrix_letter) + 1
            matrix = matrix_multiply_scalar(-1,defined_matrices[first_matrix_index])
            equation[0] = matrix
    while "" in equation:
        index = equation.index("")
        if ((operations[index-1] == '-' or operations[index-1] == '+') and
            (operations[index] == '-' or operations[index] == '+')):
            equation.remove("")
            operations[index] = combine_operations(operations[index-1],operations[index])
            operations.pop(index-1)
        else:
            error.set(True)
            screen.delete('1.0','end')
            screen.insert(END, "IMPOSSIBLE EQUATION SYNTAX")
            raise Exception("IMPOSSIBLE EQUATION SYNTAX")
            
    final_equation = []
    #Puts the syntaxed numbers, matrices and operations back into a standard equation
    for index,i in enumerate(equation):
        final_equation.append(i)
        try:
            final_equation.append(operations[index])
        except:
            pass

    #Switches out the worded symbols for the matrices with the actual pre-defined matrices
    equation = []
    for index,i in enumerate(final_equation):
        if i in defined_matrices:
            matrix_index = defined_matrices.index(i) + 1
            i = defined_matrices[matrix_index]
        equation.append(i)
        
    #From here in the function the equation is calculated term by term
    #Starting with the actual worded functions
    #Then moving onto the operations in order of priority deduced by the rules of 'BIDMAS'
    equation_without_functions = []
    for i in equation:
        if 'Minors(' in i or 'Det(' in i or 'Transpose(' in i or 'Cofactors(' in i:
            i = i.split('(')
            i[1] = i[1][:-1]
            matrix_index = defined_matrices.index(i[1]) + 1
            i[1] = defined_matrices[matrix_index]
            if i[0] == 'Minors':
                i = minors(i[1])
            elif i[0] == 'Det':
                i = det(i[1])
            elif i[0] == 'Transpose':
                i = transpose(i[1])
            elif i[0] == 'Cofactors':
                i = cofactors(i[1])
        equation_without_functions.append(i)
    equation = equation_without_functions


    while '**' in equation:
        index = equation.index("**")
        a = equation[index-1]
        check_if_square(a)
        b = int(equation[index+1])
        matrix = a
        #Multiplying a matrix by 0 yields an identity matrix
        if b == 0:
            if len(a) == 2 and len(a[0]) == 2:
                matrix = identity_2
            elif len(a) == 3 and len(a[0]) == 3:
                matrix = identity_3
        #Checking whether to inverse the matrix or multiply it by itself
        if b < 0:
            matrix = matrix_inverse(a)
            b = b *-1
        b -= 1
        while b > 0:
            matrix = matrix_multiply(matrix,a)
            b -= 1
        equation[index-1] = matrix
        del equation[index]
        del equation[index]

    while '*' in equation:
        index = equation.index("*")
        a = equation[index-1]
        b = equation[index+1]
        #Finds out if it is two matrices, two numbers or one of both that is being multiplied
        if isinstance(a,list) and isinstance(b,list):
            multiplied = matrix_multiply(a,b)
        elif isinstance(a,list):
            multiplied = matrix_multiply_scalar(eval(b),a)
        elif isinstance(b,list):
            multiplied = matrix_multiply_scalar(eval(a),b)
        else:
            multiplied = eval(a) * eval(b)
        equation[index-1] = multiplied
        del equation[index]
        del equation[index]

    #In maths + and - happen in order and have no priority over each other
    #so they must be checked in conjunction
    while '+' in equation or '-' in equation:
        for i in equation:
            if i == '+' or i == '-':
                function = i
                break
        if function == '+':
            index = equation.index("+") 
            a = equation[index-1]
            b = equation[index+1]
            matrix = matrix_add(a,b)
            equation[index-1] = matrix
            del equation[index]
            del equation[index]
        else:
            index = equation.index("-") 
            a = equation[index-1]
            b = equation[index+1]
            matrix = matrix_minus(a,b)
            equation[index-1] = matrix
            del equation[index]
            del equation[index]

    return equation[0]








# VECTOR FUNCTIONS


#For many vecotor functions the two vectors must have the same dimensions
def check_compatible(A,B):
    if len(A) == len(B):
        pass
    else:
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "INCOMPATIBLE VECTOR DIMENSIONS")
        raise Exception("INCOMPATIBLE VECTOR DIMENSIONS")

#Adds vectors by adding the corresponding numbers in the list
def vector_add(A,B):
    check_compatible(A,B)
    C = []
    for index, i in enumerate(A):
        C.append(i + B[index])
    return C

#Minuses vectors by minusing the corresponding numbers in the list
def vector_minus(A,B):
    check_compatible(A,B)
    C = []
    for index, i in enumerate(A):
        C.append(i - B[index])
    return C

#Multiplies all elements in the vector by the same number
def vector_scalar_multiply(b,A):
    b = float(b)
    B = []
    for i in A:
        B.append(un_dec(round(i*b,5)))
    return B

#Uses pythagoras to find the absolute magnitude of the vector
def vector_abs(A):
    sum_of_squares = 0
    for i in A:
        sum_of_squares += i**2
    absolute = sum_of_squares**0.5
    absolute = un_dec(absolute)
    return absolute

#Finds the unit vector of a vector by dividing the vector by its absolute magnitude
def vector_unit(A):
    B = vector_scalar_multiply(1/vector_abs(A), A)
    return B

#Finds the dot product of two vectors by using the rule A.B = Ai*Bi + Aj*Bj + Ak*Bk
def vector_dot_product(A,B):
    check_compatible(A,B)
    scalar = 0
    for index, i in enumerate(A):
        scalar += un_dec(i * B[index])
    return scalar

#Uses the dot product identity to calculate cosx to find the angle between two vectors
def vector_angle(A,B):
    cosx = vector_dot_product(A,B) / (vector_abs(A) * vector_abs(B))
    x = arccos(cosx)
    return x

#Uses the cross product rule to multiply two 3 dimentional vectors
def vector_cross_product(A,B):
    if not (len(A) == 3 and len(B) == 3) :
        error.set(True)
        screen.delete('1.0','end')
        screen.insert(END, "INCOMPATIBLE VECTOR DIMENSIONS")
        raise Exception("INCOMPATIBLE VECTOR DIMENSIONS")
    C = []
    C.append(un_dec(A[1]*B[2] - A[2]*B[1]))
    C.append(un_dec(A[2]*B[0] - A[0]*B[2]))
    C.append(un_dec(A[0]*B[1] - A[1]*B[0]))
    return C


#The function that calculates a vector equation
def vector_calc(equation,A,B,C,D,Ans):

    #Similar process to the matrices calculation function
    defined_vectors = ['VecA',A,'VecB',B,'VecC',C,'VecD',D,'VecAns',Ans]
    while " " in equation:
        equation = equation.replace(" ","")
    while '**-' in equation:
        equation = equation.replace('**-', '**a')
    while '/-' in equation:
        equation = equation.replace('/-', '/a')
    while '*-' in equation:
        equation = equation.replace('*-', '*a')
    if equation[0] == '-':
        equation = equation[1:]
        equation = "a" + equation
    operations = [x for x in equation if x == '+' or x == '-' or x == '*' or x== '.']
    while '.' in equation:
        equation = equation.replace('.', '+')
    while '-' in equation:
        equation = equation.replace('-', '+')
    while '*' in equation:
        equation = equation.replace('*', '+')
    while 'a' in equation:
        equation = equation.replace('a', '-')
    equation = equation.split('+')
    if '-' in equation[0]:
        try:
            eval(equation[0])
        except:
            vector_letter = equation[0].replace("-", "")
            vector_index = defined_vectors.index(vector_letter) + 1
            vector = vector_scalar_multiply(-1,defined_vectors[vector_index])
            equation[0] = vector
    while "" in equation:
        index = equation.index("")
        if ((operations[index-1] == '-' or operations[index-1] == '+') and
            (operations[index] == '-' or operations[index] == '+')):
            equation.remove("")
            operations[index] = combine_operations(operations[index-1],operations[index])
            operations.pop(index-1)
        else:
            error.set(True)
            screen.delete('1.0','end')
            screen.insert(END, "IMPOSSIBLE EQUATION SYNTAX")
            raise Exception("IMPOSSIBLE EQUATION SYNTAX")
            
    final_equation = []
    for index,i in enumerate(equation):
        final_equation.append(i)
        try:
            final_equation.append(operations[index])
        except:
            pass

    equation = []
    for index,i in enumerate(final_equation):
        if i in defined_vectors:
            vector_index = defined_vectors.index(i) + 1
            i = defined_vectors[vector_index]
        equation.append(i)

    equation_without_functions = []
    for i in equation:
        if 'Unit(' in i or 'Abs(' in i or 'Angle(' in i:
            i = i.split('(')
            i[1] = i[1][:-1]
            if i[0] == 'Angle':
                i[1] = i[1].split(',')
                vector_index_1 = defined_vectors.index(i[1][0]) + 1
                i[1][0] = defined_vectors[vector_index_1]
                vector_index_2 = defined_vectors.index(i[1][1]) + 1
                i[1][1] = defined_vectors[vector_index_2]
                i = vector_angle(i[1][0],i[1][1])
            else:
                vector_index = defined_vectors.index(i[1]) + 1
                i[1] = defined_vectors[vector_index]
                if i[0] == 'Unit':
                    i = vector_unit(i[1])
                elif i[0] == 'Abs':
                    i = vector_abs(i[1])
        equation_without_functions.append(i)
    equation = equation_without_functions

    while '.' in equation:
        index = equation.index('.')
        a = equation[index-1]
        b = equation[index+1]
        try:
            a = int(a)
            b = int(b)
            scalar = str(a) + "." + str(b)
        except:
            scalar = vector_dot_product(a,b)
        equation[index-1] = scalar
        del equation[index]
        del equation[index]

    while '*' in equation:
        index = equation.index('*')
        a = equation[index-1]
        b = equation[index+1]
        if isinstance(a, str) or isinstance(b, str):
            if isinstance(a, list) or isinstance(b, list):
                try:
                    result = vector_scalar_multiply(a,b)
                except:
                    result = vector_scalar_multiply(b,a)
            else:
                result = str(eval(a + '*' + b))
        else:
            result = vector_cross_product(a,b)
        equation[index-1] = result
        del equation[index]
        del equation[index]

    while '+' in equation or '-' in equation:
        for i in equation:
            if i == '+' or i == '-':
                function = i
                break
        if function == '+':
            index = equation.index("+") 
            a = equation[index-1]
            b = equation[index+1]
            if isinstance(a, list) or isinstance(b, list):
                vector = vector_add(a,b)
                equation[index-1] = vector
            else:
                equation[index-1] = float(a) + float(b)
                equation[index-1] = un_dec(equation[index-1])
            del equation[index]
            del equation[index]
        else:
            index = equation.index("-") 
            a = equation[index-1]
            b = equation[index+1]
            if isinstance(a, list) or isinstance(b, list):
                vector = vector_minus(a,b)
                equation[index-1] = vector
            else:
                equation[index-1] = float(a) - float(b)
                equation[index-1] = un_dec(equation[index-1])
            del equation[index]
            del equation[index]

    return equation[0]








#   STATISTICAL FUNCTIONS


# ONE VARIABLE STATISTICS


#Adds the whole of x together
def sumx(x):
    sumx = 0
    for i in x:
        sumx += float(i)
    sumx = un_dec(sumx)
    return sumx

#Finds the mean of all the values of x
def meanx(x):
    meanx = sumx(x) / len(x)
    meanx = un_dec(meanx)
    return meanx

#Finds the sum of all the squares of x
def sumx_squared(x):
    sumx_squared = 0
    for i in x:
        sumx_squared += float(i)**2
    sumx_squared = un_dec(sumx_squared)
    return sumx_squared

#Finds the variance of x by using the identity variance = Σx^2/n - meanx^2
def variance(x):
    variance = (sumx_squared(x) / len(x)) - meanx(x)**2
    variance = un_dec(variance)
    return variance

#Finds standard deviation by square rooting variance
def standard_deviation(x):
    standard_deviation = variance(x)**0.5
    standard_deviation = un_dec(standard_deviation)
    return standard_deviation

#Uses the identity Sxx = Σ(x-meanx)^2 = Σx^2 - meanx^2*n
def Sxx(x):
    Sxx = sumx_squared(x) - ((sumx(x)**2) / len(x))
    Sxx = un_dec(Sxx)
    return Sxx




# TWO VARIABLE STATISTICS

#Calculates a list where the corresponding terms of x and y have been multiplied
def xy(x,y):
    xy = []
    for index,i in enumerate(x):
        xy.append(str(float(i) * float(y[index])))
    return xy

#Uses the identity Sxy = Σ(x-meanx)*Σ(y-meany) = Σxy - meanx*meany*n
def Sxy(x,y):
    xy_list = xy(x,y)
    Sxy = sumx(xy_list) - ((sumx(x) * sumx(y)) / len(x))
    Sxy = un_dec(Sxy)
    return Sxy

#Following functions to work out the equation of regression
#Uses gradient = Sxy/Sxx
def gradient_b(x,y):
    b = Sxy(x,y) / Sxx(x)
    b = un_dec(b)
    return b

#Uses intercept = meany - gradient* meanx
def intercept_a(x,y):
    a = meanx(y) - gradient_b(x,y) * meanx(x)
    a = un_dec(a)
    return a

#Calculates the value of regression by using the identity r = gradient * (σx/σy)
def regression(x,y):
    r = (gradient_b(x,y) * standard_deviation(x)) / standard_deviation(y)
    r = un_dec(r)
    return r




# FREQUENCY ONE VARIABLE STATISTICS

#Same idea as with the functions with x but now fx is calculated and used instead of x
def sumf(f):
    sumf = 0
    for i in f:
        sumf += float(i)
    sumf = un_dec(sumf)
    return sumf
    
def fx(f,x):
    fx = []
    for index, i in enumerate(x):
        fx.append(str(float(i) * float(f[index])))
    return fx

def sumfx(f,x):
    fx_list = fx(f,x)
    sumfx = 0
    for i in fx_list:
        sumfx += float(i)
    sumfx = un_dec(sumfx)
    return sumfx

def fx_squared(f,x):
    fx_squared = []
    for index, i in enumerate(x):
        fx_squared.append(str((float(i)**2) * float(f[index])))
    return fx_squared

def meanfx(f,x):
    meanfx = sumfx(f,x) / sumf(f)
    meanfx = un_dec(meanfx)
    return meanfx

def sumfx_squared(f,x):
    fx_squared_list = fx_squared(f,x)
    sumfx_squared = 0
    for i in fx_squared_list:
        sumfx_squared += float(i)
    sumfx_squared = un_dec(sumfx_squared)
    return sumfx_squared

def variance_fx(f,x):
    variance = (sumfx_squared(f,x) / sumf(f)) - meanfx(f,x)**2
    variance = un_dec(variance)
    return variance

def standard_deviation_fx(f,x):
    standard_deviation = variance_fx(f,x) ** 0.5
    standard_deviation = un_dec(standard_deviation)
    return standard_deviation




# QUARTILES

#calculates the three quartiles by using the formula for their positions in the data set
def Q(x):
    a = x.copy()
    even = False
    if len(a) % 2 == 0:
        even = True
    if even:
        Q2 = (float(a[int((len(a) / 2) - 1)]) + float(a[int(len(a) / 2)])) / 2
    else:
        Q2 = float(a[int((len(a) / 2) - 0.5)])
        a.pop(int((len(a) / 2) - 0.5))
    even = False
    if len(a) % 4 == 0:
        even = True
    if even:
        Q1 = (float(a[int((len(a) / 4) - 1)]) + float(a[int(len(a) / 4)])) / 2
        Q3 = (float(a[int((len(a) * (3 / 4)) - 1)]) + float(a[int((len(a) * (3 / 4)))])) / 2
    else:
        Q1 = float(a[int((len(a) / 4) - 0.5)])
        Q3 = float(a[int((len(a) * (3 / 4)) - 0.5)])
    minx = float(a[0])
    maxx = float(a[-1])
    Q1 = un_dec(Q1)
    Q2 = un_dec(Q2)
    Q3 = un_dec(Q3)
    minx = un_dec(minx)
    maxx = un_dec(maxx)
    return Q1, Q2, Q3, minx, maxx

def Q_frequency(f,x):
    fx = []
    for index,i in enumerate(x):
        for j in range(0,int(f[index])):
            fx.append(i)
    Q1, Q2, Q3, minx, maxx = Q(fx)
    return Q1, Q2, Q3, minx, maxx








# TABLE FUNCTION

def table(L,U,step,equation1,equation2):

    #Syntaxes the input equations so they are always in a standardized form
    second_equation = two_equations.get()
    L = float(L)
    U = float(U)
    step = float(step)
    
    equation1 = list(equation1)
    syntaxed_equation1 = ""
    for index, i in enumerate(equation1):
        try:
            if not index == 0 and i == 'x':
                a = int(equation1[index-1])
                syntaxed_equation1 += "*"
        except:
            pass
        syntaxed_equation1 += i
    if second_equation:
        equation2 = list(equation2)
        syntaxed_equation2 = ""
        for index, i in enumerate(equation2):
            try:
                if not index == 0 and i == 'x':
                    a = int(equation2[index-1])
                    syntaxed_equation2 += "*"
            except:
                pass
            syntaxed_equation2 += i     

    #The step input is used to work out which values of x need to be used
    #Then each value of x is inserted into the equation and used to get an output
    x = L
    results1 = []
    while x <= U:
        outcome = eval(syntaxed_equation1)
        if int(outcome) == outcome:
            outcome = int(outcome)
        results1.append(outcome)
        x += step
    x = L
    results2 = []
    if second_equation:
        while x <= U:
            outcome = eval(syntaxed_equation2)
            if int(outcome) == outcome:
                outcome = int(outcome)
            results2.append(outcome)
            x += step

    #Calculates how many values of x have been used
    numbers = []
    for index, i in enumerate(results1):
        n = L + (index*step)
        if int(n) == n:
            n = int(n)
        numbers.append(n)
        
    return numbers,results1,results2



    




# FUNCTION TO SOLVE QUADRATICS

def solve_quad(equation):

    #Syntaxes the input equations so they are always in a standardized form
    #So a value for a, b and c can be isolated
    single_root.set(True)
    input_equation = equation
    equation = list(equation)
    while " " in equation:
        equation.remove(" ")
    equation = "".join(equation)
    equation = equation.split('x')
    equation[1] = equation[1].replace("**2","")
    syntaxed_equation = []
    for i in equation:
        if i == "":
            i = "1"
        if i == "-":
            i = "-1"
        if i == "+":
            i = "1"
        if "=" in i:
            i = i.replace("=","")
        i = eval(i)
        syntaxed_equation.append(i)
    equation = syntaxed_equation
    a = float(equation[0])
    b = float(equation[1])
    c = float(equation[2])

    #Uses the equation (-b±√(b^2-4ac))/2a to get the two solutions
    real_part = fractions.Fraction(-b / (2 * a)).limit_denominator()
    if real_part.denominator == 1:
        real_part = str(real_part.numerator)
    else:
        real_part = str(real_part.numerator) + "/" + str(real_part.denominator)
    solution_1 = real_part + " + " + sqrt((b**2 - (4 * a * c)) / (4 * (a**2)))
    solution_2 = real_part + " - " + sqrt((b**2 - (4 * a * c)) / (4 * (a**2)))
    if not '√' in solution_1:
        solution_1 = answer_format(eval(solution_1))
    if not '√' in solution_2:
        solution_2 = answer_format(eval(solution_2))
    solution1.set(solution_1)
    solution2.set(solution_2)

    #The minimum/maximum is calculated for x using the equation x = -b/(2*a)
    #Then this value is inserted into the equation to acquire the minimum/maximum for y
    min_x = (-b) / (2 * a)
    x = min_x
    syntaxed_equation = ""
    for index, i in enumerate(input_equation):
        try:
            if not index == 0 and i == 'x':
                a = int(input_equation[index-1])
                syntaxed_equation += "*"
        except:
            pass
        syntaxed_equation += i
    min_y = eval(syntaxed_equation)

    min_x = answer_format(min_x)
    min_y = answer_format(min_y)

    return solution_1, solution_2, min_x, min_y








# FUNCTION TO SOLVE SIMULTANEOUS EQUATIONS

def solve_simultaneous(equation_list):

    coefficient_matrix = []
    constant_matrix = []
    for equation in equation_list:
        while " " in equation:
            equation = equation.replace(" ","")
        equation = equation.replace("y","x")
        if "z" in equation:
            equation = equation.replace("z","x")
        equation = equation.split("x")
        syntaxed_equation = []
        #Syntaxes the input equations so they are always in a standardized form
        #and creates a matrix of coefficients and a matrix of the equalled numbers
        #to be used in an inverse matrix equation to calculate x and y or x, y and z
        for i in equation:
            if i == "":
                i = "1"
            if i == "-":
                i = "-1"
            if i == "+":
                i = "1"
            if "=" in i:
                i = i.replace("=","")
            i = eval(i)
            syntaxed_equation.append(i)
        equation = syntaxed_equation
        constant = []
        constant.append(equation.pop(-1))
        coefficient_matrix.append(equation)
        constant_matrix.append(constant)
    inversed_coefficient_matrix = matrix_inverse(coefficient_matrix)
    variable_matrix = matrix_multiply(inversed_coefficient_matrix,constant_matrix)
    return variable_matrix










# KEY UI FUNCTIONS

#Puts characters onto the screen; used for numbers, operations and functions
def insert_characters(x):
    answer_showing = check_answered()
    if answer_showing:
        screen.delete('end-2l', 'end')
    screen.insert(INSERT, x)

#Checks if the answer is being displayed
def check_answered():
    read_screen = screen.get('1.0', 'end')
    answer_showing = False
    if 'Answer' in read_screen:
        answer_showing = True
    return answer_showing



#switches out the symbols for functions with the actual words so that a calculation can be done
def syntax_functions(read_screen):
    while '√' in read_screen:
        read_screen = read_screen.replace("√", "sqrt")
    while '%' in read_screen:
        read_screen = read_screen.replace("%", "per")
    menu_before_called = current_menu.get()
    if not menu_before_called == 'matrices_menu()' and not menu_before_called == 'vectors_menu()':  
        while 'Ans' in read_screen:
            read_screen = read_screen.replace("Ans", ans.get())
    return read_screen

#Syntaxes the input equations so they are always in a standardized form
#to link up the conventional inputs of a calculator to the syntax that python needs
def syntax_input(read_screen):
    while ' ' in read_screen:
        read_screen = read_screen.replace(" ", "")
    while '\n' in read_screen:
        read_screen = read_screen.replace("\n", "")
    syntaxed_screen = []
    for index, i in enumerate(list(read_screen)):
        try:
            if not index == 0 and i == '√':
                a = int(read_screen[index-1])
                syntaxed_screen += "*"
        except:
            pass
        try:
            if i == 'j':
                a = int(read_screen[index-1])
        except:
            syntaxed_screen += "1"
        syntaxed_screen += i
    read_screen = "".join(syntaxed_screen)
    read_screen = syntax_functions(read_screen)
    return read_screen

#Puts the answer into the correct form (i.e. correct rounding or standard form)
#depending on whether the number is complex in a list (matrix or vector) or real
#and it always outputs the answer as a string
def answer_into_form(answer, instance):
    rounding_place = 14
    if instance == 'list':
        rounding_place = 5
    if instance == "complex":
        rounding_place = 10
    if answer > 0.0001 or answer < -0.0001 or answer == 0:
        answer = round(answer,rounding_place)
        answer = str(un_dec(answer))
    else:
        answer = str(answer).split('e')
        answer[0] = round(float(answer[0]),rounding_place)
        if '0' in answer[1]:
            answer[1] = answer[1].replace("0", "")
        answer = str(answer[0]) + "*10**" + answer[1]
    if len(answer) > (rounding_place * 2):
        answer = float(answer)
        answer = int(answer)
        answer = list(str(answer))
        length = len(answer)
        answer.insert(1, ".")
        answer = float("".join(answer))
        answer = str(round(answer,rounding_place)) + "*10**" + str(length - 1)
    return answer

#Seperates a number into its parts before putting it into the above function
def answer_format(answer):
    if isinstance(answer, complex):
        real_part = answer.real
        imaginary_part = answer.imag
        if imaginary_part == 0:
            answer = answer_into_form(real_part, "real")
        elif real_part == 0:
            answer = answer_into_form(imaginary_part, "real") + "*j"
        else:
            answer = ("(" + answer_into_form(real_part, "complex") + "+" +
                      answer_into_form(imaginary_part, "complex") + "*j)")
    elif not isinstance(answer, list):
        answer = answer_into_form(answer, "real")
    elif isinstance(answer, list):
        rows = []
        for i in answer:
            columns = []
            if isinstance(i, list):
                for j in i:
                    j = eval(answer_into_form(j, 'list'))
                    columns.append(j)
                rows.append(columns)
            else:
                i = eval(answer_into_form(i, 'list'))
                rows.append(i)
        answer = rows
    return answer

#Outputs the calculated answer to the screen in a desirable form
def output_answer(answer):
    menu_before_called = current_menu.get()
    answer = answer_format(answer)
    if isinstance(answer, list) and menu_before_called == 'matrices_menu()':
        screen.insert(END, "\n\nAnswer = ")
        for i in answer:
            screen.insert(END, "\n" + str(i))
    else:
        screen.insert(END, "\n\nAnswer = " + str(answer))
    ans.set(str(answer))
    #Eater egg
    if answer == '666':
        master.configure(bg = 'dark red')


def equals_command():
    try:
        #Recalls the previously stored constants
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        single_root.set(False)
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
            pass
        
        else:
            #Syntaxing the input to fix user errors such as extra spaces or no * before a variable.
            read_screen = syntax_input(read_screen)
            
            # Checks how many √ there are
            #if there is only one by itself the answer will output a surd
            if 'sqrt' in read_screen:
                check_single_term = read_screen
                while 'sqrt(' in check_single_term:
                    check_single_term = check_single_term.replace('sqrt(', ')')
                check_single_term = list(filter(None, check_single_term.split(')')))
                if len(check_single_term) == 1:
                    single_root.set(True)

            #calculates the on screen equation in normal circumstances
            a_single_root = single_root.get()
            if not '=' in read_screen and not a_single_root:
                answer = eval(read_screen)
                output_answer(answer)
            
            # This runs when a variable is being defined instead of an answer being calculated
            elif '=' in read_screen:
                read_screen = read_screen.replace('=', '_global.set("')
                read_screen = read_screen + '")'
                exec(read_screen)
                screen.delete('1.0','end')
        
            # This runs when the √ is the only part of the input expression 
            else:
                answer = str(eval(read_screen))
                screen.insert(END, "\n\nAnswer = " + answer)
                ans.set(answer)

    #Function is inside a try / except catch
    #so that if something goes wrong an error is outputted to the screen             
    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")



#Deletes the last character before the cursor using the cursors current index
def delete_character():
    answer_showing = check_answered()
    error_encountered = error.get()
    if error_encountered:
        screen.delete('1.0','end')
        error.set(False)
    if not answer_showing:
        index = screen.index(INSERT)
        decimal_place = index.split('.')
        screen.delete(decimal_place[0] + "." + str(int(int(decimal_place[1]) - 1)), index)
    else:
        answer_position = screen.search('Answer', '1.0', 'end')
        screen.delete(str(float(answer_position) - 1), 'end')

#Deletes all characters from the screen
def clear():
    screen.delete('1.0','end')
    error.set(False)

#Clears the screen and resets all variables   
def clear_command():
    clear()
    menu_before_called = current_menu.get()
    eval(menu_before_called)
    in_info.set(False)
    shift.set(False)
    in_options.set(False)
    in_menus.set(False)
    
#Converts the form of the answer
def StoD(x):
    answer_showing = check_answered()
    menu_before_called = current_menu.get()
    decimalise = single_root.get()
    read_screen = screen.get('1.0', 'end')
    
    if answer_showing and not '[' in x and not (
        menu_before_called == 'one_variable_statistics_menu()' or
        menu_before_called == 'one_variable_statistics_with_frequency_menu()' or
        menu_before_called == 'two_variable_statistics_menu()' or
        menu_before_called == 'solve_quadratics_menu()' or
        menu_before_called == 'solve_simultaneous_menu_1()' or
        menu_before_called == "solve_simultaneous_menu_2('2')" or
        menu_before_called == "solve_simultaneous_menu_2('3')"):
        screen.delete('end-2l', 'end')
        #Turns the answer into a real number if it is a surd
        if decimalise:
            screen.insert(END, "+0")
            equals_command()
            converted = ans.get()
            screen.delete('end-2l', 'end')
            screen.delete('end-3c', 'end')
        #Under normal circumstances converts between real and fraction
        elif not "/" in x:
            fraction = fractions.Fraction(x).limit_denominator()
            if fraction.denominator == 1:
                converted = str(fraction.numerator)
            else:
                converted = str(fraction.numerator) + "/" + str(fraction.denominator) 
        else:
            converted = answer_format(eval(x))
        ans.set(converted)
        screen.insert(END, "\n\nAnswer = " + converted)
    #In solve quadratics menu StoD used to convert between surd and real    
    if answer_showing and menu_before_called == 'solve_quadratics_menu()':
        if '√' in read_screen:
            single_root.set(False)
            #Seperates all the parts of the output then reconstructs it after changing the soloutions
            cut = read_screen.split(';')
            solution_1 = cut[1].split('=')
            solution_2 = cut[2].split('=')
            solution_1 = solution_1[1]
            solution_2 = solution_2[1]
            #Syntaxes the input equations so they are always in a standardized form
            while '\n' in solution_1:
                solution_1 = solution_1.replace("\n", "")
            while '\n' in solution_2:
                solution_2 = solution_2.replace("\n", "")
            if 'j' in solution_1:
                solution_1 = solution_1.replace("j", "1j")
            if 'j' in solution_2:
                solution_2 = solution_2.replace("j", "1j")
            solution1.set(solution_1)
            solution2.set(solution_2)
            while '√' in solution_1:
                solution_1 = solution_1.replace("√", "sqrt")
            while '√' in solution_2:
                solution_2 = solution_2.replace("√", "sqrt")
            cut[1] = "Solution 1 = " + answer_format(eval(solution_1)) + "\n"
            cut[2] = "Solution 2 = " + answer_format(eval(solution_2)) + "\n"
            converted = cut[0] + ";" + cut[1] + ";" + cut[2] + ";" + cut[3] + ";" + cut[4]
            clear()
            screen.insert(END, converted)
        else:
            cut = read_screen.split(';')
            cut[1] = "Solution 1 = " + solution1.get() + "\n"
            cut[2] = "Solution 2 = " + solution2.get() + "\n"
            converted = cut[0] + ";" + cut[1] + ";" + cut[2] + ";" + cut[3] + ";" + cut[4]
            clear()
            screen.insert(END, converted)
            
    if answer_showing and (menu_before_called == "solve_simultaneous_menu_2('2')" or
                           menu_before_called == "solve_simultaneous_menu_2('3')"):
        third_variable = False
        if menu_before_called == "solve_simultaneous_menu_2('3')":
            third_variable = True
        #Isolates the x, y and z answers and converts them individually
        #so it can change them and then reconstruct the output
        read_screen = read_screen.split(':')
        outputs = read_screen[1].split(';')
        x = outputs[1].split('=')
        x = x[1]
        y = outputs[2].split('=')
        y = y[1]
        if third_variable:
            z = outputs[3].split('=')
            z = z[1]
        if not "/" in x:
            fraction = fractions.Fraction(x).limit_denominator()
            if fraction.denominator == 1:
                x = str(fraction.numerator)
            else:
                x = str(fraction.numerator) + "/" + str(fraction.denominator) 
        else:
            x = answer_format(eval(x))
        if not "/" in y:
            fraction = fractions.Fraction(y).limit_denominator()
            if fraction.denominator == 1:
                y = str(fraction.numerator)
            else:
                y = str(fraction.numerator) + "/" + str(fraction.denominator) 
        else:
            y = answer_format(eval(y))
        if third_variable:
            if not "/" in z:
                fraction = fractions.Fraction(z).limit_denominator()
                if fraction.denominator == 1:
                    z = str(fraction.numerator)
                else:
                    z = str(fraction.numerator) + "/" + str(fraction.denominator) 
            else:
                z = answer_format(eval(z))
        clear()
        screen.insert(END, read_screen[0] + " : " +
                      "\n\n;x = " + x +
                      "\n;y = " + y)
        if third_variable:
            screen.insert(END, "\n;z = " + z)
          
#Moves the cursor by changing it's index
def move_cursor(x):
    index = screen.index(INSERT)
    decimal_place = index.split('.')
    if x == '←':
        screen.mark_set(INSERT, decimal_place[0] + "." + str(int(int(decimal_place[1]) - 1)))
    elif x == '→':
        screen.mark_set(INSERT, decimal_place[0] + "." + str(int(int(decimal_place[1]) + 1)))
    elif x == '↑':
        screen.mark_set(INSERT, str(round((float(index)-1),1)))
    elif x == '↓':
        screen.mark_set(INSERT, str(round((float(index)+1),1)))

#Takes the user to the webpage that contains instructions for the calculator
def help_guide():
    webbrowser.open('https://zebsummerfield1.wixsite.com/thegcecalculator', new=2)

#Shows stored information on the calculator
def recall():
    recalled = in_info.get()
    screen_before_called = last_screen.get()
    menu_before_called = current_menu.get()
    if not recalled:
        last_screen.set(screen.get('1.0', 'end'))
        #If in the matrices menu the stored matrices are displayed
        if menu_before_called == 'matrices_menu()':
            screen.delete('1.0','end')
            screen.insert(END, "MatA = ")
            for i in eval(MatA_global.get()):
                screen.insert(END, "\n" + str(i))
            if len(eval(MatA_global.get())) == 0:
                screen.insert(END, "[]")
            screen.insert(END, "\nMatB = ")
            for i in eval(MatB_global.get()):
                screen.insert(END, "\n" + str(i))
            if len(eval(MatA_global.get())) == 0:
                screen.insert(END, "[]")
            screen.insert(END, "\nMatC = ")
            for i in eval(MatC_global.get()):
                screen.insert(END, "\n" + str(i))
            if len(eval(MatA_global.get())) == 0:
                screen.insert(END, "[]")
            screen.insert(END, "\nMatD = ")
            for i in eval(MatD_global.get()):
                screen.insert(END, "\n" + str(i))
            if len(eval(MatA_global.get())) == 0:
                screen.insert(END, "[]")
        #If in the vectors menu the stored vectors are displayed
        elif menu_before_called == 'vectors_menu()':
            screen.delete('1.0','end')
            screen.insert(END, "VecA = " + VecA_global.get())
            screen.insert(END, "\nVecB = " + VecB_global.get())
            screen.insert(END, "\nVecC = " + VecC_global.get())
            screen.insert(END, "\nVecD = " + VecD_global.get())
        #Otherwise the normal stored numbers are displayed
        else:
            screen.delete('1.0','end')
            screen.insert(END, "A = " + A_global.get())
            screen.insert(END, "\nB = " + B_global.get())
            screen.insert(END, "\nC = " + C_global.get())
            screen.insert(END, "\nD = " + D_global.get())
        screen.insert(END, "\nAns = " + ans.get())
        in_info.set(True)
    else:
        clear()
        if screen_before_called.strip():
            screen.insert(END, screen_before_called)
        in_info.set(False)





# OPTIONS FUNCTIONS

#Swithces the angle unit between radians and degrees
def change_angle_unit():
    screen_before_called = last_screen.get()
    clear()
    degrees = not_radians.get()
    menu = current_menu.get()
    if degrees:
        not_radians.set(False)
    else:
        not_radians.set(True)
    eval(menu)
    clear()
    screen.insert(END, screen_before_called)

#Alows the user to change the background colour
def change_colour(x):
    screen_before_called = last_screen.get()
    clear()
    menu = current_menu.get()
    master.configure(bg = x)
    eval(menu)
    clear()
    screen.insert(END, screen_before_called)

#In the solve simultaneous equations menu
#this function allows the user to switch between 2 and 3 equations and unknowns
def change_number_of_equations():
    second_equation = two_equations.get()
    menu = current_menu.get()
    if not second_equation:
        eval(menu)
        screen.insert(END, "\n;Equation g(x) = ")
        two_equations.set(True)
    else:
        eval(menu)



    




#ALTERNATIVE UI FUNTICIONS OF EQUALS FOR DIFFERENT MENUS

def integration_equals():
    try:
        #Recalls the previously stored constants
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            #Syntaxing the input to fix user errors such as extra spaces or no * before a variable.
            read_screen = syntax_input(read_screen)
            #Isolates the different user inputs to be used in the integrate function
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            upper_bound = read_screen[1]
            lower_bound = read_screen[3]
            equation = read_screen[5]
            answer = integrate(lower_bound, upper_bound, equation)
            output_answer(answer)

    #Function is inside a try / except catch
    #so that if something goes wrong an error is outputted to the screen      
    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def differentiation_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            x = read_screen[1]
            equation = read_screen[3]
            answer = differentiate(x, equation)
            output_answer(answer)
            
    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def summation_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            r = read_screen[1]
            n = read_screen[3]
            equation = read_screen[5]
            answer = summate(r, n, equation)
            output_answer(answer)
            
    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def matrix_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        MatA = eval(MatA_global.get())
        MatB = eval(MatB_global.get())
        MatC = eval(MatC_global.get())
        MatD = eval(MatD_global.get())
        answer = eval(ans.get())
        if isinstance(answer, list):
            MatAns = answer
        else:
            MatAns = []
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            matrices = ['MatA','MatB','MatC','MatD','MatAns']
            read_screen = syntax_input(read_screen) 
            if '=' in read_screen:
                #This runs when a matrix is being defined instead of an answer being calculated
                read_screen = read_screen.replace("=", '_global.set("')
                read_screen = read_screen + '")'
                exec(read_screen)
                screen.delete('1.0','end')
            else:
                #Runs a different funciton depending on whether there are matrices in the equation
                if not any(i in read_screen for i in matrices):
                    answer = eval(read_screen)
                    output_answer(answer)
                else:
                    answer = matrix_calc(read_screen,MatA,MatB,MatC,MatD,MatAns)
                    output_answer(answer)

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def vector_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        VecA = eval(VecA_global.get())
        VecB = eval(VecB_global.get())
        VecC = eval(VecC_global.get())
        VecD = eval(VecD_global.get())
        answer = eval(ans.get())
        if isinstance(answer, list):
            VecAns = answer
        else:
            VecAns = []
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            vectors = ['VecA','VecB','VecC','VecD','VecAns']
            read_screen = syntax_input(read_screen)
            if '=' in read_screen:
                read_screen = read_screen.replace("=", '_global.set("')
                read_screen = read_screen + '")'
                exec(read_screen)
                screen.delete('1.0','end')
            else:
                if not any(i in read_screen for i in vectors):
                    answer = eval(read_screen)
                    output_answer(answer)
                else:
                    answer = vector_calc(read_screen,VecA,VecB,VecC,VecD,VecAns)
                    output_answer(answer)

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def one_variable_statistics_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            read_screen = read_screen.split('=')
            x = read_screen[1].split(',')
            #Finds out each statistical output individually by using the user inputted list
            sum_x = answer_format(sumx(x))
            mean_x = answer_format(meanx(x))
            sum_x_squared = answer_format(sumx_squared(x))
            variance_x = answer_format(variance(x))
            standard_deviation_x = answer_format(standard_deviation(x))
            Q1, Q2, Q3, minx, maxx = Q(x)
            Q1 = answer_format(Q1)
            Q2 = answer_format(Q2)
            Q3 = answer_format(Q3)
            minx = answer_format(minx)
            maxx = answer_format(maxx)
            S_xx = answer_format(Sxx(x))
            #Outputs the data to the screen
            statistic_output_list = [sum_x,mean_x,sum_x_squared,variance_x,standard_deviation_x,
                                     Q1,Q2,Q3,minx,maxx,S_xx]
            statistic_variable_list = ["Σx","meanx","Σx**2","σ**2x","σx",
                                       "Q1","Q2","Q3","min","max","Sxx"]
            screen.insert(END, "\n\nAnswers :")
            for index,i in enumerate(statistic_variable_list):
                screen.insert(END, "\n" + i + " = " + statistic_output_list[index])

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")
            

def one_variable_statistics_with_frequency_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            read_screen = read_screen.split(';')
            x = read_screen[0].split('=')
            x = x[1].split(',')
            f = read_screen[1].split('=')
            f = f[1].split(',')
            if not len(x) == len(f):
                error.set(True)
                screen.delete('1.0','end')
                screen.insert(END, "INCOMPATIBLE VARIABLE AND FREQUENCY LIST LENGTHS")
                raise Exception("INCOMPATIBLE VARIABLE AND FREQUENCY LIST LENGTHS")
            n = answer_format(sumf(f))
            sum_fx = answer_format(sumfx(f,x))
            mean_fx = answer_format(meanfx(f,x))
            sum_fx_squared = answer_format(sumfx_squared(f,x))
            variance = answer_format(variance_fx(f,x))
            standard_deviation = answer_format(standard_deviation_fx(f,x))
            Q1, Q2, Q3, minx, maxx = Q_frequency(f,x)
            Q1 = answer_format(Q1)
            Q2 = answer_format(Q2)
            Q3 = answer_format(Q3)
            minx = answer_format(minx)
            maxx = answer_format(maxx)
            statistic_output_list = [sum_fx,mean_fx,sum_fx_squared,variance,standard_deviation,
                                     Q1,Q2,Q3,minx,maxx]
            statistic_variable_list = ["Σfx","meanfx","Σfx**2","σ**2fx","σfx",
                                       "Q1","Q2","Q3","min","max"]
            screen.insert(END, "\n\nAnswers :")
            for index,i in enumerate(statistic_variable_list):
                screen.insert(END, "\n" + i + " = " + statistic_output_list[index])

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")

            
def two_variable_statistics_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            read_screen = read_screen.split(';')
            x = read_screen[0].split('=')
            x = x[1].split(',')
            y = read_screen[1].split('=')
            y = y[1].split(',')
            if not len(x) == len(y):
                error.set(True)
                screen.delete('1.0','end')
                screen.insert(END, "INCOMPATIBLE VARIABLE LIST LENGTHS")
                raise Exception("INCOMPATIBLE VARIABLE LIST LENGTHS")
            sum_x = answer_format(sumx(x))
            mean_x = answer_format(meanx(x))
            sum_x_squared = answer_format(sumx_squared(x))
            variance_x = answer_format(variance(x))
            standard_deviation_x = answer_format(standard_deviation(x))
            S_xx = answer_format(Sxx(x))
            S_xy = answer_format(Sxy(x,y))
            S_yy = answer_format(Sxx(y))
            sum_y = answer_format(sumx(y))
            mean_y = answer_format(meanx(y))
            sum_y_squared = answer_format(sumx_squared(y))
            variance_y = answer_format(variance(y))
            standard_deviation_y = answer_format(standard_deviation(y))
            sum_xy = answer_format(sumx(xy(x,y)))
            b = answer_format(gradient_b(x,y))
            a = answer_format(intercept_a(x,y))
            r = answer_format(regression(x,y))
            statistic_output_list = [sum_x,mean_x,sum_x_squared,variance_x,standard_deviation_x,
                                     S_xx,S_xy,S_yy,sum_y,mean_y,sum_y_squared,variance_y,
                                     standard_deviation_y,sum_xy,a,b,r]
            statistic_variable_list = ["Σx","meanx","Σx**2","σ**2x","σx","Sxx","Sxy","Syy",
                                       "Σy","meany","Σy**2","σ**2y","σy","Σxy",
                                       "y = bx + a:\na","b","r"]
            screen.insert(END, "\n\nAnswers :")
            for index,i in enumerate(statistic_variable_list):
                screen.insert(END, "\n" + i + " = " + statistic_output_list[index])

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")

            
def binomial_PD_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            X = read_screen[1]
            N = read_screen[3]
            p = read_screen[5]
            if ',' in X:
                X = X.split(',')
                answer = []
                for i in X:
                    answer.append(answer_format(binomialP(i,N,p)))
                screen.insert(END, "\n\nAnswers : ")
                for index,i in enumerate(X):
                    screen.insert(END, "\nX = " + i + " : p = " + answer[index])
            else:
                answer = binomialP(X,N,p)
                output_answer(answer)

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def binomial_CD_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            X = read_screen[1]
            N = read_screen[3]
            p = read_screen[5]
            if ',' in X:
                X = X.split(',')
                answer = []
                for i in X:
                    answer.append(answer_format(binomialC(i,N,p)))
                screen.insert(END, "\n\nAnswers : ")
                for index,i in enumerate(X):
                    screen.insert(END, "\nX ≤ " + i + " : p = " + answer[index])
            else:
                answer = binomialC(X,N,p)
                output_answer(answer)

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")


def table_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        second_equation = two_equations.get()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            while '=' in read_screen:
                read_screen = read_screen.replace("=", ";")
            read_screen = read_screen.split(';')
            L = read_screen[1]
            U = read_screen[3]
            step = read_screen[5]
            fx = read_screen[7]
            if second_equation:
                gx = read_screen[9]
            else:
                gx = ""
            numbers,results1,results2 = table(L,U,step,fx,gx)
            fx_output = []
            for index,i in enumerate(results1):
                fx_output.append(answer_format(i))
            if second_equation:
                gx_output = []
                for index,i in enumerate(results2):
                    gx_output.append(answer_format(i))
            screen.insert(END, "\n\nAnswers :")
            if not second_equation:
                screen.insert(END, "\n\nx = n : f(x)")
                for index, i in enumerate(numbers):
                    screen.insert(END, "\nx = " + str(i) + " : " + fx_output[index])
            else:
                screen.insert(END, "\n\nx = n : f(x) : g(x)")
                for index, i in enumerate(numbers):
                    screen.insert(END, "\nx = " + str(i) + " : " + fx_output[index] +
                                    " : " + gx_output[index])

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")

                
def solve_quadratic_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        second_equation = two_equations.get()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            read_screen = read_screen.split('=')
            read_screen = read_screen[0]
            solution1,solution2,minx,miny = solve_quad(read_screen)
            screen.insert(END, "\n\nAnswers : " +
                          "\n\n;Solution 1 = " + solution1 +
                          "\n;Solution 2 = " + solution2 +
                          "\n;min x = " + minx +
                          "\n;min y = " + miny)
    
    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")

        
def solve_simultaneous_equals():
    #Similar process to earlier alternative UI equals functions
    try:
        A = float(A_global.get())
        B = float(B_global.get())
        C = float(C_global.get())
        D = float(D_global.get())
        answer_showing = check_answered()
        second_equation = two_equations.get()
        read_screen = screen.get('1.0', 'end')
        if answer_showing:
                pass
        else:
            read_screen = syntax_input(read_screen)
            read_screen = read_screen.split(';')
            variables = solve_simultaneous(read_screen)
            x = answer_format(variables[0][0])
            y = answer_format(variables[1][0])
            if len(variables) == 3:
                z = answer_format(variables[2][0])
            screen.insert(END, "\n\nAnswers : " +
                          "\n\n;x = " + x +
                          "\n;y = " + y)
            if len(variables) == 3:
                screen.insert(END, "\n;z = " + z)

    except:
        error_encountered = error.get()
        if error_encountered:
            pass
        else:
            screen.delete('1.0','end')
            screen.insert(END, "ERROR")



        
        
        




# CREATING THE UI


#Creates and configures the Core part of the UI
master = Tk()
master.title("Calculator")
master.geometry('500x600')
master.configure(bg = 'dark grey')
master.option_add('*Button.Font', 'verdana 14')
master.option_add('*Button.bg', 'light grey')
master.option_add('*Button.relief', 'raised')
master.option_add('*Button.fg', 'black')
master.option_add('Button.height', '50')


#Creates the global variables
last_screen = StringVar()
last_screen.set('')
ans = StringVar()
ans.set('0')
error = BooleanVar()
error.set(False)
current_menu = StringVar()
current_menu.set("calculations_menu()")
shift = BooleanVar()
shift.set(False)
in_options = BooleanVar()
in_options.set(False)
in_menus = BooleanVar()
in_menus.set(False)
in_info = BooleanVar()
in_info.set(False)
A_global = StringVar()
B_global = StringVar()
C_global = StringVar()
D_global = StringVar()
A_global.set('0')
B_global.set('0')
C_global.set('0')
D_global.set('0')
MatA_global = StringVar()
MatB_global = StringVar()
MatC_global = StringVar()
MatD_global = StringVar()
MatA_global.set('[]')
MatB_global.set('[]')
MatC_global.set('[]')
MatD_global.set('[]')
VecA_global = StringVar()
VecB_global = StringVar()
VecC_global = StringVar()
VecD_global = StringVar()
VecA_global.set('[]')
VecB_global.set('[]')
VecC_global.set('[]')
VecD_global.set('[]')
not_radians = BooleanVar()
not_radians.set(True)
single_root = BooleanVar()
single_root.set(False)
two_equations = BooleanVar()
two_equations.set(False)
solution1 = StringVar()
solution2 = StringVar()
solution1.set('0')
solution2.set('0')


#Creates and configures the screen
screen_text = StringVar()
screen_text.set("")
scroll = Scrollbar(master)
scroll.place(x = 460, y = 10, width = 20, height = 80)
screen = Text(master, fg = 'black',relief = 'sunken',bg = 'light grey',
              font = ('verdana', 12), yscrollcommand = scroll.set)
screen.place(x = 20, y = 10, width = 440, height = 80)
scroll.configure(command = screen.yview,  bg = 'light grey')


#Creates, configures and places the buttons
button_7 = Button(master)
button_7.place(x = 60, y = 340, width = 60, height = 40)
button_8 = Button(master)
button_8.place(x = 140, y = 340, width = 60, height = 40)
button_9 = Button(master)
button_9.place(x = 220, y = 340, width = 60, height = 40)
button_del = Button(master)
button_del.place(x = 300, y = 340, width = 60, height = 40)
button_clear = Button(master)
button_clear.place(x = 380, y = 340, width = 60, height = 40)
button_4 = Button(master)
button_4.place(x = 60, y = 400, width = 60, height = 40)
button_5 = Button(master)
button_5.place(x = 140, y = 400, width = 60, height = 40)
button_6 = Button(master)
button_6.place(x = 220, y = 400, width = 60, height = 40)
button_minus = Button(master)
button_minus.place(x = 300, y = 400, width = 60, height = 40)
button_times = Button(master)
button_times.place(x = 380, y = 400, width = 60, height = 40)
button_1 = Button(master)
button_1.place(x = 60, y = 460, width = 60, height = 40)
button_2 = Button(master)
button_2.place(x = 140, y = 460, width = 60, height = 40)
button_3 = Button(master)
button_3.place(x = 220, y = 460, width = 60, height = 40)
button_plus = Button(master)
button_plus.place(x = 300, y = 460, width = 60, height = 40)
button_divide = Button(master)
button_divide.place(x = 380, y = 460, width = 60, height = 40)
button_decimal = Button(master)
button_decimal.place(x = 60, y = 520, width = 60, height = 40)
button_0 = Button(master)
button_0.place(x = 140, y = 520, width = 60, height = 40)
button_ans = Button(master)
button_ans.place(x = 220, y = 520, width = 60, height = 40)
button_StoD = Button(master)
button_StoD.place(x = 300, y = 520, width = 60, height = 40)
button_equals = Button(master)
button_equals.place(x = 380, y = 520, width = 60, height = 40)

button_A = Button(master)
button_A.place(x = 35, y = 280, width = 55, height = 30)
button_B = Button(master)
button_B.place(x = 110, y = 280, width = 55, height = 30)
button_open_bracket = Button(master)
button_open_bracket.place(x = 185, y = 280, width = 55, height = 30)
button_close_bracket = Button(master)
button_close_bracket.place(x = 260, y = 280, width = 55, height = 30)
button_C = Button(master)
button_C.place(x = 335, y = 280, width = 55, height = 30)
button_D = Button(master)
button_D.place(x = 405, y = 280, width = 55, height = 30)

button_square = Button(master)
button_square.place(x = 15, y = 180, width = 50, height = 30)
button_log = Button(master)
button_log.place(x = 85, y = 180, width = 50, height = 30)
button_pi = Button(master)
button_pi.place(x = 155, y = 180, width = 50, height = 30)
button_square_root = Button(master)
button_square_root.place(x = 225, y = 180, width = 50, height = 30)
button_factorial = Button(master)
button_factorial.place(x = 295, y = 180, width = 50, height = 30)
button_power = Button(master)
button_power.place(x = 365, y = 180, width = 50, height = 30)
button_comma = Button(master)
button_comma.place(x = 435, y = 180, width = 50, height = 30)
button_sine = Button(master)
button_sine.place(x = 15, y = 230, width = 50, height = 30)
button_cosine = Button(master)
button_cosine.place(x = 85, y = 230, width = 50, height = 30)
button_tangent = Button(master)
button_tangent.place(x = 155, y = 230, width = 50, height = 30)
button_store = Button(master)
button_store.place(x = 225, y = 230, width = 50, height = 30)
button_nCr = Button(master)
button_nCr.place(x = 295, y = 230, width = 50, height = 30)
button_x10 = Button(master)
button_x10.place(x = 365, y = 230, width = 50, height = 30)
button_random = Button(master)
button_random.place(x = 435, y = 230, width = 50, height = 30)

button_shift = Button(master)
button_shift.place(x = 20, y = 120, width = 70, height = 30)
button_options = Button(master)
button_options.place(x = 100, y = 120, width = 70, height = 30)
button_left = Button(master)
button_left.place(x = 185, y = 110, width = 30, height = 50)
button_up = Button(master)
button_up.place(x = 225, y = 110, width = 50, height = 20)
button_down = Button(master)
button_down.place(x = 225, y = 140, width = 50, height = 20)
button_right = Button(master)
button_right.place(x = 285, y = 110, width = 30, height = 50)
button_menu = Button(master)
button_menu.place(x = 330, y = 120, width = 70, height = 30)
button_help = Button(master)
button_help.place(x = 410, y = 120, width = 70, height = 30)


#Creates a list of the buttons
#so they can all be easily changed in the same way if it is necessary
button_list = [button_7, button_8, button_9, button_del, button_clear,
               button_4, button_5, button_6, button_minus, button_times,
               button_1, button_2, button_3, button_plus, button_divide,
               button_0, button_decimal, button_ans, button_StoD, button_equals,
               button_A, button_B, button_open_bracket, button_close_bracket,
               button_C, button_D,
               button_square, button_log, button_pi, button_square_root,
               button_factorial, button_power, button_comma, button_sine,
               button_cosine, button_tangent, button_store, button_nCr,
               button_x10, button_random,
               button_shift, button_options, button_left, button_up,
               button_down, button_right, button_menu, button_help]





# FUNCTIONS FOR CONFIGURING THE BUTTONS


#Function that gives a button what equates to no command
def no_command():
    pass


#Configures the buttons to work in their normal state
def normal_setup():

    button_7.configure(text = "7", command = lambda: insert_characters('7'))
    button_8.configure(text = "8", command = lambda: insert_characters('8'))
    button_9.configure(text = "9", command = lambda: insert_characters('9'))
    button_del.configure(text = "DEL", command = delete_character)
    button_clear.configure(text = "AC", command = clear_command)
    button_4.configure(text = "4", command = lambda: insert_characters('4'))
    button_5.configure(text = "5", command = lambda: insert_characters('5'))
    button_6.configure(text = "6", command = lambda: insert_characters('6'))
    button_minus.configure(text = "-", command = lambda: insert_characters('-'))
    button_times.configure(text = "*", command = lambda: insert_characters('*'))
    button_1.configure(text = "1", command = lambda: insert_characters('1'))
    button_2.configure(text = "2", command = lambda: insert_characters('2'))
    button_3.configure(text = "3", command = lambda: insert_characters('3'))
    button_plus.configure(text = "+", command = lambda: insert_characters('+'))
    button_divide.configure(text = "/", command = lambda: insert_characters('/'))
    button_0.configure(text = "0", command = lambda: insert_characters('0'))
    button_decimal.configure(text = ".", command = lambda: insert_characters('.'))
    button_ans.configure(text = "Ans", command = lambda: insert_characters('Ans'))
    button_StoD.configure(text = "S↔D", command = lambda: StoD(ans.get()))
    button_equals.configure(text = "=", command = equals_command)

    button_A.configure(text = "A", command = lambda: insert_characters('A'))
    button_B.configure(text = "B", command = lambda: insert_characters('B'))
    button_open_bracket.configure(text = "(", command = lambda: insert_characters('('))
    button_close_bracket.configure(text = ")", command = lambda: insert_characters(')'))
    button_C.configure(text = "C", command = lambda: insert_characters('C'))
    button_D.configure(text = "D", command = lambda: insert_characters('D'))

    button_shift.configure(text = "SHIFT", command = change_layout_shift)
    button_options.configure(text = "OPT", command = options)
    button_left.configure(text = "←", command = lambda: move_cursor('←'))
    button_up.configure(text = "↑", command = lambda: move_cursor('↑'))
    button_down.configure(text = "↓", command = lambda: move_cursor('↓'))
    button_right.configure(text = "→", command = lambda: move_cursor('→'))
    button_menu.configure(text = "MENU", command = menu)
    button_help.configure(text = "HELP", command = help_guide)

    button_square.configure(text = "**2", command = lambda: insert_characters('**2'))
    button_log.configure(text = "log", command = lambda: insert_characters('log('))
    button_pi.configure(text = "π", command = lambda: insert_characters('π'))
    button_square_root.configure(text = "√", command = lambda: insert_characters('√('))
    button_factorial.configure(text = "x!", command = lambda: insert_characters('fac('))
    button_power.configure(text = "x**", command = lambda: insert_characters('**'))
    button_comma.configure(text = ",", command = lambda: insert_characters(','))
    button_sine.configure(text = "sin", command = lambda: insert_characters('sin('))
    button_cosine.configure(text = "cos", command = lambda: insert_characters('cos('))
    button_tangent.configure(text = "tan", command = lambda: insert_characters('tan('))
    button_store.configure(text = "STO", command = lambda: insert_characters('='))
    button_nCr.configure(text = "nCr", command = lambda: insert_characters('nCr('))
    button_x10.configure(text = "*10", command = lambda: insert_characters('*10**'))
    button_random.configure(text = "ran", command = lambda: insert_characters('ran('))


#Configures the important buttons that never change
def key_buttons():
    button_options.configure(text = "OPT", command = options)
    button_left.configure(text = "←", command = lambda: move_cursor('←'))
    button_up.configure(text = "↑", command = lambda: move_cursor('↑'))
    button_down.configure(text = "↓", command = lambda: move_cursor('↓'))
    button_right.configure(text = "→", command = lambda: move_cursor('→'))
    button_menu.configure(text = "MENU", command = menu)
    button_help.configure(text = "HELP", command = help_guide)
    button_clear.configure(text = "AC", command = clear_command)


#Configures the numbers and letters that are often in this format
def numbers_and_letters():
    button_7.configure(text = "7", command = lambda: insert_characters('7'))
    button_8.configure(text = "8", command = lambda: insert_characters('8'))
    button_9.configure(text = "9", command = lambda: insert_characters('9'))
    button_4.configure(text = "4", command = lambda: insert_characters('4'))
    button_5.configure(text = "5", command = lambda: insert_characters('5'))
    button_6.configure(text = "6", command = lambda: insert_characters('6'))
    button_1.configure(text = "1", command = lambda: insert_characters('1'))
    button_2.configure(text = "2", command = lambda: insert_characters('2'))
    button_3.configure(text = "3", command = lambda: insert_characters('3'))
    button_0.configure(text = "0", command = lambda: insert_characters('0'))
    button_A.configure(text = "A", command = lambda: insert_characters('A'))
    button_B.configure(text = "B", command = lambda: insert_characters('B'))
    button_C.configure(text = "C", command = lambda: insert_characters('C'))
    button_D.configure(text = "D", command = lambda: insert_characters('D'))


#Sets the command of all the buttons to no command
def clear_buttons():
    for i in button_list:
        i.configure(text = "", command = no_command)


#Configures the buttons for their shift setup
def shift_setup():
    menu_before_called = current_menu.get()
    button_square.configure(text = "**3", command = lambda: insert_characters('**3'))
    button_log.configure(text = "ln", command = lambda: insert_characters('ln('))
    button_pi.configure(text = "e", command = lambda: insert_characters('e'))
    button_square_root.configure(text = "Abs", command = lambda: insert_characters('Abs('))
    button_factorial.configure(text = "ẋ", command = lambda: insert_characters('rec('))
    button_power.configure(text = "i", command = lambda: insert_characters('j'))
    button_comma.configure(text = "x", command = lambda: insert_characters('x'))
    button_sine.configure(text = "sin-", command = lambda: insert_characters('arcsin('))
    button_cosine.configure(text = "cos-", command = lambda: insert_characters('arccos('))
    button_tangent.configure(text = "tan-", command = lambda: insert_characters('arctan('))
    button_store.configure(text = "INF", command = recall)
    button_nCr.configure(text = "nPr", command = lambda: insert_characters('nPr('))
    button_x10.configure(text = "%", command = lambda: insert_characters('%('))
    button_random.configure(text = "ran#", command = lambda: insert_characters('ran_int()'))
    if not (menu_before_called == 'calculations_menu()' or
            menu_before_called == 'matrices_menu()' or
            menu_before_called == 'vectors_menu()'):
        button_power.configure(text = "**-1", command = lambda: insert_characters('**-1'))
    if (menu_before_called == 'matrices_menu()' or
        menu_before_called == 'vectors_menu()'):
        button_power.configure(text = "π", command = lambda: insert_characters('π'))
        button_random.configure(text = "log", command = lambda: insert_characters('log('))


#The function that runs when shift is pressed
#swithces between the normal setup and the shift setup
def change_layout_shift():
    shifted = shift.get()
    menu_before_called = current_menu.get()
    last_screen.set(screen.get('1.0', 'end'))
    screen_before_called = last_screen.get()
    if not shifted:
        shift_setup()
        shift.set(True)
    else:
        eval(menu_before_called)
        clear()
        screen.insert(END, screen_before_called)
        screen.delete('end-1l','end')
        shift.set(False)


#The configuration of the calculator when in the menu for changing the background colour
def change_background_colour():
    clear()
    clear_buttons()
    key_buttons()
    numbers_and_letters()
    screen.insert(END, "1 - dark grey \n" +
                  "2 - light grey \n" +
                  "3 - black \n" +
                  "4 - white \n" +
                  "5 - orange \n" +
                  "6 - light blue \n" +
                  "7 - dark blue \n" +
                  "8 - brown \n"  +
                  "9 - purple \n" +
                  "0 - red \n" +
                  "A - pink \n" +
                  "B - yellow \n" +
                  "C - dark green \n" +
                  "D - light green")
    #Sets up the number buttons to change the background to a certain specific colour
    button_1.configure(text = "1", command = lambda: change_colour('dark grey'))
    button_2.configure(text = "2", command = lambda: change_colour('light grey'))
    button_3.configure(text = "3", command = lambda: change_colour('black'))
    button_4.configure(text = "4", command = lambda: change_colour('white'))
    button_5.configure(text = "5", command = lambda: change_colour('orange'))
    button_6.configure(text = "6", command = lambda: change_colour('light blue'))
    button_7.configure(text = "7", command = lambda: change_colour('dark blue'))
    button_8.configure(text = "8", command = lambda: change_colour('brown'))
    button_9.configure(text = "9", command = lambda: change_colour('purple'))
    button_0.configure(text = "0", command = lambda: change_colour('red'))
    button_A.configure(text = "A", command = lambda: change_colour('pink'))
    button_B.configure(text = "B", command = lambda: change_colour('yellow'))
    button_C.configure(text = "C", command = lambda: change_colour('dark green'))
    button_D.configure(text = "D", command = lambda: change_colour('light green'))


#The function that runs when the options button is pressed
#switches to the options configuration
def options():
    in_option = in_options.get()
    menu_before_called = current_menu.get()
    screen_before_called = last_screen.get()
    degrees = not_radians.get()
    second_equation = two_equations.get()
    clear_buttons()
    key_buttons()
    #Sets up the buttons to allow you to change a certain configuration
    if not in_option:
        last_screen.set(screen.get('1.0', 'end'))
        clear()
        if degrees:
            screen.insert(END, "1 - Change Angle Units to Radians \n")
        else:
            screen.insert(END, "1 - Change Angle Units to Degrees \n")
        screen.insert(END, "2 - Background Colour \n")
        if menu_before_called == 'table_menu()':
            if not second_equation:
                screen.insert(END, "3 - Add second Equation for Table Function ")
            else:
                screen.insert(END, "3 - Remove second Equation for Table Function ")
        in_options.set(True)
        button_1.configure(text = "1", command = change_angle_unit)
        button_2.configure(text = "2", command = change_background_colour)
        button_3.configure(text = "3", command = change_number_of_equations)
    #If already in the options menu the button returns you to the previous menu
    else:
        clear()
        eval(menu_before_called)
        clear()
        screen.insert(END, screen_before_called)
        in_options.set(False)


#The function that is called when the menu button is pressed
#that changes the number buttons to allow you to access alternative function menus 
def menu():
    in_menu = in_menus.get()
    menu_before_called = current_menu.get()
    clear()
    clear_buttons()
    key_buttons()
    if not in_menu:
        screen.insert(END, "1 - Calculations \n" +
                      "2 - Integration \n" +
                      "3 - Differentiation \n" +
                      "4 - Summation \n" +
                      "5 - Matrices \n" +
                      "6 - Vectors \n" +
                      "7 - 1-Variable Statistics \n" +
                      "8 - 1-Variable Statistics with frequency \n" +
                      "9 - 2-Variable Statistics \n" +
                      "0 - Binomial PD \n" +
                      "A - Binomial CD \n" +
                      "B - Table \n" +
                      "C - Solve Quadratics \n" +
                      "D - Solve Simultaneous")
        in_menus.set(True)
        button_1.configure(text = "1", command = calculations_menu)
        button_2.configure(text = "2", command = integration_menu)
        button_3.configure(text = "3", command = differentiation_menu)
        button_4.configure(text = "4", command = summation_menu)
        button_5.configure(text = "5", command = matrices_menu)
        button_6.configure(text = "6", command = vectors_menu)
        button_7.configure(text = "7", command = one_variable_statistics_menu)
        button_8.configure(text = "8", command = one_variable_statistics_with_frequency_menu)
        button_9.configure(text = "9", command = two_variable_statistics_menu)
        button_0.configure(text = "0", command = binomial_PD_menu)
        button_A.configure(text = "A", command = binomial_CD_menu)
        button_B.configure(text = "B", command = table_menu)
        button_C.configure(text = "C", command = solve_quadratics_menu)
        button_D.configure(text = "D", command = solve_simultaneous_menu_1)
    else:
        in_menus.set(False)
        eval(menu_before_called)


#This function runs when you need to choose the dimensions of a matrix or vector
def dimensions():
    last_screen.set(screen.get('1.0', 'end'))
    menu_before_called = current_menu.get()
    clear()
    clear_buttons()
    key_buttons()
    numbers_and_letters()
    button_A.configure(text = "", command = no_command)
    button_B.configure(text = "", command = no_command)
    button_C.configure(text = "", command = no_command)
    button_D.configure(text = "", command = no_command)
    if menu_before_called == 'matrices_menu()':
        button_times.configure(text = "*", command = lambda: insert_characters('*'))
    button_del.configure(text = "DEL", command = delete_character)
    button_equals.configure(text = "=", command = dimensions_chosen)
    screen.insert(END, "Choose Dimensions: \n")


#This function runs after the dimensions of the matrix or vector have been chosen
def dimensions_chosen():
    dimensions = screen.get('2.0', 'end')
    clear()
    screen_before_called = last_screen.get()
    menu_before_called = current_menu.get()
    clear_buttons()
    key_buttons()
    numbers_and_letters()
    eval(menu_before_called)
    screen.insert(END, screen_before_called)
    delete_character()
    #A matrix needs 2 dimensional inputs seperated by a '*' while a vector only needs 1
    #here the necessary list size is outputted to the screen saving the user from lots of typing
    if '*' in dimensions:
        dimensions = dimensions.split('*')
        screen.insert(END, ' = \n[')
        for i in range(1, int(dimensions[0]) + 1):
            if not i == 1:
                screen.insert(END, ' ')
            screen.insert(END, '[')
            for j in range(1, int(dimensions[1])):
                screen.insert(END, ' ,')
            screen.insert(END, ' ]')
            if not i == int(dimensions[0]):
                screen.insert(END, ',\n')
        screen.insert(END, ']')
    else:
        screen.insert(END, ' = \n[ ')
        for i in range(1, int(dimensions)):
            screen.insert(END, ', ')
        screen.insert(END, ']')    









# MENU FUNCTIONS

#The following functions configure the UI, the screen and the buttons
#for each of their menus for their specific functions

def calculations_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("calculations_menu()")
    clear()
    normal_setup()


def integration_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("integration_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "x", command = lambda: insert_characters('x'))
    button_equals.configure(text = "=", command = integration_equals)
    screen.insert(END, "Upper Bound = \n" +
                  ";Lower Bound = \n" +
                  ";Equation = ")
    screen.mark_set(INSERT, '1.14')
                

def differentiation_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("differentiation_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "x", command = lambda: insert_characters('x'))
    button_equals.configure(text = "=", command = differentiation_equals)
    screen.insert(END, "x = \n" +
                  ";Equation = ")
    screen.mark_set(INSERT, '1.4')


def summation_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("summation_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "x", command = lambda: insert_characters('x'))
    button_equals.configure(text = "=", command = summation_equals)
    screen.insert(END, "r = \n" +
                  ";n = \n" +
                  ";Equation = ")
    screen.mark_set(INSERT, '1.4')


def matrices_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("matrices_menu()")
    clear()
    normal_setup()
    button_equals.configure(text = "=", command = matrix_equals)
    button_ans.configure(text = "Ans", command = lambda: insert_characters('MatAns'))
    button_A.configure(text = "MatA", command = lambda: insert_characters('MatA'))
    button_B.configure(text = "MatB", command = lambda: insert_characters('MatB'))
    button_C.configure(text = "MatC", command = lambda: insert_characters('MatC'))
    button_D.configure(text = "MatD", command = lambda: insert_characters('MatD'))
    button_square.configure(text = "det", command = lambda: insert_characters('Det('))
    button_log.configure(text = "co", command = lambda: insert_characters('Cofactors('))
    button_pi.configure(text = "min", command = lambda: insert_characters('Minors('))
    button_square_root.configure(text = "tran", command = lambda: insert_characters('Transpose('))
    button_random.configure(text = "√", command = lambda: insert_characters('√('))
    button_store.configure(text = "STO", command = dimensions)
    
      
def vectors_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("vectors_menu()")
    clear()
    normal_setup()
    button_equals.configure(text = "=", command = vector_equals)
    button_ans.configure(text = "Ans", command = lambda: insert_characters('VecAns'))
    button_A.configure(text = "VecA", command = lambda: insert_characters('VecA'))
    button_B.configure(text = "VecB", command = lambda: insert_characters('VecB'))
    button_C.configure(text = "VecC", command = lambda: insert_characters('VecC'))
    button_D.configure(text = "VecD", command = lambda: insert_characters('VecD'))
    button_log.configure(text = "Abs", command = lambda: insert_characters('Abs('))
    button_pi.configure(text = "unit", command = lambda: insert_characters('Unit('))
    button_square_root.configure(text = "ang", command = lambda: insert_characters('Angle('))
    button_random.configure(text = "√", command = lambda: insert_characters('√('))
    button_store.configure(text = "STO", command = dimensions)

  
def one_variable_statistics_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("one_variable_statistics_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = one_variable_statistics_equals)
    screen.insert(END, "x = ")
    screen.mark_set(INSERT, '1.4')


def one_variable_statistics_with_frequency_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("one_variable_statistics_with_frequency_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = one_variable_statistics_with_frequency_equals)
    screen.insert(END, "x = ")
    screen.insert(END, "\n;f = ")
    screen.mark_set(INSERT, '1.4')


def two_variable_statistics_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("two_variable_statistics_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = two_variable_statistics_equals)
    screen.insert(END, "x = ")
    screen.insert(END, "\n;y = ")
    screen.mark_set(INSERT, '1.4')

                  
def binomial_PD_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("binomial_PD_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = binomial_PD_equals)
    screen.insert(END, "X = \n" +
                  "; N = \n" +
                  "; p = ")
    screen.mark_set(INSERT, '1.4')


def binomial_CD_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("binomial_CD_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = binomial_CD_equals)
    screen.insert(END, "X = \n" +
                  "; N = \n" +
                  "; p = ")
    screen.mark_set(INSERT, '1.4')


def table_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    two_equations.set(False)
    current_menu.set("table_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "x", command = lambda: insert_characters('x'))
    button_equals.configure(text = "=", command = table_equals)
    screen.insert(END, "Lower Bound = " +
                  "\n;Upper Bound = " +
                  "\n;Step = " +
                  "\n;Equation f(x) = ")
    screen.mark_set(INSERT, '1.14')

    
def solve_quadratics_menu():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("solve_quadratics_menu()")
    clear()
    normal_setup()
    button_store.configure(text = "x", command = lambda: insert_characters('x'))
    button_equals.configure(text = "=", command = solve_quadratic_equals)
    screen.insert(END, " x**2 + x +   = 0")
    screen.mark_set(INSERT, '1.1')


#In this menu the user chooses the number of equations they want to solve
def solve_simultaneous_menu_1():
    in_menus.set(False)
    in_options.set(False)
    shift.set(False)
    current_menu.set("solve_simultaneous_menu_1()")
    clear()
    clear_buttons()
    key_buttons()
    screen.insert(END, "1 - Two Variables" +
                  "\n2 - Three Variables")
    button_1.configure(text = "1", command = lambda: solve_simultaneous_menu_2('2'))
    button_2.configure(text = "2", command = lambda: solve_simultaneous_menu_2('3'))


#This function sets up the menu for simulataneous equations
#the screen output is different depending on the users choice for the number of equations
def solve_simultaneous_menu_2(n):
    clear()
    normal_setup()
    button_store.configure(text = "INF", command = recall)
    button_equals.configure(text = "=", command = solve_simultaneous_equals)
    if n == '2':
        current_menu.set("solve_simultaneous_menu_2('2')")
        screen.insert(END, " x + y = " +
                      "\n; x + y = ")
    elif n == '3':
        current_menu.set("solve_simultaneous_menu_2('3')")
        screen.insert(END, " x + y + z = " +
                      "\n; x + y + z = " +
                      "\n; x + y + z = ")
    screen.mark_set(INSERT, '1.1')










calculations_menu()
master.mainloop()




