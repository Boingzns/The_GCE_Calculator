# The_GCE_Calculator
Welcome to The GCE Calculator, the scientific calculator that I have designed, using python and tkinter,for my EPQ project 
that has all the functions required by a student studying for their first year of GCE advanced level mathematics and further mathematics.
The Windows executable file, Mac executable file and the python file are available to download and use.


​
​


The GCE Calculator Guide:

​

1 – Calculations:

​

Functions:

​

All functions need to be followed by an open bracket, the input, then the close bracket.

Example: fac(x)

Some functions require two inputs this is done by entering: the first input, a comma, then the second input, within the brackets. The functions that need two inputs are log and ran.

Example: log(a,b)

With log the first input is the base and the second input is the argument.

With ran the first input is the lower bound and the second input is the upper bound; the output of this function is a random number between and including the two bounds.

Ran# outputs a random integer between 1 and 1000

The Ẋ button symbolises the recursive function and enters rec( to the screen which works by turning the decimal part of the of the input into a recursive decimal part.

Example: rec(7.12)      Output = 7.12121212…

The square root function outputs a surd only when it is the only part of the calculation on the screen.

Example: √(12)            Output = 2√(3)            √(12)*2            Output = 6.928…

Imaginary numbers can be used in this menu unlike most others by using the ‘i’ button although the output to the screen is ‘j’ which is the imaginary number for the calculator

​

Storing:

​

To store a number as a constant, enter one of the four letters A, B, C or D into the calculator and click ‘STO’ then enter the number you want to store and click ‘=’ to link the letter to the number.

Example: A=7

To see what values you have already stored hit shift then press ‘INF’.

​

Indices:

​

To use a power in the calculator, instead of using x^a use x ** a.

Example: 4 ** 2             Output: 16

​

Converting between Fractions and Decimals:

​

The ‘StoD’ button is used to convert answers between fractions and decimals.

It is also used to convert a surd into a real number approximation.

Warning: please be aware that the conversion to fractions is an approximation; the calculator will convert any number outputted to the screen to a fraction, it does this by assuming that the number is rational and does not take into account that the number may have been rounded and so a fraction is given regardless of whether the answer was rational or irrational and so it may not always give a correct fraction.

 

​

2/3/4 – Integration, Differentiation and Summation:

​

Integration:

​

This calculation takes 3 inputs the equation and the lower and upper bounds that the equation is integrated between.

The equation must contain no brackets between terms, each term must be separated by either a ‘+’ or a ‘-’, there may not be either a ‘+’ or a ‘-’ in the coefficients of a term.

Form: ax**k + bx**n…

Example: Equation = 3x**2 – x**-4 + (7/2)x + 3

​

Differentiation:

​

Same rules apply as with integration except that the inputs are the equation and the value of x for the differentiation to take place at.

​

Summation:

​

Same rules apply as with integration except that the inputs are the equation, the starting number r and the final number n.

 

​

5/6 – Matrices and Vectors:

​

Matrices:

​

Matrices are defined by entering one of the 4 matrix assignable constants (MatA, MatB, MatC and MatD) and clicking the button ‘STO’

You will then be asked to enter dimensions for the matrix; do this by entering the number of rows times (‘*’) the number of columns you want your matrix to have.

Example: 2*2

Then a number of brackets will show up on the screen indicating the size of the matrix you have chosen and you need to enter the numbers for each part of the matrix into these brackets.

Example for a 1*2 matrix: [[3 , 4]]

Defined matrices can be recalled with the ‘INF’ button which will display them on the screen

The following matrix functions have been made into buttons: determinant (‘det’), transpose (‘tran’), cofactors (‘co’) and minors (‘min’). They all need matrix inputs and the last two require 3*3 matrices.

Matrix inversion works by entering the matrix followed by ‘**-1’, but only with matrices of dimensions 2*2 or 3*3.

​

Vectors:

​

Defining the vector constants works the same way as with matrices however only one number for dimensions is needed.

The following vector functions have been made into buttons: Absolute value (‘Abs’), unit vector (‘unit’), angle between two vectors (‘ang’). They all need vector inputs.

Use ‘*’ to find the cross product of two vectors and ‘.’ To find the dot product of two vectors.

Example: VecA*VecB              Example: VecA.VecB

 

​

7/8/9 – 1 and 2 Variable Statistics

​

1 – Variable Statistics:

​

The list of x values needs to be entered by separating each of the numbers with a comma (‘,’).

Example: x = 2,4,7

Upon pressing the ‘=’ button, the calculator will output an array of statistical calculations made from the list of values for x

​

1 – Variable Statistics with Frequency:

​

Same rules as with 1 – variable statistics except that two lists need to be entered: the values of x and the values of f. The two lists need to be the same length.

​

2 – Variable Statistics:

​

Same rules as with 1 – variable statistics except that two lists need to be entered: the values of x and the values of y. The two lists need to be the same length.

 

​

0/A – Binomial PD and CD:

​

Binomial PD:

​

Binomial probability distribution; it needs 3 inputs: X, N and p and calculates the probability of choosing X items from a sample of N with a probability of p.

If you want to quickly calculate this for a number of X values, a list of X values can be entered separating each one with a comma (‘,’) like with the 1 – variable statistics.

​

Binomial CD:

​

Binomial cumulative distribution; it needs 3 inputs: X, N and p and calculates the probability of choosing up to X items from a sample of N with a probability of p.

Otherwise follows the same rules as Binomial PD.

 

​

B – Table:

​

This menu calculates the range of a function for an entered domain.

The table takes in 4 inputs: the upper and lower bounds and the step number that defines the range by using every number between the upper and lower bounds that is a multiple of the step larger that the lower bound; and the equation whose domain is to be found.

The equation must follow the same syntax rules as the equation specified in the integration section.

A second equation for use with the same range can be used by clicking ‘OPT’ to enter the options menu and choosing the option to add a second equation; this method can also be used to remove the second equation.

 

​

C – Solve Quadratic:

​

Calculates the solutions and the minimum or maximum values for a quadratic equation.

Form: ax**2 + bx + c = 0

The equation must follow the same syntax rules as the equation specified in the integration section.

The ‘StoD’ button can be used to convert the solutions outputted to the screen between surds and real numbers.

 

​

D – Solve Simultaneous

​

Solves simultaneous equations with either 2 or 3 unknown variables.

Solving for 2 or 3 variables is chosen upon the selection of this menu.

Form: ax + by = c         Form: ax + by + cz = d

The ‘StoD’ button can be used to convert the solutions outputted to the screen between fractions and decimalised numbers.
