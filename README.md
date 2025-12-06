# CSM: GAUSS-JORDAN ELIMINATION PROJECT (Square matrix solver)

**_Progress:_** ![Progress](https://progress-bar.xyz/100)

An application to solve matrices through **Gauss-Jordan**
Elimination Method of a Linear Equation with the integration
of _Python_ programming language and importing a package called
_NumPy_ for proper computing and _PyInstaller_ for compiling the application.
and by using an algorithm in solving the problem.
<br></br>

## WHAT IS GAUSS-JORDAN ELIMINATION

* Gauss-Jordan Elimination is a method of eliminating
  the unknown from all other equations rather than the
  subsequent ones.
  <br></br>
  And all rows are normalized by dividing them by their
  pivot elements resulting into an identity matrix
  rather than a triangular matrix.
  <br></br>
  Consequently, it is not needed to do back substitution
  to get the solution.

## INSTALLATION

* Made use of an alternative compiling `auto-py-to-exe`
* Added requirements for dependencies
* But the application is standalone, offline, and do not need external imports

## ALGORITHM

* [refer to mechtutor's code algorithm and step-by-step
  solving method]
* ![Screenshot 2025-11-24 212440.png](ref/Screenshot%202025-11-24%20212440.png)
* Goal of Gauss-Jordan
  is that
  the `A matrix` would become an `Identity matrix` (main diagonal is 1)
* General Formulas
  ![Screenshot 2025-11-24 212440.png](ref/Screenshot%202025-11-24%20212440.png)

* Array Indexes
    - Array starts in index zero
    - so index is n-1

* Algorithm
    - k-loop for pivot rows
    - j-loop for columns
    - i-loop for elimination above and below rows
    - j-loop substraction steps for columns

## REFERENCES

* Video Guide by _mechtutor com_:
    * [Gauss-Jordan Method Tutorial](https://www.youtube.com/watch?v=xOLJMKGNivU)
      is a command-line tutorial (no gui) but hardcoded input

Please kindly read the [License](LICENSE) before using the application

---
