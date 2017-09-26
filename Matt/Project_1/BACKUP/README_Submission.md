# Artificial Intelligence Nanodegree
## Introductory Project: Extending the Sudoku Solver Algorithm

<BR>

In this project we write code to implement two extensions of the Sudoku Solver.  The **first** one will be to implement the technique called "naked twins". The **second** one will be to modify our existing code to solve a Diagonal Sudoku.



When successfully run, the output resembles that of the below screens.

<BR>

<img src='img/sudoku_run.jpg'>


# Question 1 (Naked Twins)
**Q:** *How do we use constraint propagation to solve the naked twins problem?*

**A:** Constraint satisfaction is used to reduce the search space of a problem (the set of possibilities) to the point where it makes the problem easier to solve. Constraint propagation are the transformations that are executed on the problem which change the problem (space) without changing the problem solution(s). 


The **Naked Twins** is one such transformation that identifies a **pair of boxes** belonging to the same set of **peers** (all other boxes that belong to a common **unit** [aka: those that belong to the same row, column, or 3x3 square]), which have the same 2 numbers as **possibilities**, and then eradicate those two numbers from **all** the boxes that have these two boxes as **peers** as shown below:
<BR>


<BR>

<img src='img/naked-twins.jpg'>

<BR>

> The implementation of this code is **below**: 


<img src='img/naked_twins_code.jpg'>



# Question 2 (Diagonal Sudoku)
**Q:** *How do we use constraint propagation to solve the diagonal Sudoku problem?*

**A:** DEFN: A diagonal sudoku is like a regular sudoku, except that among the two main diagonals, the numbers 1 to 9 should all appear **exactly** once. 

<BR>

<img src='img/diagonal-sudoku.jpg'>

The most efficient method for including the constraint of "the diagonal" is to include it as a **unit** into the set of **peers** that comprised the Sudoku problem data, this  effectively prevents accepting of solutions that do not satisfy the diagonal constraint, as implemented in the highlighted code below, and illustrated in the solution screen at the top.

<BR>

<img src='img/diagonal_sudoku_code.jpg'>















