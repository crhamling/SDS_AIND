#
# Sudoku Agent Script
#

rows = 'ABCDEFGHI'
cols = '123456789'


# FUNCTION Cross:
# helper function, cross(a, b), which, given two strings — a and b — 
# will return the list formed by all the possible concatenations of a 
# letter s in string a with a letter t in string b.

def cross(a, b):
    return [s+t for s in a for t in b]

# create the labels of the boxes
# Create BOXES by executing the FN cross on all of the rows and columns
boxes = cross(rows, cols)

# What is the type of "boxes"? -> LIST
type(boxes)

#
# create the various containers:
#
# Boxes, Units and Peers
# And let's start naming the important elements created by these rows and columns that are relevant to solving a Sudoku:
#
#	* The individual squares at the intersection of rows and columns will be called boxes. These boxes will have labels 'A1', 'A2', ..., 'I9'.
#	* The complete rows, columns, and 3x3 squares, will be called units. Thus, each unit is a set of 9 boxes, and there are 27 units in total.
#	* For a particular box (such as 'A1'), its peers will be all other boxes that belong to a common unit (namely, those that belong to the same row, column, or 3x3 square).
#
#

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)



def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

# Example
# display(grid_values('..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'))


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Input: A grid in string form.
    Output: A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))
    
    
# Strategy One - Eliminate
def eliminate(values):
    """Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    # TODO: Implement only choice strategy here
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values        
    
    
# Strategy Two - Only Choice
def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    # TODO: Implement only choice strategy here
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values    
    


def reduce_puzzle(values):
    stalled = False
    # Loop until stalled
    while not stalled:
        
        # Check how many boxes have a determined value AKA : (len(values[box]) == 1] )
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        values = eliminate(values)

        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
            
    return values


    
#    
# Testing the Functions
#


# Create the test grid:
# sudoku_grid = grid_values('..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..')

# Harder Problem:
grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
sudoku_grid = grid_values(grid2)


# Check Boxes Variable
# boxes

# Zip Check
# zip(boxes, sudoku_grid)
    
# Dictionary Check
# dict(zip(boxes, sudoku_grid))


# Show the grid before ANY strategy:
# display(sudoku_grid)


# Show the grid after using the Elimination Strategy:
# NOTE: Run this once.
# display(eliminate(sudoku_grid))


# Show the grid after using the Only Choice Strategy:
# NOTE: Run this at least twice to get the same result as the grader
# display(only_choice(sudoku_grid))

# Execute the solution Strategy - Reduce_Puzzle
# display(reduce_puzzle(sudoku_grid))





