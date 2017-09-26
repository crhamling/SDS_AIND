#
# Sudoku Agent Script
#

#
# Assignment List - to be used in visualizations
#
assignments = []



#
# Rows & Columns
#
rows = 'ABCDEFGHI'
cols = '123456789'


#
# FUNCTION Cross
# 
# helper function, cross(a, b), which, given two strings — a and b — 
# will return the list formed by all the possible concatenations of a 
# letter s in string a with a letter t in string b.

def cross(a, b):
    "Cross product of elements in A and elements in B."
    return [s+t for s in a for t in b]
    

#
# create the various containers:
#
# Boxes, Units and Peers
# And let's start naming the important elements created by these rows and columns that are relevant to solving a Sudoku:
#
#	* The individual squares at the intersection of rows and columns will be called boxes. 
#        These boxes will have labels 'A1', 'A2', ..., 'I9'.
#	* The complete rows, columns, and 3x3 squares, will be called units. Thus, each unit is a set of 9 boxes, 
#        and there are 27 units in total.
#	* For a particular box (such as 'A1'), its peers will be all other boxes that belong to a common unit 
#        (namely, those that belong to the same row, column, or 3x3 square).


# Create BOXES of a sudoku by executing the FN cross on all of the rows and columns
boxes = cross(rows, cols)

# Create ROW UNITS of a sudoku
row_units = [cross(r, cols) for r in rows]

# Create COLUMN UNITS of a sudoku
column_units = [cross(rows, c) for c in cols]

# Create SQUARE UNITS of a sudoku
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

# Define diagonal units of a sudoku (This is how you solve the Diagnonal Sudoku problem)
diagonal_units = [[x+y for x, y in zip(rows, cols)], [x+y for x, y in zip(rows, cols[::-1])]]

# Create UNITSLISTS of ALL units in a sudoku 
# Note - once you include the diagonal units, the code no longer solves regular sudoku problems)
# unitlist = row_units + column_units + square_units 
unitlist = row_units + column_units + square_units + diagonal_units

# Create UNITS of a sudoku
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)

# Create PEERS of a sudoku
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)



#
# FUNCTION assign_values
#
# To visualize your solution, please *only* assign values to the values_dict 
# using the assign_value function provided in solution.py
#
def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board, record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    # NOTE: Causes Trouble in the Diagonal Sudoku - so it's Nix'd
    #
    # if values[box] == value:
    #     return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

#
# FUNCTION: grid_values
#
def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
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


#
# FUNCTION: display
#
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


#
# Strategy One - Eliminate
#
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
    evalues = values.copy()
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            # Standard Txt Visualization
            values[peer] = values[peer].replace(digit,'')

            # Assign Values for pygame visual display
            assign_value(evalues, peer, evalues[peer].replace(values[box], ''))
    return values 


#
# Strategy Two - Only Choice
#
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
                # Standard Txt Visualization
                values[dplaces[0]] = digit
                
                # Assign Values for pygame visual display
                assign_value(values, dplaces[0], digit)
    return values 


#
# Strategy Three - Naked Twins
#

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        values(dict): dictionary with the naked twins eliminated from all peers.
    """
    # Find all instances of naked twins
    naked_twin_dict = {}
    for unit in unitlist:
        # Build a dictionary to identify a naked twin pair
        nt_dict = {}
        for box in unit:
            # Identify box containing only 2 possibilities as possible naked twin
            if len(values[box]) == 2:
                if not values[box] in nt_dict:
                    nt_dict[values[box]] = [box]
                else:
                    nt_dict[values[box]].append(box)
        # Examine the dictionary to validate the NT candidates 
        for key in nt_dict:
            # Test the candidate - is naked twin pair?
            if len(nt_dict[key]) == 2:
                if not key in naked_twin_dict:
                    naked_twin_dict[key] = [unit]
                else:
                    naked_twin_dict[key].append(unit)

    # Eliminate the naked twins as possibilities for their peers
    for key in naked_twin_dict:
        for unit in naked_twin_dict[key]:
            for box in unit:
                if values[box] != key:
                    # Standard Txt Visualization
                    values[box] = values[box].replace(key[0], '')
                    
                    # Assign Values for pygame visual display
                    assign_value(values, box, values[box].replace(key[0], ''))
                    
                    # Standard Txt Visualization
                    values[box] = values[box].replace(key[1], '')
                    
                    # Assign Values for pygame visual display
                    assign_value(values, box, values[box].replace(key[1], ''))
                    
    return values


#
# Strategy Four - Reduce   
#
# Note - This function employs the following Strategies:
#         - Eliminate
#         - Only Choice
#         - Naked Twins
#         - 
def reduce_puzzle(values):
    stalled = False
    # Loop until stalled
    
    while not stalled:
        
        # Check how many boxes have a determined value AKA : (len(values[box]) == 1] )
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        
        # Use the Eliminate Strategy
        values = eliminate(values)

        # Use the Only Choice Strategy
        values = only_choice(values)

        # Use the Naked Twins Strategy
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
            
    return values


#
# Strategy Five - Search
#
def search(values):
    
    "Using depth-first search and propagation, try all possible values."
    
    # First, reduce the puzzle using the previous function
    # Choose one of the unfilled squares with the fewest possibilities
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    
    # First, reduce the puzzle using Strategy Three - "The Reduce Function" (which uses the 'elimination' and 'only choice' strategies)
    values = reduce_puzzle(values)
    
    if values is False:
        return False ## Failed earlier
        
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved Puzzle!
        
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # print(n, s)
    
    # Now use recursion to solve each one of the resulting sudokus, 
    # and if one returns a value (not False), return that answer!
    for value in values[s]:
        # print(value)
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt


#
# Submission Solver 
# Takes Grid in Raw Grid Form, not "Values" Form like the strategy functions
#
def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    
    values = grid_values(grid)
    values = search(values)
    return values

#
# BEGIN Testing Harness - grader ignore - for testing end understanding purposes only
#


# Create the test grid:
# sudoku_grid = grid_values('..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..')

# Harder Problem:
# sudoku_grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
# sudoku_grid = grid_values(sudoku_grid2)

# Submission Solver Test harnass - Diagonal Grid
# diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
# sudoku_grid = grid_values(diag_sudoku_grid)


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


# Execute the solution Strategy - Search
# display(search(sudoku_grid))

# display(solve(sudoku_grid2))

# Execute the SUBMISSION solution Strategy - Solve
# display(solve(diag_sudoku_grid))


#
# END Testing Harness - grader ignore - for testing end understanding purposes only
#


#
# Main Module (if run from the Command Line)
#
if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')




