
import random
#import pygame


from collections import Counter

rows = 'ABCDEFGHI'
cols = '123456789'
assignments = []
itercount = 0

def cross(a,b):
    return [s+t for s in a for t in b]

boxes = cross(rows, cols)

row_units = [cross(r,cols) for r in rows]
col_units = [cross(rows,c) for c in cols]
sq_units = [cross(rs,cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diag_units = [[ir+ic for ir in rows for ic in cols if rows.index(ir) == cols.index(ic)],                  [ir+ic for ir in rows for ic in cols if rows.index(ir) == 8 - cols.index(ic)]]

unitlist = row_units + col_units + sq_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)




def assign_value(values, box, value):
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def naked_twins(values):

    for unit in unitlist:
        pairs = [(box,values[box]) for box in unit if len(values[box]) == 2]
        hist = {}
        for item in pairs:
            if item[1] in hist.keys():
                for box in [b for b in unit if values[b] != item[1]]:
                    assign_value(values, box, values[box].replace(item[1][0],''))
                    assign_value(values, box, values[box].replace(item[1][1],''))
            else:
                hist[item[1]] = 1
      
    return values
    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers


def grid_values(grid):

    output = {}
    for i in range(len(grid)):
        if(grid[i] == '.'):
            output[boxes[i]] = '123456789'
        else:
            output[boxes[i]] = grid[i]
    return output


def display(values):
    
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


def eliminate(values):
    
    for box in values:
        if len(values[box]) == 1:
            for peer in peers[box]:
                assign_value(values,peer,values[peer].replace(values[box],''))
    return values


def only_choice(values):
    
    for unit in unitlist:
        unitStr = ''
        for box in unit:
            unitStr += values[box]
        for box in unit:
            if len(values[box]) != 1:
                for choice in values[box]:
                   if unitStr.count(choice) == 1:
                       assign_value(values, box, choice)
    return values


def reduce_puzzle(values):
    
    global itercount
    stalled = False
    while not stalled:
        itercount += 1
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        #======================================================================
        #values = eliminate(only_choice(values))
        values = naked_twins(eliminate(only_choice(values)))
        #======================================================================
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        
        stalled = solved_values_before == solved_values_after
        
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    
    values = reduce_puzzle(values)
    if not values:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values 
    #display(values)
    
    _,i = min((len(values[i]), i) for i in boxes if len(values[i]) > 1)
    
    for digit in values[i]:
        new = values.copy()
        new[i] = digit
        res = search(new)
        if res:
            return res


def solve(grid):
    
    return(search(grid_values(grid)))

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
    

