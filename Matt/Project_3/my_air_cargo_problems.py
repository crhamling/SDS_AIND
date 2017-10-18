from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """
        
        #
        # The get actions method is used to construct a list of all possible actions/operator objects which 
        # can act of the various states. There are 3 types of actions, LOAD, UNLOAD, and FLY. 
        #
        #

        #print("\n SELF_DICT: ", self.__dict__, "\n")
        #
        #        self_dict = 
        #        {
        #        'state_map': [
        #            At(C1, SFO), At(C2, JFK), At(P1, SFO), At(P2, JFK), At(C2, SFO), In(C2, P1), 
        #            In(C2, P2), At(C1, JFK), In(C1, P1), In(C1, P2), 
        #            At(P1, JFK), At(P2, SFO)
        #            ], 
        #        'initial_state_TF': 'TTTTFFFFFFFF', 'initial': 'TT TTFFFFFFFF', 
        #        'goal': [At(C1, JFK), At(C2, SFO)], 
        #        'cargos': ['C1', 'C2'], 
        #        'planes': ['P1', 'P2'], 
        #        'airports': ['JFK', 'SFO']
        #        }
                
        #       CARGOS:  ['C1', 'C2']
        #       PLANES:  ['P1', 'P2']
        #       AIRPORTS:  ['JFK', 'SFO']
        

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        
        # print("\n SELF_DICT: ", self.__dict__, "\n")
        
        
        
        #
        # FN - load_actions
        #
        def load_actions():
            """Create all concrete Load actions and return a list
            
            FORMAL EXPRESSION:
            
            Action(Load(c, p, a),
            	PRECOND: At(c, a) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            	EFFECT: ¬ At(c, a) ∧ In(c, p))

            :return: list of Action objects
            """
            
            loads = []
            # TODO create all load ground actions from the domain Load action
            
            ##
            ## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
            ## we create concrete Action objects based on the domain action schema for: Load
            ##
            
            # print("\n CARGO: ", self.cargos)
            # print("PLANES: ", self.planes)
            # print("AIRPORTS: ", self.airports)
                        
            for _cargo in self.cargos:
                for _plane in self.planes:
                    for _airport in self.airports:
            
                        # print("\nCARGO: ", _cargo)
                        # print("PLANES: ", _plane)
                        # print("AIRPORTS: ", _airport)
                        
                        # PRECOND_POS: At(c, a) ∧ At(p, a)
                        # FORMAT: precond_pos = [expr("Human(person)"), expr("Hungry(Person)")]
                        # ENGLISH: There is cargo and a plane at the airport
                        precond_pos = [expr("At({}, {})".format(_cargo, _airport)), expr("At({}, {})".format(_plane, _airport))]
                        
                        # PRECOND_NEG: 
                        # FORMAT: precond_neg = [expr("Eaten(food)")]
                        # ENGLISH: -None-
                        precond_neg = []
                        
                        # FORMAT: effect_add =  [expr("Eaten(food)")]
                        # Add the cargo to the plane
                        # ENGLISH: Add the cargo to the plane
                        effect_add = [expr("In({}, {})".format(_cargo, _plane))]
                        
                        # FORMAT: effect_rem =  [expr("Hungry(person)")]
                        # ENGLISH: Remove cargo from the airport
                        effect_rem = [expr("At({}, {})".format(_cargo, _airport))]
                        
                        # Create list of Action objects to be loaded
                        # FORMAT: eat = Action(expr("Eat(person, food)"), [precond_pos, precond_neg], [effect_add, effect_rem]) 
                        load = Action(
                            expr("Load({}, {}, {})".format(_cargo, _plane, _airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem]
                         )
                        # Append actions to loads list
                        loads.append(load)
            return loads


        #
        # FN - unload_actions
        #
        def unload_actions():
            """Create all concrete Unload actions and return a list

            FORMAL EXPRESSION:
            
            Action(Unload(c, p, a),
            	PRECOND: In(c, p) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            	EFFECT: At(c, a) ∧ ¬ In(c, p))


            :return: list of Action objects
            """
            
            # TODO create all Unload ground actions from the domain Unload action
            
            ##
            ## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
            ##
            
            unloads = []
            
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        
                        # PRECOND_POS: In(c, p) ∧ At(p, a)
                        # ENGLISH:
                        precond_pos = [expr("In({}, {})".format(cargo, plane)), expr("At({}, {})".format(plane, airport))]

                        # PRECOND_NEG: 
                        # -None-
                        precond_neg = []
                        
                        # ENGLISH: Add the cargo the the airport
                        effect_add = [expr("At({}, {})".format(cargo, airport))]
                        
                        # ENGLISH: Remove the cargo from the plane
                        effect_rem = [expr("In({}, {})".format(cargo, plane))]
                        
                        # Create list of Action objects to be UN_loaded
                        unload = Action(
                            expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem]
                        )
                        
                        # Append actions to unloads list
                        unloads.append(unload)            

            return unloads


        #
        # FN - fly
        # 
        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()


    #
    # actions
    #
    # This function returns the list of actions that *can* be executed in any given state of the planning process.
    # It strategic function is to check to see if a set of preconditions of the action in question are contained
    # the set of clauses articulated in the input state - aka, is this action actually possible.
    #
    #
    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        
        # Reminder - From get-actions:
        #
        # print("\n SELF_DICT: ", self.__dict__, "\n")
        #
        #        self_dict = 
        #        {
        #        'state_map': [
        #            At(C1, SFO), At(C2, JFK), At(P1, SFO), At(P2, JFK), At(C2, SFO), In(C2, P1), 
        #            In(C2, P2), At(C1, JFK), In(C1, P1), In(C1, P2), 
        #            At(P1, JFK), At(P2, SFO)
        #            ], 
        #        'initial_state_TF': 'TTTTFFFFFFFF', 'initial': 'TT TTFFFFFFFF', 
        #        'goal': [At(C1, JFK), At(C2, SFO)], 
        #        'cargos': ['C1', 'C2'], 
        #        'planes': ['P1', 'P2'], 
        #        'airports': ['JFK', 'SFO']
        #        }
                
        #       CARGOS:  ['C1', 'C2']
        #       PLANES:  ['P1', 'P2']
        #       AIRPORTS:  ['JFK', 'SFO']        
        
        # print("STATE: ", state)
        # print("STATE MAP: ", self.state_map)
        
        possible_actions = []
        
        ##
        ## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
        ##
        
        # PropKB() - is just a helper FN from the AIMA codebase that makes 
        # A Knowledge base for propositional logic. Inefficient, with no indexing - per logic.py
        # Get a handle to our knowledge base of logical expressions (propositional logic)
        # Methods:
        #     tell
        #     ask_generator
        #     ask_if_true
        #     retract
        
        kb = PropKB()
        
        # Add the current state's positive sentence's clauses to the propositional logic KB
        kb.tell( decode_state(state, self.state_map).pos_sentence() )
        
        # print("DECODED : ",decode_state(state, self.state_map).pos_sentence())
        # DECODED :  (At(C2, SFO) & In(C1, P1) & At(P1, JFK) & At(P2, SFO))
        
        # print("dict_kb", kb.__dict__)
        #
        # dict_kb {'clauses': [At(C1, SFO), At(C2, JFK), At(P1, SFO), At(P2, JFK)]}
        # dict_kb {'clauses': [At(C2, JFK), At(P1, SFO), At(P2, JFK), In(C1, P1)]}
        # dict_kb {'clauses': [At(C1, SFO), At(P1, SFO), At(P2, JFK), In(C2, P2)]}
        # .....
        
        possible_actions = []
        
        for action in self.actions_list:
            
            # self.actions_list = self.get_actions() : from AirCargoProblem (init)
            #
            # print("ACTION: ", action )
            #
            # ACTION:  Unload(C2, P2, JFK)
            # ACTION:  Load(C1, P2, JFK)
            # ACTION:  Fly(P1, JFK, SFO)
            # ....

            # Base Assumption: The action is possible until proven otherwise
            is_action_possible = True

            # For the negation Clauses - stop if you find one.
            for _clause in action.precond_neg:
                if _clause in kb.clauses:
                    # This action is not possible - stop searching
                    is_action_possible = False
                    break # Action not possible - stop the search

            # Only check if action is still possible
            if is_action_possible:
                for _clause in action.precond_pos:
                    # print("C: ",_clause)
                    # C:  At(P1, JFK)
                    # C:  In(C1, P1)
                    # C:  At(P1, SFO)
                    if _clause not in kb.clauses:
                        is_action_possible = False
                        break # Action not possible - stop the search

            # if the action is still possible, append it to the list.
            if is_action_possible: 
                possible_actions.append(action)
        
        return possible_actions
        
        
    #
    # FN - result
    #
    
    ##
    ## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
    ##
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given action in the given state. 
        The action must be one of self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        
        # TODO implement
        
        prev_state = decode_state(state, self.state_map)
        new_state = FluentState([], [])
        
        # print("PREV STATE: ", prev_state.__dict__ )
        # PREV STATE:  {
        # 'pos': [At(P1, SFO), In(C2, P1), In(C1, P1), At(P2, SFO)], 
        # 'neg': [At(C1, SFO), At(C2, JFK), At(P2, JFK), At(C2, SFO), In(C2, P2), At(C1, JFK), In(C1, P2), At(P1, JFK)]
        #  }
        
        # The precondition and effect of an action are each conjunctions of literals (positive or negated atomic sentences).
        # The precondition defines the states in which the action can be executed
        # The effect defines the result of executing the action. 
        # For each fluent (positive or negative) in the old state, we need to look in the action's effects 
        # (both the add and remove effects) to determine how the fluent will be carried over to the new state, 
        # as a positive or a negative fluent.
        

        # Prev POSITIVE State:
        # add positive fluents which are in the old state and should not be removed
        for fluent in prev_state.pos:
            # print("FLUENT_POS: ", fluent)
            # FLUENT_POS:  At(P2, SFO)
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
                
        # Prev NEGATIVE State:
        # add positive fluents which should be added and have not already been added
        for fluent in prev_state.neg:
            # print("FLUENT_NEG: ", fluent)
            # FLUENT_NEG:  At(C1, SFO)
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)

        # Any action may have positive and negative effects INDEPENDENT of the old state. If they're ADD effects,
        # make sure these fluents are added to the new state's "positive" sentences. If they're REMOVE effects, these 
        # fluents need to be added to the new state's "negative" sentences.

        # EFFECT - insert Add into new state POSITIVE, if not already there
        for fluent in action.effect_add:
            # print("FLUENT_ADD: ", fluent)
            # FLUENT_ADD:  At(P2, JFK)
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
                
        # EFFECT - insert Remove into new state NEGATIVE, if not already there
        for fluent in action.effect_rem:
            # print("FLUENT_REM: ", fluent)
            # FLUENT_REM:  At(P2, SFO)
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)        

        return encode_state(new_state, self.state_map)
        
        

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum


    #
    # FN - h_ignore_preconditions
    # 

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        # count = 0
        # return count
        
        # Get a handle to our knowledge base of logical expressions (propositional logic)
        kb = PropKB()
        
        # Add the current state's positive sentence's clauses to the propositional logic KB
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())
        
        kb_clauses = kb.clauses
        actions_count = 0
        
        for clause in self.goal:
            if clause not in kb_clauses:
                actions_count += 1
        
        return actions_count

#
# This is PROBLEM ONE of the AirCargoProblem
#

##
## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
##

## Problem 1 initial state and goal:
## Init(At(C1, SFO) ∧ At(C2, JFK) 
##	∧ At(P1, SFO) ∧ At(P2, JFK) 
##	∧ Cargo(C1) ∧ Cargo(C2) 
##	∧ Plane(P1) ∧ Plane(P2)
##	∧ Airport(JFK) ∧ Airport(SFO))
## Goal(At(C1, JFK) ∧ At(C2, SFO))

# FN - air_cargo_p1
#
# The function that formally defines the above formulation is below:
#
def air_cargo_p1() -> AirCargoProblem:

    ##	∧ Cargo(C1) ∧ Cargo(C2) 
    ##	∧ Plane(P1) ∧ Plane(P2)
    ##	∧ Airport(JFK) ∧ Airport(SFO))
    #
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    
    ## At(P1, SFO) ∧ At(P2, JFK)
    ## Init(At(C1, SFO) ∧ At(C2, JFK)
    #
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
           
    # The permutations of possibilities to arrive at the  goal state  
    #
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
           
    init = FluentState(pos, neg)
    
    ## Goal(At(C1, JFK) ∧ At(C2, SFO))
    #
    goal = [expr('At(C1, JFK)'), 
            expr('At(C2, SFO)'),
            ]
            
    return AirCargoProblem(cargos, planes, airports, init, goal)


#
# This is PROBLEM TWO of the AirCargoProblem
#

##
## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
##

## Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) 
##	∧ At(P1, SFO) ∧ At(P2, JFK) ∧ At(P3, ATL) 
##	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
##	∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
##	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))
## Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))

def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    
    ##	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
    ##	∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
    ##	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))
    #
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    
    ## At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) 
    ##	∧ At(P1, SFO) ∧ At(P2, JFK) ∧ At(P3, ATL) 
    pos = [
        expr('At(C1, SFO)'),
        expr('At(C2, JFK)'),
        expr('At(C3, ATL)'),
        expr('At(P1, SFO)'),
        expr('At(P2, JFK)'),
        expr('At(P3, ATL)')
    ]
    
    # The permutations of possibilities to arrive at the  goal state
    #
    neg = [
        expr('At(C1, JFK)'),
        expr('At(C1, ATL)'),
        expr('In(C1, P1)'),
        expr('In(C1, P2)'),
        expr('In(C1, P3)'),
        expr('At(C2, SFO)'),
        expr('At(C2, ATL)'),
        expr('In(C2, P1)'),
        expr('In(C2, P2)'),
        expr('In(C2, P3)'),
        expr('At(C3, SFO)'),
        expr('At(C3, JFK)'),
        expr('In(C3, P1)'),
        expr('In(C3, P2)'),
        expr('In(C3, P3)'),
        expr('At(P1, JFK)'),
        expr('At(P1, ATL)'),
        expr('At(P2, SFO)'),
        expr('At(P2, ATL)'),
        expr('At(P3, JFK)'),
        expr('At(P3, SFO)')
    ]
    
    init = FluentState(pos, neg)
    
    ## Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))
    #
    goal = [
        expr('At(C1, JFK)'),
        expr('At(C2, SFO)'),
        expr('At(C3, SFO)')
    ]
    
    return AirCargoProblem(cargos, planes, airports, init, goal)



#
# This is PROBLEM THREE of the AirCargoProblem
#

##
## Inspiration: https://medium.com/towards-data-science/ai-planning-historical-developments-edcd9f24c991
##

## Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD) 
##	∧ At(P1, SFO) ∧ At(P2, JFK) 
##	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
##	∧ Plane(P1) ∧ Plane(P2)
##	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
## Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))


def air_cargo_p3() -> AirCargoProblem:
    
    # TODO implement Problem 3 definition
    
    ##	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
    ##	∧ Plane(P1) ∧ Plane(P2)
    ##	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    
    # At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD) 
    ##	∧ At(P1, SFO) ∧ At(P2, JFK) 
    pos = [
        expr('At(C1, SFO)'),
        expr('At(C2, JFK)'),
        expr('At(C3, ATL)'),
        expr('At(C4, ORD)'),
        expr('At(P1, SFO)'),
        expr('At(P2, JFK)')
    ]
    
    # The permutations of possibilities to arrive at the  goal state
    #
    neg = [
        expr('At(C1, JFK)'),
        expr('At(C1, ATL)'),
        expr('At(C1, ORD)'),
        expr('In(C1, P1)'),
        expr('In(C1, P2)'),
        expr('At(C2, SFO)'),
        expr('At(C2, ATL)'),
        expr('At(C2, ORD)'),
        expr('In(C2, P1)'),
        expr('In(C2, P2)'),
        expr('At(C3, SFO)'),
        expr('At(C3, JFK)'),
        expr('At(C3, ORD)'),
        expr('In(C3, P1)'),
        expr('In(C3, P2)'),
        expr('At(C4, SFO)'),
        expr('At(C4, JFK)'),
        expr('At(C4, ATL)'),
        expr('In(C4, P1)'),
        expr('In(C4, P2)'),
        expr('At(P1, JFK)'),
        expr('At(P1, ATL)'),
        expr('At(P1, ORD)'),
        expr('At(P2, SFO)'),
        expr('At(P2, ATL)'),
        expr('At(P2, ORD)')
    ]
    
    init = FluentState(pos, neg)
    
    ## Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    #
    goal = [
        expr('At(C1, JFK)'),
        expr('At(C3, JFK)'),
        expr('At(C2, SFO)'),
        expr('At(C4, SFO)')
    ]
    
    return AirCargoProblem(cargos, planes, airports, init, goal)






