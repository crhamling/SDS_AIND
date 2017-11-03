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
        Returns:
        ----------
        list<Action>
            list of Action objects
        """
        loads = []
        unloads = []
        flys = []

        #CARGO ACTIONS
        for c in self.cargos:
            for p in self.planes:
                for a in self.airports:
    
                    #PRECONDITIONS
                    load_precond_pos = [expr("At({}, {})".format(c, a)), expr("At({}, {})".format(p, a))]
                    load_precond_neg = []
                    
                    unload_precond_pos = [expr("In({}, {})".format(c, p)), expr("At({}, {})".format(p, a))]
                    unload_precond_neg = []
                    
                    #EFFECTS
                    load_effect_add = [expr("In({}, {})".format(c, p))]
                    load_effect_rem = [expr("At({}, {})".format(c, a))]
                    
                    unload_effect_add = [expr("At({}, {})".format(c, a))]
                    unload_effect_rem = [expr("In({}, {})".format(c, p))]
                    
                    #ACTIONS
                    load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                        [load_precond_pos, load_precond_neg],
                        [load_effect_add, load_effect_rem])

                    unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                        [unload_precond_pos, unload_precond_neg],
                        [unload_effect_add, unload_effect_rem])
                    
                    #ADD
                    loads.append(load)
                    unloads.append(unload)

        #FLY ACTIONS
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
                        
        return loads + unloads + flys

    def actions(self, state: str) -> list:
        """
        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        possible_actions = []
        
        kb = PropKB()
        kb.tell( decode_state(state, self.state_map).pos_sentence())
        
        for action in self.actions_list:
            action_possible = True
            
            #CHECK NEGATIVE PRECONDITIONS
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    action_possible = False
            
            #CHECK POSITIVE PRECIONDITIONS
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    action_possible = False
                    
            if action_possible: 
                possible_actions.append(action)
        
        return possible_actions

    def result(self, state: str, action: Action):
        """
        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        old_state = decode_state(state, self.state_map)
        new_state = FluentState([], [])
        
        #ADD PASSED ACTION EFFECTS TO NEW STATE
        for ef in action.effect_add:
            if ef not in new_state.pos:
                new_state.pos.append(ef)
        for ef in action.effect_rem:
            if ef not in new_state.neg:
                new_state.neg.append(ef) 

        #PRESERVE OLD STATE STATUS IF UNALTERED BY PASSED ACTION
        for ef in old_state.neg:
            if ef not in action.effect_add:
                new_state.neg.append(ef)
        for ef in old_state.pos:
            if ef not in action.effect_rem:
                new_state.pos.append(ef)
        
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
        return 1

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        #Russell Norvig Ed. 3 - 10.2.3
        
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())
        
        n = 0
        for c in self.goal:
            if c not in kb.clauses:
                n += 1
        return n


def air_cargo_p1() -> AirCargoProblem:
    '''
    Init(At(C1, SFO) ∧ At(C2, JFK) 
	∧ At(P1, SFO) ∧ At(P2, JFK) 
	∧ Cargo(C1) ∧ Cargo(C2) 
	∧ Plane(P1) ∧ Plane(P2)
	∧ Airport(JFK) ∧ Airport(SFO))
    Goal(At(C1, JFK) ∧ At(C2, SFO))
    '''
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
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
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    '''
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) 
    ∧ At(P1, SFO) ∧ At(P2, JFK) ∧ At(P3, ATL) 
    ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
    ∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
    ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))
    Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))
    '''
    
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)')
           ]

    neg = [expr('At(C1, JFK)'),
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
    
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)')
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    '''
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD) 
    ∧ At(P1, SFO) ∧ At(P2, JFK) 
    ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
    ∧ Plane(P1) ∧ Plane(P2)
    ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
    Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    '''
    
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')
           ]
    
    neg = [expr('At(C1, JFK)'),
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
    
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)')
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
