right_of(X, Y) :- X is Y+1.
left_of(X, Y) :- right_of(Y, X).

next_to(X, Y) :- right_of(X, Y).
next_to(X, Y) :- left_of(X, Y).

solution(Street, FishOwner) :-
    Street = [
           house(1, Nationality1,  Color1,     Pet1,   Drinks1,    Smokes1),
           house(2, Nationality2,  Color2,     Pet2,   Drinks2,    Smokes2),
           house(3, Nationality3,  Color3,     Pet3,   Drinks3,    Smokes3),
           house(4, Nationality4,  Color4,     Pet4,   Drinks4,    Smokes4),
           house(5, Nationality5,  Color5,     Pet5,   Drinks5,    Smokes5)],
    member(house(_, brit,          red,        _,      _,          _           ), Street),
    member(house(_, swede,         _,          dog,    _,          _           ), Street),
    member(house(_, dane,          _,          _,      tea,        _           ), Street),
    member(house(A, _,             green,      _,      _,          _           ), Street),
    member(house(B, _,             white,      _,      _,          _           ), Street),
    left_of(A, B),
    member(house(_, _,             green,      _,      coffee,     _           ), Street),
    member(house(_, _,             _,          birds,  _,          pall_mall   ), Street),
    member(house(_, _,             yellow,     _,      _,          dunhill     ), Street),
    member(house(3, _,             _,          _,      milk,       _           ), Street),
    member(house(1, norweigan,     _,          _,      _,          _           ), Street),
    member(house(C, _,             _,          _,      _,          blend       ), Street),
    member(house(D, _,             _,          cats,   _,          _           ), Street),
    next_to(C, D),
    member(house(E, _,             _,          horse,  _,          _           ), Street),
    member(house(F, _,             _,          _,      _,          dunhill     ), Street),
    next_to(E, F),
    member(house(_, _,             _,          _,      bluemaster, beer        ), Street),
    member(house(_, german,        _,          _,      _,          prince      ), Street),
    member(house(G, norweigan,     _,          _,      _,          _           ), Street),
    member(house(H, _,             blue,       _,      _,          _           ), Street),
    next_to(G, H),
    member(house(I, _,             _,          _,      _,          blend       ), Street),
    member(house(J, _,             _,          _,      water,      _           ), Street),
    next_to(I, J),
    member(house(_, FishOwner,     _,          fish,   _,          _           ), Street).