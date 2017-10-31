% Constraints satisfaction problem from Albert Einstein's Puzzle 
% Author : Feng Di @K.U.Leuven  
  
% Please run this program by ECLiPSe  
% ECLiPSe is a constraint processing tool using Prolog  
  
:-lib(ic).  
  
einstein_problem:-  
    % I use 5 list of 5 variables  
    Nat = [English,Swedes,Danish,Norwegian,German],  
    Color = [Red,White,Green,Yellow,Blue],  
    Pat = [Dog,Bird,Cat,Horse,Fish],  
    Drink = [Tea,Coffee,Milk,Beer,Water],  
    Cigarette = [PallMall,Dunhill,Blends,BlueMaster,Prince],  
  
    %domain of variables  
    Nat :: 1..5,  
    Color :: 1..5,  
    Pat :: 1..5,  
    Drink :: 1..5,  
    Cigarette :: 1..5,  
      
    %constraints  
    alldifferent(Nat),  
    alldifferent(Color),  
    alldifferent(Pat),  
    alldifferent(Drink),  
    alldifferent(Cigarette),  
  
    English $= Red,  
    Swedes $= Dog,  
    Danish $= Tea,  
    White #= Green+1,  
    Green $= Coffee,  
    PallMall $= Bird,  
    Yellow $= Dunhill,  
    Milk $= 3,  
    Norwegian $= 1,  
    abs(Blends-Cat) $= 1,  
    abs(Horse-Dunhill) $= 1,  
    BlueMaster $= Beer,  
    German $= Prince,  
    abs(Norwegian-Blue) $= 1,  
    abs(Blends-Water) $=1,  
  
    %search  
    flatten([Nat,Color,Pat,Drink,Cigarette],List),  
    labeling(List),  
      
    %print solution  
    NatNames = [English-english,Swedes-swedes,Danish-danish,Norwegian-norwegian,German-german],  
    memberchk(Fish-FishNat,NatNames),  
    write('The '),write(FishNat),write(' owns fish.'),nl.