
using Oscar

P = polyhedron([[1,0], [0,1]], [1,1])
vertices(P)
dim(P)
rays(P)

I = grassmann_pluecker_ideal(3, 6)
degree(I)
