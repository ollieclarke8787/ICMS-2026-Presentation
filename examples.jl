using AbstractAlgebra
include("PuiseuxRegression.jl")
using .PuiseuxRegression

# Example ring
Kt, (t,) = puiseux_polynomial_ring(QQ, ["t"])

f = t^(2//3) + 4*t^(1//3)
println("val(f) = ", valuation(f))

g = t^(5//3) - t^(4//3);
println("f + g = ", f + g)
println("f * g = ", f * g)

h = 1 - t - t^2;
invf = inv_puiseux(h);
println("inv approx = ", invf)



# Tiny dataset (line y = c0 + c1*x + noise)
c0 = Kt(2) + t^(1//2)
c1 = Kt(3) - t^(1//3)

x1 = Kt(1) + t
x2 = Kt(2) + t^(2//3)
x3 = Kt(3) + t^(1//4)

y1 = c0 + c1*x1 + t^5
y2 = c0 + c1*x2 + t^4
y3 = c0 + c1*x3 + t^6

D = [
    [x1, y1],
    [x2, y2],
    [x3, y3],
]

# line through points 1 & 2 and 2 & 3
[c0, c1]
coef12 = dataPointsToFunction(D[[1,2]], truncNTerms=5)
coef23 = dataPointsToFunction(D[[2,3]], truncNTerms=5)

println("Interpolated coefficients: ", coef12)

println("loss(D, {1,2}) = ", loss(D, [1,2], Verbose=true))
println("loss(D, {1,3}) = ", loss(D, [1,3], Verbose=false))

println("lossVector(D, {1,2}) = ", lossVector(D, [1,2], Verbose=false))
println("lossVector(D, {1,3}) = ", lossVector(D, [1,3], Verbose=false))


## testing PuiseuxRegression.jl

# Puiseux context (univariate)
ctx = PContext(QQ, ["t"])
Kt = ctx.Kt
t = ctx.vars[1]

# Generate true coefficients for affine model y = c0 + c1*x (so d = 1)
c = trueCoeffs(ctx, 1; L=3)
println("True coefficients:")
println(c)

# Generate data points
D = dataSet(c, 10; L=5, minShiftVal=3//1, maxShiftVal=10//1)

println("\nFirst 3 generated points [x, y]:")
for i in 1:3
    println("D[$i] = ", D[i])
end

# Fit using a subset S of size d+1 = 3
S = [1,2,3]
coef = dataPointsToFunction(D[S]; truncOutput=true, truncNTerms=20)

println("\nInterpolated coefficients from points S=$S:")
println(coef)

# Evaluate loss and loss vector
ℓ = loss(D, S; Verbose=false)
ℓvec = lossVector(D, S; Verbose=false, truncOutput=true, truncNTerms=20)

println("\nloss(D,S) = ", ℓ)
println("lossVector(D,S) = ", ℓvec)
