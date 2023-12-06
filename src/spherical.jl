# Area of the two-sided spherical cap, i.e., going from abscissa -1 to s and
# from s to 1; assuming the total area (i.e., when s=0) is 1.
function area_2sided_cap(s::Real, n::Integer)
    D = Beta((n - 1)/2, 1/2)
    return cdf(D, 1 - s^2)
end

# Abscissa of the two-sided spherical cap, i.e., going from abscissa -1 to s and
# from s to 1 such that the area is A; assuming the total area is 1.
function area_2sided_cap_inv(A::Real, n::Integer)
    A = max(0, min(1, A))
    D = Beta((n - 1)/2, 1/2)
    return sqrt(1 - quantile(D, A))
end