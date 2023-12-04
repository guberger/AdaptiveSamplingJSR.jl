x = 1 / π

for d = 2:100
    display(x ^(-1/(d-1)))
    global x = 1 / (2 * π * d * x)
end