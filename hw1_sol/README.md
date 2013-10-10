def legit_allocations(incr, width):
    import itertools as it
    all_possible = it.product(xrange(0, 101, incr), repeat=width)
    for allocations in all_possible:
        if (sum(allocations) == 100):
            alloc = map(lambda x: float(x)/100, allocations)
            yield list(alloc)


for lf_alloc in legit_allocations(10, len(ls_symbols)):
    result = simulate(dt_start, dt_end, ls_symbols, lf_alloc)
    (f_sharpe, f_stdev, f_retavg, f_rettot) = result
    ...

