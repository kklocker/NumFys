from eqsolver import main_routine
from dask import delayed, compute
from time import time

if __name__ == "__main__":
    g_min = 5
    g_max = 5
    p_min = 1
    p_max = 3
    config = [
        tuple((g, pts))
        for g in range(g_min, g_max + 1)
        for pts in range(p_min, p_max + 1)
    ]
    print(
        f"Starting calculating {len(config)} solutions for recursion"
        f"levels {g_min} - {g_max} and {p_min} - {p_max} points between"
    )
    start_time = time()
    # solu = []
    for cfg in config:
        s = time()
        # sol = delayed(main_routine)(cfg)
        # solu.append(sol)
        main_routine(cfg, k=15)
        st = time()
        print(f"Config {cfg} took {st-s:.2f}s to finish. Elapsed: {st-start_time:.2f}s")
    # results = compute(*solu)

    end = time()

    print(
        f"Finished solving systems: {config}.\n\n Total time spent: {end-start_time:.2f}s"
    )
