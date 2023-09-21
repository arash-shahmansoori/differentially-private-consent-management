from time import perf_counter


def custom_timer(original_func):
    def wrapper(*args, **kwargs):

        # Start running time
        t_start = perf_counter()

        original_func(*args, **kwargs)

        # Compute the elapsed time
        t_stop = perf_counter()
        td = t_stop - t_start

        return td

    return wrapper


def custom_timer_with_return(original_func):
    def wrapper(*args, **kwargs):

        # Start running time
        t_start = perf_counter()

        out = original_func(*args, **kwargs)

        # Compute the elapsed time
        t_stop = perf_counter()
        td = t_stop - t_start

        return td, out

    return wrapper