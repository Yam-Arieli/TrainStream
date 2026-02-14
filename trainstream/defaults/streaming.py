__all__ = ["generator_from_args", "from_list", "from_csv"]

from typing import Callable, Iterable, Any, Generator

def generator_from_args(loader_func: Callable[[Any], Any],
                        args_iterable: Iterable[Any]) \
                            -> Callable[[], Generator[Any, None, None]]:
    """
    Creates a stream factory that generates batches by mapping an iterable of arguments 
    to a loading function.

    Args:
        loader_func: A function that takes one argument (e.g., a batch index or mask) 
                     and returns a data batch.
        args_iterable: An iterable (list, generator, range) where each item 
                       is passed to loader_func.

    Returns:
        A factory function that returns a fresh generator when called.
    """
    # Note: If args_iterable is a one-time generator (like (x for x in range(10))), 
    # it will be exhausted after the first epoch. 
    # For multi-epoch safety, args_iterable should be a reusable collection (list/range) 
    # or re-created inside. 
    
    def factory():
        for arg in args_iterable:
            yield loader_func(arg)
            
    return factory

def from_list(full_data, batch_size):
    """
    Factory that returns a generator over a list.
    Useful for simulating streams from static data.
    """
    def generator():
        for i in range(0, len(full_data), batch_size):
            yield full_data[i : i + batch_size]
    return generator

def from_csv(filepath, batch_size, parser_fn=None):
    """
    Factory that streams lines from a CSV file.
    """
    import csv
    
    def generator():
        batch = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            for row in reader:
                item = parser_fn(row) if parser_fn else row
                batch.append(item)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            if batch: # Yield remainder
                yield batch
                
    return generator