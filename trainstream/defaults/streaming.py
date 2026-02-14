

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