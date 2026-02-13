import threading
import queue
from typing import TypeVar, Generic, Iterable, Callable, Any, Optional

# T represents the data type of the stream batches (e.g., List, np.ndarray, torch.Tensor)
T = TypeVar('T')

class StreamLoop(Generic[T]):
    """
    TrainStream: A framework-agnostic engine for training on streaming data 
    while maintaining a representative Coreset.

    The engine handles the concurrent loading of data, merging with the memory buffer,
    and executing the user-defined training and selection logic.
    """
    def __init__(self, 
                 # Infrastructure
                 stream_factory: Callable[[], Iterable[T]], 
                 buffer_size_m: int, 
                 
                 # Logic Hooks (REQUIRED)
                 merge_fn: Callable[[T, T], T],
                 train_fn: Callable[..., Any], 
                 select_fn: Callable[..., T],
                 
                 # Optional Settings
                 prefetch: int = 1,
                 on_step_end: Optional[Callable[..., None]] = None,
                 
                 # User Context
                 *args, **kwargs):
        """
        Configures the TrainStream pipeline.
        
        Args:
            stream_factory: A function that returns the stream iterator when called.
            buffer_size_m: Maximum size of the coreset to keep (m).
            merge_fn: Function to combine the buffer (m) and the new batch (k).
            train_fn: User function to train the model. 
                      Signature: (data: T, *args, **kwargs) -> train_info: Any
            select_fn: User function to select the best 'm' items to keep. 
                       Signature: (data: T, m: int, *args, train_info=..., **kwargs) -> T
            prefetch: Number of batches to load in the background (default 1).
            on_step_end: Optional callback running after every step.
            *args, **kwargs: Extra arguments passed directly to your train/select functions 
                             (e.g., model, optimizer, device).
        """
        self.stream_factory = stream_factory
        self.m = buffer_size_m
        self.prefetch = prefetch
        self.buffer: Optional[T] = None
        
        self.merge_fn = merge_fn
        self.train_fn = train_fn
        self.select_fn = select_fn
        
        self.on_step_end = on_step_end
        
        self.user_args = args
        self.user_kwargs = kwargs

    def _background_loader(self, generator, q, stop_event):
        """Runs in a separate thread to feed the queue from the generator."""
        try:
            for batch in generator:
                if stop_event.is_set(): break
                q.put(batch)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None) # Signal End of Stream

    def run(self) -> Optional[T]:
        """
        Executes the streaming pipeline until the stream is exhausted.
        
        Returns:
            The final Coreset buffer (type T) containing the best 'm' samples.
        """
        print(f"TrainStream started. Buffer Size: {self.m}")
        
        q = queue.Queue(maxsize=self.prefetch)
        stop_event = threading.Event()
        
        # Start the single pass
        current_stream = self.stream_factory() 
        loader_thread = threading.Thread(
            target=self._background_loader,
            args=(current_stream, q, stop_event),
            daemon=True
        )
        loader_thread.start()
        
        step = 0
        try:
            while True:
                # 1. Get k (parallel load)
                item = q.get()
                
                # Error/Exit Handling
                if isinstance(item, Exception): raise item
                if item is None: break 
                
                k_batch = item

                # 2. Merge (m + k)
                if self.buffer is None:
                    train_set = k_batch
                else:
                    train_set = self.merge_fn(self.buffer, k_batch)
                
                # 3. Short Train (User defines internal epochs here)
                # Passes: data, *args, **kwargs
                train_info = self.train_fn(train_set, *self.user_args, **self.user_kwargs)
                
                # 4. Select (Coreset Update)
                # Passes: data, m, *args, train_info=..., **kwargs
                self.buffer = self.select_fn(
                    train_set, 
                    self.m, 
                    *self.user_args,
                    train_info=train_info, 
                    **self.user_kwargs
                )
                
                step += 1
                
                if self.on_step_end:
                    self.on_step_end(step, self.m, train_info)
                    
        except KeyboardInterrupt:
            stop_event.set()
            print("\nStream stopped by user.")
            return self.buffer
        
        print(f"TrainStream Finished. Final Buffer Size: {len(self.buffer) if self.buffer else 0}")
        
        return self.buffer