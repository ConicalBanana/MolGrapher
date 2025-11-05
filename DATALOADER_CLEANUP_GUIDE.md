# DataLoader Cleanup Guide

## Problem
When using PyTorch DataLoader with `persistent_workers=True` and multiple
workers (`num_workers > 0`), file handlers can remain open even after the
DataLoader is no longer needed. This can lead to:
- Resource leaks (unclosed file descriptors)
- Memory leaks
- "Too many open files" errors on long-running processes
- Orphaned worker processes

## Solution Implemented

### 1. Conditional Persistent Workers
Modified `get_dataloader()` method to only enable `persistent_workers` during
training:

```python
def get_dataloader(self, dataset, is_training=False):
    # Only use persistent workers for training to avoid file handler leaks
    use_persistent_workers = (
        is_training and self.config["nb_workers"] > 0
    )
    
    # ... DataLoader creation with persistent_workers=use_persistent_workers
```

**Why this works:**
- Training benefits from persistent workers (faster epoch transitions)
- Validation/prediction don't need persistent workers (run once)
- Disabling persistent workers ensures proper cleanup after each run

### 2. Added teardown() Method
Implemented PyTorch Lightning's `teardown()` hook to properly clean up
resources:

```python
def teardown(self, stage=None):
    """Clean up resources when module is torn down."""
    # Close any open file handlers from caption remover
    if hasattr(self, 'caption_remover'):
        del self.caption_remover

    # Clear dataset references to free memory and close file handlers
    self.train_dataset = None
    self.val_dataset = None
    self.benchmarks_datasets = None

    # Clear real datasets if they exist
    if hasattr(self, 'train_dataset_real'):
        self.train_dataset_real = None
    if hasattr(self, 'val_dataset_real'):
        self.val_dataset_real = None
```

**Why this works:**
- PyTorch Lightning automatically calls `teardown()` when done
- Explicitly clearing references helps garbage collection
- Ensures file handlers are released promptly

## Best Practices for DataLoader

### 1. Choose the Right Worker Strategy

**Use persistent_workers=True when:**
- Training over multiple epochs
- You need fast epoch transitions
- You have enough system resources

**Use persistent_workers=False when:**
- Running prediction/inference once
- Memory is constrained
- You want guaranteed cleanup

### 2. Properly Configure Workers

```python
# Good configuration for training
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
)

# Good configuration for inference
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=False,  # Ensures cleanup
    prefetch_factor=2,
    pin_memory=True,
)

# Good configuration for debugging
DataLoader(
    dataset,
    num_workers=0,  # No multiprocessing, no file handler issues
    persistent_workers=False,
)
```

### 3. Always Implement teardown()

In your PyTorch Lightning DataModule:

```python
def teardown(self, stage=None):
    """Clean up resources."""
    # Clear any custom resources
    if hasattr(self, 'custom_resource'):
        self.custom_resource.close()
        del self.custom_resource
    
    # Clear dataset references
    self.train_dataset = None
    self.val_dataset = None
    self.test_dataset = None
```

### 4. Handle PIL Images Properly

Always close PIL images explicitly:

```python
# Good
image = Image.open(path)
try:
    # ... process image ...
finally:
    image.close()

# Better - using context manager
with Image.open(path) as image:
    # ... process image ...
    # Automatically closed
```

### 5. Monitor Resource Usage

Use these commands to check for resource leaks:

```bash
# Check open file descriptors
lsof -p <pid> | wc -l

# Check for zombie processes
ps aux | grep -i defunct

# Monitor memory usage
watch -n 1 'ps aux | grep python'
```

## Changes Made to data_module.py

1. **Line 1148, 1151**: Added `is_training=True` parameter to
   `get_dataloader()` calls in `train_dataloader()`

2. **Line 1231-1260**: Modified `get_dataloader()` to accept `is_training`
   parameter and conditionally set `persistent_workers`

3. **Line 1141-1165**: Added `teardown()` method for proper resource cleanup

4. **Line 1238-1239**: Fixed linting error by changing `!= None` to
   `is not None`

## Testing the Changes

To verify the fix works:

1. **Check file descriptors before and after:**
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Open files: {len(process.open_files())}")
   ```

2. **Run a prediction loop:**
   ```python
   for i in range(10):
       trainer.predict(model, datamodule)
       # Check that file descriptors don't keep growing
   ```

3. **Monitor with system tools:**
   ```bash
   # While script is running
   watch -n 1 'lsof -p $(pgrep -f your_script.py) | wc -l'
   ```

## Additional Recommendations

1. **Use context managers when possible:**
   ```python
   with torch.no_grad():
       predictions = trainer.predict(model, datamodule)
   ```

2. **Explicitly delete large objects:**
   ```python
   results = trainer.fit(model, datamodule)
   del results  # Help garbage collector
   ```

3. **Consider using torch.multiprocessing:**
   ```python
   import torch.multiprocessing as mp
   mp.set_sharing_strategy('file_system')  # Or 'file_descriptor'
   ```

4. **Set appropriate ulimit:**
   ```bash
   # Increase file descriptor limit
   ulimit -n 4096
   ```

## References

- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [PyTorch Lightning DataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)
- [Python Resource Management](https://docs.python.org/3/library/contextlib.html)

