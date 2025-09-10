# AutoCollector

A Rust garbage collector library that provides automatic memory management through tracing.

## Features

- **Automatic Memory Management**: Handles allocation and deallocation of objects with cyclic references
- **Tracing Garbage Collection**: Uses mark-and-sweep algorithm to identify reachable objects
- **Thread Safety**: Provides `AtomicAutoPtr` for concurrent access to managed objects
- **Background Collection**: Optional background thread for automatic garbage collection
- **Manual Control**: Ability to trigger garbage collection manually when needed

## Basic Usage

```rust
use rust_gc::{AutoCollector, AutoCollectorConfig, AutoPtr, Traceable};

let collector = AutoCollector::new(AutoCollectorConfig::default());

// Allocate a new object, automatically added as a root
let ptr = collector.alloc(MyObject::new());

// Objects implement `Traceable` to reference each other
impl Traceable for MyObject {
    fn trace(&self, tracer: &mut Tracer) {
        if let Some(ptr) = self.other_object.load(Ordering::SeqCst) {
            tracer.edge(&ptr);
        }
    }
}
```

## Key Types

- `AutoPtr<T>`: Smart pointer for garbage-collected objects
- `AtomicAutoPtr<T>`: Thread-safe version of AutoPtr
- `AutoCollector`: The main garbage collector instance
- `Traceable`: Trait for objects that can be traced by the collector

## Configuration

The collector can be configured with custom settings:

```rust
let config = AutoCollectorConfig::default()
    .with_steps_per_increment(100)
    .with_background_collector_interval(Duration::from_millis(10));
```
