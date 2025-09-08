#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::{with_exposed_provenance, with_exposed_provenance_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::{Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Gray,
    Black,
}

pub trait Traceable: Send + Sync {
    fn trace(&self, tracer: &mut Tracer);
}

impl<'a> Tracer<'a> {
    pub fn edge(&mut self, ptr: *const ()) {
        let ptr_addr = ptr as usize;
        if let Some(edges) = &mut self.edges {
            edges.push(ptr_addr);
        }
    }
}

pub struct Tracer<'a> {
    edges: Option<&'a mut Vec<usize>>,
}

pub struct AutoPtr<T> {
    ptr: *mut T,
    collector: *const AutoCollector,
}

impl<T> Copy for AutoPtr<T> {}
impl<T> Clone for AutoPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AutoPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T> AutoPtr<T> {
    pub fn as_ptr(self) -> *mut T {
        self.ptr
    }

    fn verify_deref_preconditions(self) -> Option<Self> {
        let collector = unsafe { &*self.collector };
        let ptr_addr = self.ptr as *const () as usize;

        collector
            .allocations
            .read()
            .ok()
            .and_then(|allocations| allocations.get(&ptr_addr).map(|_| self))
    }
}

pub struct AtomicAutoPtr<T> {
    inner: AtomicUsize,
    collector: usize,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for AtomicAutoPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = self.inner.load(Ordering::Relaxed);
        write!(f, "GcAtomic({:#x})", addr)
    }
}

impl<T> AtomicAutoPtr<T> {
    pub fn new(collector: &AutoCollector) -> Self {
        Self {
            inner: AtomicUsize::new(0),
            collector: collector as *const AutoCollector as usize,
            _marker: PhantomData,
        }
    }

    pub fn load(&self, order: Ordering) -> Option<AutoPtr<T>> {
        let raw = self.inner.load(order);
        if raw == 0 {
            None
        } else {
            Some(AutoPtr {
                ptr: with_exposed_provenance_mut(raw),
                collector: with_exposed_provenance(self.collector),
            })
        }
    }

    pub fn store(&self, value: Option<AutoPtr<T>>, order: Ordering) {
        let raw = value.map(|g| g.as_ptr() as usize).unwrap_or(0);
        self.inner.store(raw, order);
    }

    pub fn clear(&self, order: Ordering) {
        self.inner.store(0, order);
    }
}

impl<T> Deref for AutoPtr<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        let this = self.verify_deref_preconditions().expect(
            "internal preconditions do not hold: allocation has been freed before dereferencing",
        );
        unsafe { &*this.ptr }
    }
}

impl AutoCollector {
    pub fn alloc_gc<T: Traceable + 'static>(&self, value: T) -> AutoPtr<T> {
        AutoPtr {
            ptr: self.alloc(value),
            collector: self as *const _,
        }
    }

    pub fn add_root_gc<T>(&self, handle: AutoPtr<T>) -> AutoPtr<T> {
        self.add_root(handle.ptr);
        handle
    }

    pub fn remove_root_gc<T>(&self, handle: AutoPtr<T>) -> AutoPtr<T> {
        self.remove_root(handle.ptr);
        handle
    }
}

impl<'a> Tracer<'a> {
    fn new_edge_collector(edges: &'a mut Vec<usize>) -> Self {
        Self { edges: Some(edges) }
    }
}

struct TraceableObject {
    color: Color,
    trace_fn: fn(*const (), &mut Tracer),
    destructor: Box<dyn Fn() + Send + Sync>,
}

pub struct AutoCollector {
    collecting: AtomicBool,
    running: AtomicBool,
    background_collector_interval: i32,
    steps_per_increment: usize,
    roots: RwLock<HashMap<usize, ()>>,
    allocations: RwLock<HashMap<usize, TraceableObject>>,
    gray_objects: Mutex<VecDeque<usize>>,
    background_collector_thread: Mutex<Option<JoinHandle<()>>>,
    background_cv: Condvar,
    background_mutex: Mutex<()>,
}

pub struct AutoCollectorConfig {
    pub steps_per_increment: usize,
    pub background_collector_interval: i32,
}

impl Default for AutoCollectorConfig {
    fn default() -> Self {
        Self {
            steps_per_increment: 100,
            background_collector_interval: 100,
        }
    }
}

impl AutoCollectorConfig {
    fn with_steps_per_increment(self, steps_per_increment: usize) -> Self {
        Self {
            steps_per_increment,
            ..self
        }
    }

    fn with_background_collector_interval(self, background_collector_interval: i32) -> Self {
        Self {
            background_collector_interval,
            ..self
        }
    }
}

impl AutoCollector {
    pub fn new(config: AutoCollectorConfig) -> Self {
        Self {
            collecting: AtomicBool::new(false),
            running: AtomicBool::new(false),
            background_collector_interval: config.background_collector_interval,
            steps_per_increment: config.steps_per_increment,
            roots: RwLock::new(HashMap::new()),
            allocations: RwLock::new(HashMap::new()),
            gray_objects: Mutex::new(VecDeque::new()),
            background_collector_thread: Mutex::new(None),
            background_cv: Condvar::new(),
            background_mutex: Mutex::new(()),
        }
    }

    pub fn create<T: Traceable + 'static>(&self, value: T) -> *mut T {
        let boxed_value = Box::new(value);
        let ptr = Box::into_raw(boxed_value);
        let ptr_addr = ptr as *const () as usize;

        let destructor = {
            let ptr_copy = ptr as usize;
            Box::new(move || unsafe {
                let typed_ptr = with_exposed_provenance_mut::<T>(ptr_copy);
                let _ = Box::from_raw(typed_ptr);
            }) as Box<dyn Fn() + Send + Sync>
        };

        let trace_fn = |ptr: *const (), tracer: &mut Tracer| {
            let typed_ptr = ptr as *const T;
            unsafe { (*typed_ptr).trace(tracer) }
        };

        let traceable_obj = TraceableObject {
            color: Color::White,
            trace_fn,
            destructor,
        };

        if let Ok(mut allocations) = self.allocations.write() {
            allocations.insert(ptr_addr, traceable_obj);
        }

        self.add_root_raw(ptr as *const () as usize);
        ptr
    }

    pub fn alloc<T: Traceable + 'static>(&self, value: T) -> *mut T {
        self.create(value)
    }

    pub fn remove_root<T>(&self, ptr: *mut T) -> *mut T {
        self.remove_root_raw(ptr as *const () as usize);
        ptr
    }

    pub fn remove_root_raw(&self, ptr_addr: usize) {
        if let Ok(mut roots) = self.roots.write() {
            roots.remove(&ptr_addr);
        }
    }

    pub fn add_root<T>(&self, ptr: *mut T) -> *mut T {
        self.add_root_raw(ptr as *const () as usize);
        ptr
    }

    pub fn add_root_raw(&self, ptr_addr: usize) {
        if let Ok(mut roots) = self.roots.write() {
            roots.insert(ptr_addr, ());
        }
    }

    fn reset_marks(&self) {
        if let Ok(mut allocations) = self.allocations.write() {
            allocations
                .values_mut()
                .for_each(|obj| obj.color = Color::White);
        }
    }

    fn mark_roots(&self) {
        if let (Ok(mut gray_objects), Ok(roots), Ok(mut allocations)) = (
            self.gray_objects.lock(),
            self.roots.read(),
            self.allocations.write(),
        ) {
            gray_objects.clear();

            for root_addr in roots.keys() {
                if let Some(traceable_obj) = allocations.get_mut(root_addr) {
                    traceable_obj.color = Color::Gray;
                    gray_objects.push_back(*root_addr);
                }
            }
        }
    }

    fn start_marking(&self) {
        self.collecting.store(true, Ordering::SeqCst);
        self.reset_marks();
        self.mark_roots();
    }

    fn step(&self) {
        if !self.collecting.load(Ordering::SeqCst) {
            return;
        }

        let objects_to_trace = {
            let Ok(mut gray_objects) = self.gray_objects.lock() else {
                return;
            };

            (0..self.steps_per_increment)
                .map_while(|_| gray_objects.pop_front())
                .collect::<Vec<_>>()
        };

        for current_addr in objects_to_trace {
            let Ok(mut allocations) = self.allocations.write() else {
                continue;
            };

            let Some(traceable_object) = allocations.get_mut(&current_addr) else {
                continue;
            };

            traceable_object.color = Color::Black;
            let trace_fn = traceable_object.trace_fn;

            drop(allocations);

            let mut edges = Vec::new();
            let mut tracer = Tracer::new_edge_collector(&mut edges);
            trace_fn(with_exposed_provenance(current_addr), &mut tracer);

            for edge_addr in edges {
                if let Ok(mut allocations) = self.allocations.write()
                    && let Some(traceable_obj) = allocations.get_mut(&edge_addr)
                    && traceable_obj.color == Color::White
                {
                    traceable_obj.color = Color::Gray;
                    drop(allocations);

                    if let Ok(mut gray_objects) = self.gray_objects.lock() {
                        gray_objects.push_back(edge_addr);
                    }
                }
            }
        }

        let is_marking_done = self.gray_objects.lock().is_ok_and(|go| go.is_empty());

        if is_marking_done {
            if let Ok(mut allocations) = self.allocations.write() {
                let white_addrs: Vec<usize> = allocations
                    .iter()
                    .filter_map(|(addr, obj)| (obj.color == Color::White).then_some(*addr))
                    .collect();

                for addr in white_addrs {
                    if let Some(traceable_obj) = allocations.remove(&addr) {
                        (traceable_obj.destructor)();
                    }
                }

                allocations
                    .values_mut()
                    .for_each(|obj| obj.color = Color::White);
            }

            self.collecting.store(false, Ordering::SeqCst);
        }
    }

    pub fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.running.store(true, Ordering::SeqCst);

        let collector_ptr = self as *const AutoCollector as usize;
        let handle = thread::spawn(move || {
            let collector = unsafe { &*with_exposed_provenance::<AutoCollector>(collector_ptr) };
            collector.background_auto_collector_loop();
        });

        if let Ok(mut thread_guard) = self.background_collector_thread.lock() {
            *thread_guard = Some(handle);
        }

        Ok(())
    }

    fn stop(&self) {
        if !self.running.load(Ordering::SeqCst) {
            return;
        }

        self.running.store(false, Ordering::SeqCst);

        if let Ok(_guard) = self.background_mutex.lock() {
            self.background_cv.notify_one();
        }

        if let Ok(mut thread_guard) = self.background_collector_thread.lock()
            && let Some(handle) = thread_guard.take()
        {
            let _ = handle.join();
        }
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn wait_for_collect(&self) -> bool {
        if let Ok(guard) = self.background_mutex.lock() {
            let wait_time = Duration::from_millis(self.background_collector_interval as u64);
            let _ = self.background_cv.wait_timeout(guard, wait_time);
        }
        self.is_running()
    }

    fn background_auto_collector_loop(&self) {
        while self.is_running() {
            if !self.wait_for_collect() {
                break;
            }

            if !self.collecting.load(Ordering::SeqCst) {
                self.start_marking();
            }

            self.step();
        }
    }

    pub fn manual_gc(&self) {
        assert!(
            !self.is_running(),
            "Cannot run manual GC while background collector is running"
        );
        self.start_marking();
        while self.collecting.load(Ordering::SeqCst) {
            self.step();
        }
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations
            .read()
            .map_or(0, |allocations| allocations.len())
    }
}

impl Drop for AutoCollector {
    fn drop(&mut self) {
        if self.is_running() {
            self.stop();
        }

        if let Ok(mut allocations) = self.allocations.write() {
            allocations
                .drain()
                .for_each(|(_, traceable_obj)| (traceable_obj.destructor)());
        }
    }
}

macro_rules! impl_traceable_basic {
    ($($t:ty),*) => {
        $(
            impl Traceable for $t {
                fn trace(&self, _tracer: &mut Tracer) { }
            }
        )*
    };
}

impl_traceable_basic!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, bool, char, String,
    &str, AtomicU32
);

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::Ordering as AtomicOrdering;

    #[derive(Debug)]
    struct Node {
        value: u32,
        next: AtomicAutoPtr<Node>,
    }

    impl Node {
        fn set_next(&self, other: AutoPtr<Node>) {
            self.next.store(Some(other), AtomicOrdering::Release);
        }

        fn clear_next(&self) {
            self.next.clear(AtomicOrdering::Release);
        }
    }

    impl Traceable for Node {
        fn trace(&self, tracer: &mut Tracer) {
            if let Some(next_gc) = self.next.load(AtomicOrdering::Acquire) {
                tracer.edge(next_gc.as_ptr() as *const ());
            }
        }
    }

    #[test]
    fn test_basic_allocation_and_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let obj = gc.alloc_gc(Node {
            value: 42,
            next: AtomicAutoPtr::new(&gc),
        });
        gc.remove_root_gc(obj);

        let before_count = gc.allocation_count();
        assert_eq!(before_count, 1);

        gc.manual_gc();

        let after_count = gc.allocation_count();
        assert_eq!(after_count, 0);
    }

    #[test]
    fn test_object_reachability_through_traces() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.set_next(child);

        gc.remove_root_gc(child);

        let before_count = gc.allocation_count();
        assert_eq!(before_count, 2);

        gc.manual_gc();

        let after_count = gc.allocation_count();
        assert_eq!(after_count, 2);

        gc.remove_root_gc(parent);

        gc.manual_gc();

        let final_count = gc.allocation_count();
        assert_eq!(final_count, 0);
    }

    #[test]
    fn test_breaking_reachability() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.set_next(child);

        gc.remove_root_gc(child);

        assert_eq!(gc.allocation_count(), 2);

        parent.clear_next();

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn test_cyclic_references_with_trace() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        a.set_next(b);
        b.set_next(a);

        gc.remove_root_gc(a);
        gc.remove_root_gc(b);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_re_adding_to_roots() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.set_next(child);

        gc.remove_root_gc(parent);
        gc.remove_root_gc(child);

        gc.add_root_gc(child);

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn test_multiple_manual_collections() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        a.set_next(b);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 2);

        gc.remove_root_gc(b);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 2);

        gc.remove_root_gc(a);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_complex_object_graph_with_trace() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        #[derive(Debug)]
        struct Graph {
            nodes: [AutoPtr<Node>; 4],
        }

        unsafe impl Send for Graph {}
        unsafe impl Sync for Graph {}

        impl Traceable for Graph {
            fn trace(&self, tracer: &mut Tracer) {
                for &node in &self.nodes {
                    tracer.edge(node.as_ptr() as *const ());
                }
            }
        }

        let a = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });
        let c = gc.alloc_gc(Node {
            value: 3,
            next: AtomicAutoPtr::new(&gc),
        });
        let d = gc.alloc_gc(Node {
            value: 4,
            next: AtomicAutoPtr::new(&gc),
        });

        a.set_next(b);
        b.set_next(c);
        c.set_next(d);
        d.set_next(a);

        let graph = gc.alloc_gc(Graph {
            nodes: [a, b, c, d],
        });

        gc.remove_root_gc(a);
        gc.remove_root_gc(b);
        gc.remove_root_gc(c);
        gc.remove_root_gc(d);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 5);

        gc.remove_root_gc(graph);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_allocation_and_deallocation_cycles() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        for _ in 0..5 {
            let objects: Vec<AutoPtr<Node>> = (0..10)
                .map(|i| {
                    gc.alloc_gc(Node {
                        value: i,
                        next: AtomicAutoPtr::new(&gc),
                    })
                })
                .collect();

            for i in 0..objects.len() - 1 {
                objects[i].set_next(objects[i + 1]);
            }

            for &obj in &objects[1..] {
                gc.remove_root_gc(obj);
            }

            assert_eq!(gc.allocation_count(), 10);

            gc.remove_root_gc(objects[0]);

            gc.manual_gc();
            assert_eq!(gc.allocation_count(), 0);
        }
    }

    #[test]
    fn test_self_referencing_object() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let node = gc.alloc_gc(Node {
            value: 42,
            next: AtomicAutoPtr::new(&gc),
        });

        node.set_next(node);

        gc.remove_root_gc(node);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_partial_graph_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let root = gc.alloc_gc(Node {
            value: 0,
            next: AtomicAutoPtr::new(&gc),
        });

        let a = gc.alloc_gc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc_gc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });
        let c = gc.alloc_gc(Node {
            value: 3,
            next: AtomicAutoPtr::new(&gc),
        });
        let d = gc.alloc_gc(Node {
            value: 4,
            next: AtomicAutoPtr::new(&gc),
        });
        let e = gc.alloc_gc(Node {
            value: 5,
            next: AtomicAutoPtr::new(&gc),
        });

        gc.remove_root_gc(a);
        gc.remove_root_gc(b);
        gc.remove_root_gc(c);
        gc.remove_root_gc(d);
        gc.remove_root_gc(e);

        root.set_next(a);
        a.set_next(b);
        b.set_next(c);

        assert_eq!(gc.allocation_count(), 6);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 4);

        root.clear_next();

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn incorrect() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let obj = gc.alloc_gc(Node {
            value: 42,
            next: AtomicAutoPtr::new(&gc),
        });

        let next_node = gc.alloc_gc(Node {
            value: 179,
            next: AtomicAutoPtr::new(&gc),
        });

        gc.remove_root_gc(next_node);
        gc.manual_gc();

        obj.set_next(next_node);

        assert_eq!(gc.allocation_count(), 1);

        let result = std::panic::catch_unwind(|| {
            _ = obj
                .next
                .load(Ordering::Acquire)
                .expect("should have next")
                .value;
        });

        assert!(result.is_err());
    }

    #[test]
    fn no_data_race() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = AtomicAutoPtr::new(&gc);
        a.store(Some(gc.alloc_gc(AtomicU32::new(0))), Ordering::Release);

        thread::scope(|s| {
            s.spawn(|| {
                thread::sleep(Duration::from_millis(10));

                let x = a.load(Ordering::Acquire).unwrap();
                x.store(3, Ordering::Release);
            });

            s.spawn(|| {
                thread::sleep(Duration::from_millis(10));

                let x = a.load(Ordering::Acquire).unwrap();
                x.store(4, Ordering::Release);
            });
        });

        let final_value = a.load(Ordering::Acquire).unwrap().load(Ordering::Acquire);
        assert!(final_value == 3 || final_value == 4);
    }
}
