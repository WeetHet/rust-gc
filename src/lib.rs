#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::ptr::{with_exposed_provenance, with_exposed_provenance_mut};
use std::sync::atomic::{AtomicBool, Ordering};
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
    gc_mutex: RwLock<()>,
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
            gc_mutex: RwLock::new(()),
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
            for (_, traceable_obj) in allocations.iter_mut() {
                traceable_obj.color = Color::White;
            }
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
        let _gc_lock = self.gc_mutex.write().unwrap();
        self.collecting.store(true, Ordering::SeqCst);
        self.reset_marks();
        self.mark_roots();
    }

    fn step(&self) {
        if !self.collecting.load(Ordering::SeqCst) {
            return;
        }

        let objects_to_trace = {
            if let Ok(mut gray_objects) = self.gray_objects.lock() {
                let mut result = Vec::new();
                let mut processed = 0;

                while processed < self.steps_per_increment && !gray_objects.is_empty() {
                    if let Some(current_addr) = gray_objects.pop_front() {
                        result.push(current_addr);
                        processed += 1;
                    }
                }
                result
            } else {
                return;
            }
        };

        for current_addr in objects_to_trace {
            let trace_fn = {
                if let Ok(mut allocations) = self.allocations.write() {
                    if let Some(traceable_obj) = allocations.get_mut(&current_addr) {
                        traceable_obj.color = Color::Black;
                        traceable_obj.trace_fn
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            };

            let mut edges = Vec::new();
            let mut tracer = Tracer::new_edge_collector(&mut edges);
            trace_fn(with_exposed_provenance(current_addr), &mut tracer);

            for edge_addr in edges {
                if let Ok(mut allocations) = self.allocations.write() {
                    if let Some(traceable_obj) = allocations.get_mut(&edge_addr) {
                        if traceable_obj.color == Color::White {
                            traceable_obj.color = Color::Gray;
                            drop(allocations);

                            if let Ok(mut gray_objects) = self.gray_objects.lock() {
                                gray_objects.push_back(edge_addr);
                            }
                        }
                    }
                }
            }
        }

        let is_marking_done = {
            if let Ok(gray_objects) = self.gray_objects.lock() {
                gray_objects.is_empty()
            } else {
                false
            }
        };

        if is_marking_done {
            if let Ok(mut allocations) = self.allocations.write() {
                let white_addrs: Vec<usize> = allocations
                    .iter()
                    .filter(|(_, obj)| obj.color == Color::White)
                    .map(|(addr, _)| *addr)
                    .collect();

                for addr in white_addrs {
                    if let Some(traceable_obj) = allocations.remove(&addr) {
                        (traceable_obj.destructor)();
                    }
                }

                for (_, traceable_obj) in allocations.iter_mut() {
                    traceable_obj.color = Color::White;
                }
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

        if let Ok(mut thread_guard) = self.background_collector_thread.lock() {
            if let Some(handle) = thread_guard.take() {
                let _ = handle.join();
            }
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
        if let Ok(allocations) = self.allocations.read() {
            allocations.len()
        } else {
            0
        }
    }
}

impl Drop for AutoCollector {
    fn drop(&mut self) {
        if self.is_running() {
            self.stop();
        }

        if let Ok(mut allocations) = self.allocations.write() {
            for (_, traceable_obj) in allocations.drain() {
                (traceable_obj.destructor)();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

    #[derive(Debug)]
    struct Node {
        value: u32,
        next: Option<AtomicPtr<Node>>,
    }

    unsafe impl Send for Node {}
    unsafe impl Sync for Node {}

    impl Traceable for Node {
        fn trace(&self, tracer: &mut Tracer) {
            if let Some(ref next_atomic) = self.next {
                let next_ptr = next_atomic.load(AtomicOrdering::Acquire);
                if !next_ptr.is_null() {
                    tracer.edge(next_ptr as *const ());
                }
            }
        }
    }

    #[test]
    fn test_basic_allocation_and_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let obj = gc.alloc(Node {
            value: 42,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        gc.remove_root(obj);

        let before_count = gc.allocation_count();
        assert_eq!(before_count, 1);

        gc.manual_gc();

        let after_count = gc.allocation_count();
        assert_eq!(after_count, 0);
    }

    #[test]
    fn test_object_reachability_through_traces() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*parent).next {
                next_atomic.store(child, AtomicOrdering::Release);
            }
        }

        gc.remove_root(child);

        let before_count = gc.allocation_count();
        assert_eq!(before_count, 2);

        gc.manual_gc();

        let after_count = gc.allocation_count();
        assert_eq!(after_count, 2);

        gc.remove_root(parent);

        gc.manual_gc();

        let final_count = gc.allocation_count();
        assert_eq!(final_count, 0);
    }

    #[test]
    fn test_breaking_reachability() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*parent).next {
                next_atomic.store(child, AtomicOrdering::Release);
            }
        }

        gc.remove_root(child);

        assert_eq!(gc.allocation_count(), 2);

        unsafe {
            if let Some(ref next_atomic) = (*parent).next {
                next_atomic.store(std::ptr::null_mut(), AtomicOrdering::Release);
            }
        }

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn test_cyclic_references_with_trace() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*a).next {
                next_atomic.store(b, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*b).next {
                next_atomic.store(a, AtomicOrdering::Release);
            }
        }

        gc.remove_root(a);
        gc.remove_root(b);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_re_adding_to_roots() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let parent = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*parent).next {
                next_atomic.store(child, AtomicOrdering::Release);
            }
        }

        gc.remove_root(parent);
        gc.remove_root(child);

        gc.add_root(child);

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn test_multiple_manual_collections() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*a).next {
                next_atomic.store(b, AtomicOrdering::Release);
            }
        }

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 2);

        gc.remove_root(b);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 2);

        gc.remove_root(a);

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
            nodes: [*mut Node; 4],
        }

        unsafe impl Send for Graph {}
        unsafe impl Sync for Graph {}

        impl Traceable for Graph {
            fn trace(&self, tracer: &mut Tracer) {
                for &node in &self.nodes {
                    if !node.is_null() {
                        tracer.edge(node as *const ());
                    }
                }
            }
        }

        let a = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let c = gc.alloc(Node {
            value: 3,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let d = gc.alloc(Node {
            value: 4,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*a).next {
                next_atomic.store(b, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*b).next {
                next_atomic.store(c, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*c).next {
                next_atomic.store(d, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*d).next {
                next_atomic.store(a, AtomicOrdering::Release);
            }
        }

        let graph = gc.alloc(Graph {
            nodes: [a, b, c, d],
        });

        gc.remove_root(a);
        gc.remove_root(b);
        gc.remove_root(c);
        gc.remove_root(d);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 5);

        gc.remove_root(graph);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_allocation_and_deallocation_cycles() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        for _ in 0..5 {
            let objects: Vec<*mut Node> = (0..10)
                .map(|i| {
                    gc.alloc(Node {
                        value: i,
                        next: Some(AtomicPtr::new(std::ptr::null_mut())),
                    })
                })
                .collect();

            for i in 0..objects.len() - 1 {
                unsafe {
                    if let Some(ref next_atomic) = (*objects[i]).next {
                        next_atomic.store(objects[i + 1], AtomicOrdering::Release);
                    }
                }
            }

            for &obj in &objects[1..] {
                gc.remove_root(obj);
            }

            assert_eq!(gc.allocation_count(), 10);

            gc.remove_root(objects[0]);

            gc.manual_gc();
            assert_eq!(gc.allocation_count(), 0);
        }
    }

    #[test]
    fn test_self_referencing_object() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let node = gc.alloc(Node {
            value: 42,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        unsafe {
            if let Some(ref next_atomic) = (*node).next {
                next_atomic.store(node, AtomicOrdering::Release);
            }
        }

        gc.remove_root(node);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_partial_graph_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let root = gc.alloc(Node {
            value: 0,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        let a = gc.alloc(Node {
            value: 1,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let c = gc.alloc(Node {
            value: 3,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let d = gc.alloc(Node {
            value: 4,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });
        let e = gc.alloc(Node {
            value: 5,
            next: Some(AtomicPtr::new(std::ptr::null_mut())),
        });

        gc.remove_root(a);
        gc.remove_root(b);
        gc.remove_root(c);
        gc.remove_root(d);
        gc.remove_root(e);

        unsafe {
            if let Some(ref next_atomic) = (*root).next {
                next_atomic.store(a, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*a).next {
                next_atomic.store(b, AtomicOrdering::Release);
            }
            if let Some(ref next_atomic) = (*b).next {
                next_atomic.store(c, AtomicOrdering::Release);
            }
        }

        assert_eq!(gc.allocation_count(), 6);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 4);

        unsafe {
            if let Some(ref next_atomic) = (*root).next {
                next_atomic.store(std::ptr::null_mut(), AtomicOrdering::Release);
            }
        }

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 1);
    }
}
