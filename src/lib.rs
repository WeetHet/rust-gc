#![allow(dead_code)]

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Gray,
    Black,
}

pub trait Traceable: Send + Sync + Any {
    fn trace(&self, tracer: &mut Tracer);
}

pub struct Tracer<'a> {
    edges: Option<&'a mut Vec<usize>>,
}

impl<'a> Tracer<'a> {
    pub fn edge<T>(&mut self, to: &AutoPtr<T>) {
        if let Some(edges) = &mut self.edges {
            edges.push(to.addr);
        }
    }

    fn new_edge_collector(edges: &'a mut Vec<usize>) -> Self {
        Self { edges: Some(edges) }
    }
}

pub struct AutoPtr<T> {
    addr: usize,
    collector: *const AutoCollector,
    _marker: PhantomData<T>,
}

impl<T> Copy for AutoPtr<T> {}
impl<T> Clone for AutoPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: std::fmt::Debug + 'static> std::fmt::Debug for AutoPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_try_deref(|val| match val {
            Some(val) => val.fmt(f),
            None => write!(f, "AutoPtr(freed)"),
        })
    }
}

impl<T: 'static> AutoPtr<T> {
    pub fn with_try_deref<R>(&self, f: impl FnOnce(Option<&T>) -> R) -> R {
        let collector = unsafe { &*self.collector };

        let allocations = collector
            .allocations
            .read()
            .expect("allocations inaccessible: RWLock poisoned");

        let object = allocations
            .get(&self.addr)
            .and_then(|obj| (obj.data.as_ref() as &dyn Any).downcast_ref::<T>());

        f(object)
    }

    pub fn with_deref<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        self.with_try_deref(|val| match val {
            Some(val) => f(val),
            None => panic!("AutoPtr is freed"),
        })
    }
}

pub struct AtomicAutoPtr<T> {
    inner: AtomicUsize,
    collector: *const AutoCollector,
    _marker: PhantomData<T>,
}

impl<T: Any> Traceable for AtomicAutoPtr<T> {
    fn trace(&self, tracer: &mut Tracer) {
        if let Some(ptr) = self.load(Ordering::SeqCst) {
            tracer.edge(&ptr);
        }
    }
}

impl<T> std::fmt::Debug for AtomicAutoPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = self.inner.load(Ordering::Relaxed);
        write!(f, "AtomicAutoPtr({:#x})", addr)
    }
}

unsafe impl<T> Send for AtomicAutoPtr<T> {}
unsafe impl<T> Sync for AtomicAutoPtr<T> {}

impl<T: 'static> AtomicAutoPtr<T> {
    pub fn new(collector: &AutoCollector) -> Self {
        Self {
            inner: AtomicUsize::new(0),
            collector: collector as *const AutoCollector,
            _marker: PhantomData,
        }
    }

    pub fn with_ptr(ptr: AutoPtr<T>) -> Self {
        let addr = ptr.addr;
        Self {
            inner: AtomicUsize::new(addr),
            collector: ptr.collector,
            _marker: PhantomData,
        }
    }

    pub fn load(&self, order: Ordering) -> Option<AutoPtr<T>> {
        let addr = self.inner.load(order);
        if addr == 0 {
            None
        } else {
            Some(AutoPtr {
                addr,
                collector: self.collector,
                _marker: PhantomData,
            })
        }
    }

    pub fn store(&self, value: Option<AutoPtr<T>>, order: Ordering) {
        let addr = value.map(|gc| gc.addr).unwrap_or(0);
        self.inner.store(addr, order);
    }

    pub fn clear(&self, order: Ordering) {
        self.inner.store(0, order);
    }
}

struct TraceableObject {
    color: Color,
    data: Box<dyn Traceable>,
}

impl std::fmt::Debug for TraceableObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = format!("{:p}", self.data.as_ref());
        f.debug_struct("TraceableObject")
            .field("color", &self.color)
            .field("addr", &addr)
            .finish()
    }
}

pub struct AutoCollector {
    id: u64,
    collecting: AtomicBool,
    background_collector_interval: u64,
    steps_per_increment: usize,
    roots: RwLock<HashSet<usize>>,
    allocations: RwLock<HashMap<usize, TraceableObject>>,
    gray_objects: Mutex<VecDeque<usize>>,
    background_collector_thread: Mutex<Option<JoinHandle<()>>>,
    background_cv: Condvar,
    background_state: Mutex<bool>,
}

pub struct AutoCollectorConfig {
    pub steps_per_increment: usize,
    pub background_collector_interval: u64,
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
    pub fn with_steps_per_increment(self, steps_per_increment: usize) -> Self {
        Self {
            steps_per_increment,
            ..self
        }
    }

    pub fn with_background_collector_interval(self, background_collector_interval: u64) -> Self {
        Self {
            background_collector_interval,
            ..self
        }
    }
}

impl AutoCollector {
    pub fn new(config: AutoCollectorConfig) -> Arc<Self> {
        Arc::new(Self {
            id: rand::random(),
            collecting: AtomicBool::new(false),
            background_collector_interval: config.background_collector_interval,
            steps_per_increment: config.steps_per_increment,
            roots: RwLock::new(HashSet::new()),
            allocations: RwLock::new(HashMap::new()),
            gray_objects: Mutex::new(VecDeque::new()),
            background_collector_thread: Mutex::new(None),
            background_cv: Condvar::new(),
            background_state: Mutex::new(false),
        })
    }

    pub fn alloc<T: Traceable + 'static>(&self, value: T) -> AutoPtr<T> {
        let boxed_value = Box::new(value);
        let addr = (&raw const *boxed_value) as usize;

        let traceable_obj = TraceableObject {
            color: Color::Gray,
            data: boxed_value,
        };

        self.add_root_raw(addr);

        if let Ok(mut allocations) = self.allocations.write() {
            allocations.insert(addr, traceable_obj);
        }

        AutoPtr {
            addr,
            collector: self as *const _,
            _marker: PhantomData,
        }
    }

    pub fn add_root<T>(&self, handle: AutoPtr<T>) -> AutoPtr<T> {
        self.add_root_raw(handle.addr);
        handle
    }

    pub fn remove_root<T>(&self, handle: AutoPtr<T>) -> AutoPtr<T> {
        self.remove_root_raw(handle.addr);
        handle
    }

    fn remove_root_raw(&self, addr: usize) {
        if let Ok(mut roots) = self.roots.write() {
            roots.remove(&addr);
        }
    }

    fn add_root_raw(&self, addr: usize) {
        if let Ok(mut roots) = self.roots.write() {
            roots.insert(addr);
        }

        if let Ok(mut allocations) = self.allocations.write()
            && let Some(traceable_obj) = allocations.get_mut(&addr)
        {
            traceable_obj.color = Color::Gray;
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

            for root_addr in roots.iter() {
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

            let mut edges = Vec::new();
            let mut tracer = Tracer::new_edge_collector(&mut edges);
            traceable_object.data.trace(&mut tracer);

            drop(allocations);

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
                    _ = allocations.remove(&addr);
                }

                allocations
                    .values_mut()
                    .for_each(|obj| obj.color = Color::White);
            }

            self.collecting.store(false, Ordering::SeqCst);
        }
    }

    pub fn start(self: &Arc<Self>) {
        let mut running = self.background_state();

        *running = true;
        drop(running);

        let _self = self.clone();
        let handle = thread::spawn(move || {
            _self.background_auto_collector_loop();
        });

        if let Ok(mut thread_guard) = self.background_collector_thread.lock() {
            *thread_guard = Some(handle);
        }
    }

    fn stop(&self) {
        let mut running = self.background_state();
        if !*running {
            return;
        }

        *running = false;
        self.background_cv.notify_one();
        drop(running);

        if let Ok(mut thread_guard) = self.background_collector_thread.lock()
            && let Some(handle) = thread_guard.take()
        {
            let _ = handle.join();
        }
    }

    fn background_state(&self) -> std::sync::MutexGuard<'_, bool> {
        self.background_state.lock().unwrap()
    }

    fn is_running(&self) -> bool {
        *self.background_state()
    }

    fn wait_for_collect(&self) -> bool {
        let state = self.background_state();
        if !*state {
            return false;
        }

        let wait_time = Duration::from_millis(self.background_collector_interval);
        let (state_guard, _timeout_result) =
            self.background_cv.wait_timeout(state, wait_time).unwrap();
        *state_guard
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
    AtomicU32
);

impl<T: Send + Traceable> Traceable for Mutex<T> {
    fn trace(&self, tracer: &mut Tracer) {
        self.lock().unwrap().trace(tracer);
    }
}

impl<T: 'static> Traceable for Vec<AtomicAutoPtr<T>> {
    fn trace(&self, tracer: &mut Tracer) {
        for atomic_ptr in self {
            if let Some(ptr) = atomic_ptr.load(Ordering::SeqCst) {
                tracer.edge(&ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::Ordering;

    #[derive(Debug)]
    struct Node {
        value: u32,
        next: AtomicAutoPtr<Node>,
    }

    impl Node {
        fn set_next(&self, other: AutoPtr<Node>) {
            self.next.store(Some(other), Ordering::Release);
        }

        fn clear_next(&self) {
            self.next.clear(Ordering::Release);
        }
    }

    impl Traceable for Node {
        fn trace(&self, tracer: &mut Tracer) {
            if let Some(next_gc) = self.next.load(Ordering::Acquire) {
                tracer.edge(&next_gc);
            }
        }
    }

    #[test]
    fn test_basic_allocation_and_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let obj = gc.alloc(Node {
            value: 42,
            next: AtomicAutoPtr::new(&gc),
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
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.with_deref(|p| p.set_next(child));

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
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.with_deref(|p| p.set_next(child));

        gc.remove_root(child);

        assert_eq!(gc.allocation_count(), 2);

        parent.with_deref(|p| p.clear_next());

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn test_cyclic_references_with_trace() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = gc.alloc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        a.with_deref(|a_ref| a_ref.set_next(b));
        b.with_deref(|b_ref| b_ref.set_next(a));

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
            next: AtomicAutoPtr::new(&gc),
        });
        let child = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        parent.with_deref(|p| p.set_next(child));

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
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });

        a.with_deref(|a_ref| a_ref.set_next(b));

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
            nodes: [AtomicAutoPtr<Node>; 4],
        }

        impl Traceable for Graph {
            fn trace(&self, tracer: &mut Tracer) {
                for node in &self.nodes {
                    if let Some(node) = node.load(Ordering::SeqCst) {
                        tracer.edge(&node);
                    }
                }
            }
        }

        let a = gc.alloc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });
        let c = gc.alloc(Node {
            value: 3,
            next: AtomicAutoPtr::new(&gc),
        });
        let d = gc.alloc(Node {
            value: 4,
            next: AtomicAutoPtr::new(&gc),
        });

        a.with_deref(|a_ref| a_ref.set_next(b));
        b.with_deref(|b_ref| b_ref.set_next(c));
        c.with_deref(|c_ref| c_ref.set_next(d));
        d.with_deref(|d_ref| d_ref.set_next(a));

        let graph = gc.alloc(Graph {
            nodes: [a, b, c, d].map(AtomicAutoPtr::with_ptr),
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
            let objects: Vec<AutoPtr<Node>> = (0..10)
                .map(|i| {
                    gc.alloc(Node {
                        value: i,
                        next: AtomicAutoPtr::new(&gc),
                    })
                })
                .collect();

            for i in 0..objects.len() - 1 {
                objects[i].with_deref(|obj| obj.set_next(objects[i + 1]));
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
            next: AtomicAutoPtr::new(&gc),
        });

        node.with_deref(|n| n.set_next(node));

        gc.remove_root(node);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 0);
    }

    #[test]
    fn test_partial_graph_collection() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let root = gc.alloc(Node {
            value: 0,
            next: AtomicAutoPtr::new(&gc),
        });

        let a = gc.alloc(Node {
            value: 1,
            next: AtomicAutoPtr::new(&gc),
        });
        let b = gc.alloc(Node {
            value: 2,
            next: AtomicAutoPtr::new(&gc),
        });
        let c = gc.alloc(Node {
            value: 3,
            next: AtomicAutoPtr::new(&gc),
        });
        let d = gc.alloc(Node {
            value: 4,
            next: AtomicAutoPtr::new(&gc),
        });
        let e = gc.alloc(Node {
            value: 5,
            next: AtomicAutoPtr::new(&gc),
        });

        gc.remove_root(a);
        gc.remove_root(b);
        gc.remove_root(c);
        gc.remove_root(d);
        gc.remove_root(e);

        root.with_deref(|r| r.set_next(a));
        a.with_deref(|a_ref| a_ref.set_next(b));
        b.with_deref(|b_ref| b_ref.set_next(c));

        assert_eq!(gc.allocation_count(), 6);

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 4);

        root.with_deref(|r| r.clear_next());

        gc.manual_gc();
        assert_eq!(gc.allocation_count(), 1);
    }

    #[test]
    fn remove_before_gc() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let obj = gc.alloc(Node {
            value: 42,
            next: AtomicAutoPtr::new(&gc),
        });

        let next_node = gc.alloc(Node {
            value: 179,
            next: AtomicAutoPtr::new(&gc),
        });

        gc.remove_root(next_node);
        gc.manual_gc();

        obj.with_deref(|o| o.set_next(next_node));

        assert_eq!(gc.allocation_count(), 1);

        std::panic::set_hook(Box::new(|_| {}));

        let result = std::panic::catch_unwind(|| {
            obj.with_deref(|o| {
                _ = o
                    .next
                    .load(Ordering::Acquire)
                    .expect("should have next")
                    .with_deref(|next| next.value);
            });
        });

        // unregister custom hook so any panic after this point functions normally
        _ = std::panic::take_hook();

        assert!(result.is_err());
    }

    #[test]
    fn no_data_race() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());

        let a = AtomicAutoPtr::new(&gc);
        a.store(Some(gc.alloc(AtomicU32::new(0))), Ordering::Release);

        thread::scope(|s| {
            s.spawn(|| {
                thread::sleep(Duration::from_millis(10));

                let x = a.load(Ordering::Acquire).unwrap();
                x.with_deref(|val| val.store(3, Ordering::Release));
            });

            s.spawn(|| {
                thread::sleep(Duration::from_millis(10));

                let x = a.load(Ordering::Acquire).unwrap();
                x.with_deref(|val| val.store(4, Ordering::Release));
            });
        });

        let final_value = a
            .load(Ordering::Acquire)
            .unwrap()
            .with_deref(|val| val.load(Ordering::Acquire));
        assert!(final_value == 3 || final_value == 4);
    }

    #[test]
    fn multithreaded_usage() {
        let gc = AutoCollector::new(AutoCollectorConfig::default());
        gc.start();

        let init = gc.alloc(Node {
            value: 100,
            next: AtomicAutoPtr::new(&gc),
        });
        let last_vertex = AtomicAutoPtr::with_ptr(init);

        let roots: AtomicAutoPtr<Mutex<Vec<AtomicAutoPtr<Node>>>> =
            AtomicAutoPtr::with_ptr(gc.alloc(Mutex::new(Vec::new())));

        let add_root = |roots: &AtomicAutoPtr<Mutex<Vec<AtomicAutoPtr<Node>>>>,
                        new: &AtomicAutoPtr<Node>| {
            let Some(vertex) = new.load(Ordering::Acquire) else {
                return;
            };

            roots
                .load(Ordering::Acquire)
                .unwrap()
                .with_deref(|r| r.lock().unwrap().push(AtomicAutoPtr::with_ptr(vertex)));
        };

        add_root(&roots, &last_vertex);
        gc.remove_root(init);

        thread::scope(|s| {
            for thread_id in 0..4 {
                let (shared_ptr, gc) = (&last_vertex, &gc);
                let (roots, add_root) = (&roots, &add_root);

                s.spawn(move || {
                    for i in 0..10 {
                        if let Some(current) = shared_ptr.load(Ordering::Acquire) {
                            let new_node = gc.alloc(Node {
                                value: thread_id * 1000 + i,
                                next: AtomicAutoPtr::new(gc),
                            });

                            add_root(roots, &AtomicAutoPtr::with_ptr(new_node));

                            new_node.with_deref(|n| n.set_next(current));
                            shared_ptr.store(Some(new_node), Ordering::Release);

                            gc.remove_root(new_node);
                        }

                        thread::sleep(Duration::from_millis(1));
                    }
                });
            }
        });

        let final_node = last_vertex.load(Ordering::Acquire).unwrap();
        assert!(final_node.with_deref(|n| n.value) < 4000);

        last_vertex.clear(Ordering::Release);
        gc.remove_root(roots.load(Ordering::Acquire).unwrap());
        thread::sleep(Duration::from_millis(
            AutoCollectorConfig::default().background_collector_interval,
        ));

        gc.stop();

        gc.manual_gc();

        assert_eq!(gc.allocation_count(), 0);
    }
}
