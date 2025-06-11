use rust_gc::{AutoCollector, AutoCollectorConfig, Traceable, Tracer};
use std::sync::atomic::{AtomicPtr, Ordering};

#[derive(Debug)]
struct TreeNode {
    value: i32,
    left: Option<AtomicPtr<TreeNode>>,
    right: Option<AtomicPtr<TreeNode>>,
}

unsafe impl Send for TreeNode {}
unsafe impl Sync for TreeNode {}

impl Traceable for TreeNode {
    fn trace(&self, tracer: &mut Tracer) {
        if let Some(ref left_atomic) = self.left {
            let left_ptr = left_atomic.load(Ordering::Acquire);
            if !left_ptr.is_null() {
                tracer.edge(left_ptr as *const _);
            }
        }
        if let Some(ref right_atomic) = self.right {
            let right_ptr = right_atomic.load(Ordering::Acquire);
            if !right_ptr.is_null() {
                tracer.edge(right_ptr as *const _);
            }
        }
    }
}

impl TreeNode {
    fn new(collector: &AutoCollector, value: i32) -> *mut TreeNode {
        collector.alloc(TreeNode {
            value,
            left: Some(AtomicPtr::new(std::ptr::null_mut())),
            right: Some(AtomicPtr::new(std::ptr::null_mut())),
        })
    }

    fn insert(&self, collector: &AutoCollector, value: i32) {
        if value < self.value {
            if let Some(ref left_atomic) = self.left {
                let left_ptr = left_atomic.load(Ordering::Acquire);
                if left_ptr.is_null() {
                    let new_node = TreeNode::new(collector, value);
                    collector.remove_root(new_node);
                    left_atomic.store(new_node, Ordering::Release);
                } else {
                    unsafe {
                        (*left_ptr).insert(collector, value);
                    }
                }
            }
        } else if let Some(ref right_atomic) = self.right {
            let right_ptr = right_atomic.load(Ordering::Acquire);
            if right_ptr.is_null() {
                let new_node = TreeNode::new(collector, value);
                collector.remove_root(new_node);
                right_atomic.store(new_node, Ordering::Release);
            } else {
                unsafe {
                    (*right_ptr).insert(collector, value);
                }
            }
        }
    }

    fn print(&self, depth: usize) {
        if let Some(ref right_atomic) = self.right {
            let right_ptr = right_atomic.load(Ordering::Acquire);
            if !right_ptr.is_null() {
                unsafe {
                    (*right_ptr).print(depth + 1);
                }
            }
        }

        for _ in 0..depth {
            print!("  ");
        }
        println!("{}", self.value);

        if let Some(ref left_atomic) = self.left {
            let left_ptr = left_atomic.load(Ordering::Acquire);
            if !left_ptr.is_null() {
                unsafe {
                    (*left_ptr).print(depth + 1);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn count_nodes(&self) -> u32 {
        let mut count = 1;

        if let Some(ref left_atomic) = self.left {
            let left_ptr = left_atomic.load(Ordering::Acquire);
            if !left_ptr.is_null() {
                unsafe {
                    count += (*left_ptr).count_nodes();
                }
            }
        }

        if let Some(ref right_atomic) = self.right {
            let right_ptr = right_atomic.load(Ordering::Acquire);
            if !right_ptr.is_null() {
                unsafe {
                    count += (*right_ptr).count_nodes();
                }
            }
        }

        count
    }
}

fn main() {
    println!("AutoCollector Binary Tree Demo");
    println!("==============================\n");

    let collector = AutoCollector::new(AutoCollectorConfig::default());

    println!("Creating binary search tree with automatic garbage collection...");

    let root = TreeNode::new(&collector, 50);

    unsafe {
        println!(
            "Left: {:?}, Right: {:?}",
            (*root).left.as_ref().map(|a| a.load(Ordering::Acquire)),
            (*root).right.as_ref().map(|a| a.load(Ordering::Acquire))
        );
    }

    let values = [30, 70, 20, 40, 60, 80, 10, 25, 35, 45];

    for value in values {
        unsafe {
            (*root).insert(&collector, value);
        }
    }

    unsafe {
        (*root).print(0);
    }

    println!("Allocated objects: {}", collector.allocation_count());

    // Remove left subtree
    unsafe {
        if let Some(ref left_atomic) = (*root).left {
            left_atomic.store(std::ptr::null_mut(), Ordering::Release);
        }
    }

    collector.manual_gc();

    println!(
        "Allocated objects after removing left subtree: {}",
        collector.allocation_count()
    );
}
