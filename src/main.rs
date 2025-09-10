use rust_gc::{AtomicAutoPtr, AutoCollector, AutoCollectorConfig, AutoPtr, Traceable, Tracer};
use std::sync::atomic::Ordering;

#[derive(Debug)]
struct TreeNode {
    value: i32,
    left: AtomicAutoPtr<TreeNode>,
    right: AtomicAutoPtr<TreeNode>,
}

impl Traceable for TreeNode {
    fn trace(&self, tracer: &mut Tracer) {
        if let Some(left_gc) = self.left.load(Ordering::Acquire) {
            tracer.edge(&left_gc);
        }
        if let Some(right_gc) = self.right.load(Ordering::Acquire) {
            tracer.edge(&right_gc);
        }
    }
}

impl TreeNode {
    fn new(collector: &AutoCollector, value: i32) -> AutoPtr<TreeNode> {
        collector.alloc(TreeNode {
            value,
            left: AtomicAutoPtr::new(collector),
            right: AtomicAutoPtr::new(collector),
        })
    }

    fn insert(&self, collector: &AutoCollector, value: i32) {
        if value < self.value {
            if let Some(left) = self.left.load(Ordering::Acquire) {
                left.with_deref(|left| left.insert(collector, value));
            } else {
                let new_node = TreeNode::new(collector, value);
                collector.remove_root(new_node);
                self.left.store(Some(new_node), Ordering::Release);
            }
        } else if let Some(right) = self.right.load(Ordering::Acquire) {
            right.with_deref(|right| right.insert(collector, value));
        } else {
            let new_node = TreeNode::new(collector, value);
            collector.remove_root(new_node);
            self.right.store(Some(new_node), Ordering::Release);
        }
    }

    #[allow(dead_code)]
    fn count_nodes(&self) -> u32 {
        let mut count = 1;

        if let Some(left) = self.left.load(Ordering::Acquire) {
            left.with_try_deref(|left| count += left.map_or(0, |l| l.count_nodes()));
        }

        if let Some(right) = self.right.load(Ordering::Acquire) {
            right.with_try_deref(|right| count += right.map_or(0, |r| r.count_nodes()));
        }

        count
    }
}

impl From<&TreeNode> for Vec<i32> {
    fn from(node: &TreeNode) -> Self {
        let mut result = Vec::new();
        node.inorder_traversal(&mut result);
        result
    }
}

impl TreeNode {
    fn inorder_traversal(&self, result: &mut Vec<i32>) {
        if let Some(left) = self.left.load(Ordering::Acquire) {
            left.with_deref(|left| left.inorder_traversal(result));
        }

        result.push(self.value);

        if let Some(right) = self.right.load(Ordering::Acquire) {
            right.with_deref(|right| right.inorder_traversal(result));
        }
    }
}

fn main() {
    let collector = AutoCollector::new(AutoCollectorConfig::default());

    let root = TreeNode::new(&collector, 50);

    root.with_deref(|root| {
        println!(
            "Left: {:?}, Right: {:?}",
            root.left.load(Ordering::Acquire),
            root.right.load(Ordering::Acquire)
        )
    });

    let values = [30, 70, 20, 40, 60, 80, 10, 25, 35, 45];

    for value in values {
        root.with_deref(|root| root.insert(&collector, value));
    }

    root.with_deref(|root| {
        println!(
            "{}",
            Vec::<i32>::from(root)
                .into_iter()
                .map(|it| it.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
    });

    println!("Allocated objects: {}", collector.allocation_count());

    root.with_deref(|root| {
        root.left.clear(Ordering::Release);
    });
    collector.manual_gc();

    root.with_deref(|root| {
        println!(
            "{}",
            Vec::<i32>::from(root)
                .into_iter()
                .map(|it| it.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
    });

    println!(
        "Allocated objects after removing left subtree: {}",
        collector.allocation_count()
    );
}
