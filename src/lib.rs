#![no_std]

#![feature(alloc_layout_extra)]
#![feature(layout_for_ptr)]
#![feature(slice_ptr_get)]
#![feature(try_trait_v2)]

// To allow smart pointers to do things like `Box::new(dyn x)`
#![feature(coerce_unsized)]
#![feature(unsize)]

// We always pull in `std` during tests, because it's just easier
// to write tests when you can assume you're on a capable platform
#[cfg(test)]
extern crate std;

mod alloc_utils_;
mod owned_;
mod shared_;

pub use owned_::{Owned, XtMallocOwned};
pub use shared_::{Shared, Weak, XtMallocShared};

pub mod x_deps {
    pub use abs_mm;
    pub use atomex;
}
