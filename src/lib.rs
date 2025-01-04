#![no_std]

#![feature(alloc_layout_extra)]
#![feature(const_alloc_layout)]
#![feature(layout_for_ptr)]
#![feature(slice_ptr_get)]
#![feature(try_trait_v2)]

// To allow a struct implement Fn*
// #![feature(unboxed_closures)]
// #![feature(fn_traits)]

// We always pull in `std` during tests, because it's just easier
// to write tests when you can assume you're on a capable platform
#[cfg(test)]
extern crate std;

mod alloc_for_layout_;
mod owned_;
mod shared_;

pub use owned_::{Owned, XtMallocOwned};
pub use shared_::{Shared, Weak, XtMallocShared};

pub mod x_deps {
    pub use abs_mm;
    pub use atomex;
}
