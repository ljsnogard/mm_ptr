﻿#![no_std]

#![feature(alloc_layout_extra)]
#![feature(const_alloc_layout)]
#![feature(layout_for_ptr)]
#![feature(slice_ptr_get)]
#![feature(try_trait_v2)]

// We always pull in `std` during tests, because it's just easier
// to write tests when you can assume you're on a capable platform
#[cfg(test)]
extern crate std;

mod alloc_for_layout_;
mod owned_;
mod shared_;
mod reclaim_;

pub use owned_::{Owned, XtMallocOwned};
pub use shared_::{Shared, Weak, XtMallocShared};
pub use reclaim_::{NoReclaim, Reclaim};

pub mod x_deps {
    pub use abs_mm;
    pub use atomic_sync::x_deps::{abs_sync, atomex};
    pub use atomic_sync;
}