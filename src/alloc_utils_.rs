use core::{
    alloc::{Layout, LayoutError},
    error::Error,
};

use abs_mm::mem_alloc::TrMalloc;
use anylr::Either;

pub(crate) type TryAllocError<A> = Either<
    LayoutError,
    (Layout, <A as TrMalloc>::Err)
>;

/// Panic with alloc error information
pub(crate) fn handle_alloc_error_<T: ?Sized, E: Error>(
    error: E,
    layout: Layout,
) -> ! {
    panic!(
        "[mm_ptr::handle_alloc_error_] Err({error}) for type {} \
        with layout ({layout:?})",
        core::any::type_name::<T>(),
    )
}

pub(crate) fn handle_layout_error_<T: ?Sized>(error: LayoutError) -> ! {
    panic!(
        "[mm_ptr::handle_layout_error_] Err({error}) for type {}",
        core::any::type_name::<T>(),
    )
}

pub(crate) fn handle_try_alloc_error_<T: ?Sized, A: TrMalloc>(e: TryAllocError<A>) -> ! {
    match e {
        Either::Left(layout_err) => handle_layout_error_::<T>(layout_err),
        Either::Right((layout, alloc_err)) => handle_alloc_error_::<T, _>(alloc_err, layout)
    }
}
