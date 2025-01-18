use core::{alloc::Layout, error::Error};

/// Panic with alloc error information
pub(crate) fn handle_alloc_error_<T: ?Sized, E: Error>(
    error: E,
    layout: Layout,
) -> ! {
    panic!(
        "[alloc_for_layout_::handle_alloc_error_] Err({error}) for type {} \
        with layout ({layout:?})",
        core::any::type_name::<T>(),
    )
}
