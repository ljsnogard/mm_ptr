use core::alloc::{AllocError, Layout, LayoutError};

/// Panic with alloc error information
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn handle_alloc_error_<T>(_error: AllocError) -> !
where
    T: ?Sized,
{
    fn _extract<Ty: ?Sized>(err: AllocError) -> ! {
        panic!(
            "[mm_ptr::handle_alloc_error_] {err} for type {}",
            core::any::type_name::<Ty>(),
        )
    }

    panic!("AllocError")
}

#[allow(dead_code)]
pub(crate) fn handle_layout_error_<T: ?Sized>(error: LayoutError) -> ! {
    panic!(
        "[mm_ptr::handle_layout_error_] Layout error({error}) for type {}",
        core::any::type_name::<T>(),
    )
}

#[allow(dead_code)]
pub(crate) fn handle_unsupported_layout_<T: ?Sized>(layout: Layout) -> ! {
    panic!(
        "[mm_ptr::handle_unsupported_layout_] unsupported layout({:?}) for type {}",
        layout,
        core::any::type_name::<T>(),
    )
}
