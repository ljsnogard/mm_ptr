use core::alloc::{Layout, LayoutError};

use abs_mm::mem_alloc::AllocError;

/// Panic with alloc error information
pub(crate) fn handle_alloc_error_<T>(error: AllocError) -> !
where
    T: ?Sized,
{
    match error {
        AllocError::LayoutErr(layout_err)
            => handle_layout_error_::<T>(layout_err),
        AllocError::UnsupportedLayout(layout)
            => handle_unsupported_layout_::<T>(layout),
        _
            => extract::<T>(error),
    }

    fn extract<Ty: ?Sized>(err: AllocError) -> ! {
        panic!(
            "[mm_ptr::handle_alloc_error_] {err} for type {}",
            core::any::type_name::<Ty>(),
        )
    }
}

pub(crate) fn handle_layout_error_<T: ?Sized>(error: LayoutError) -> ! {
    panic!(
        "[mm_ptr::handle_layout_error_] Layout error({error}) for type {}",
        core::any::type_name::<T>(),
    )
}

pub(crate) fn handle_unsupported_layout_<T: ?Sized>(layout: Layout) -> ! {
    panic!(
        "[mm_ptr::handle_unsupported_layout_] unsupported layout({:?}) for type {}",
        layout,
        core::any::type_name::<T>(),
    )
}
