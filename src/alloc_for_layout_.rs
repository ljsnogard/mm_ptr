use core::{
    alloc::Layout,
    marker::PhantomData,
    ptr::{self, NonNull},
};
use abs_mm::mem_alloc::TrMalloc;

#[repr(C)]
pub(crate) struct Inner<A, I, T>
where
    A: TrMalloc,
    I: Sized,
    T: ?Sized,
{
    alloc_: A,
    info_: I,
    data_: T,
}

impl<A, I, T> Inner<A, I, T>
where
    A: TrMalloc,
    I: Sized,
    T: ?Sized,
{
    pub const fn allocator(&self) -> &A {
        &self.alloc_
    }

    pub const fn info(&self) -> NonNull<I> {
        unsafe {
            NonNull::new_unchecked(&self.info_ as *const I as *mut _)
        }
    }

    pub const fn data(&self) -> NonNull<T> {
        unsafe {
            NonNull::new_unchecked(&self.data_ as *const T as *mut _)
        }
    }
}

impl<A, I, T> Inner<A, I, T>
where
    A: TrMalloc + Clone,
    I: Sized,
    T: ?Sized,
{
    pub fn release(
        &mut self,
        may_drop: impl FnOnce(NonNull<T>),
    ) {
        let alloc = self.allocator().clone();
        unsafe {
            let layout = Layout::for_value(self);
            may_drop(self.data());
            let ptr = self as *mut _;
            let p = ptr::slice_from_raw_parts(
                ptr as *mut u8,
                layout.size(),
            );
            let ptr = NonNull::new_unchecked(p as *mut [u8]);
            let res = alloc.deallocate(ptr, layout);
            assert!(res.is_ok());
        }
    }
}

pub(crate) struct AllocForLayout<A, I, T>(PhantomData<Inner<A, I, T>>)
where
    A: TrMalloc + Clone,
    I: Sized,
    T: ?Sized;

impl<A, I, T> AllocForLayout<A, I, T>
where
    A: TrMalloc + Clone,
    I: Sized,
    T: ?Sized,
{
    pub unsafe fn allocate_for_layout(
        alloc: A,
        value_layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut Inner<A, I, T>,
    ) -> NonNull<Inner<A, I, T>> {
        fn handle_alloc_error_<T: ?Sized, E>(
            _error: E,
            value_layout: Layout,
            inner_layout: Layout,
        ) -> ! {
            panic!(
                "[alloc_for_layout_::allocate_for_layout] Err({}) for type {} \
                layout (size: {}, align: {}), inner layout(size: {}, align: {})",
                core::any::type_name::<E>(),
                core::any::type_name::<T>(),
                value_layout.size(),
                value_layout.align(),
                inner_layout.size(),
                inner_layout.align(),
            )
        }
        let layout = Self::inner_layout_for_value_layout_(value_layout);
        let try_alloc = Self::try_allocate_for_layout_(
            alloc,
            layout,
            mem_to_inner,
        );
        try_alloc.unwrap_or_else(|e| handle_alloc_error_::<T, _>(
            e,
            value_layout,
            layout
        ))
    }

    /// Allocates an `Inner<A, I, T>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_inner` is called with the data pointer and must
    /// return back a (potentially fat)-pointer for the `Inner<A, I, T>`.
    unsafe fn try_allocate_for_layout_(
        alloc: A,
        layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut Inner<A, I, T>,
    ) -> Result<NonNull<Inner<A, I, T>>, <A as TrMalloc>::Err> {
        let ptr = alloc.allocate(layout)?;
        Result::Ok(Self::initialize_inner(
            ptr,
            layout,
            alloc,
            mem_to_inner,
        ))
    }

    unsafe fn initialize_inner(
        ptr: NonNull<[u8]>,
        layout: Layout,
        alloc: A,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut Inner<A, I, T>,
    ) -> NonNull<Inner<A, I, T>> {
        let inner = mem_to_inner(ptr.as_non_null_ptr().as_ptr());
        debug_assert_eq!(unsafe { Layout::for_value_raw(inner) }, layout);

        unsafe {
            ptr::addr_of_mut!((*inner).alloc_).write(alloc.clone());
        };
        NonNull::new_unchecked(inner)
    }

    fn inner_layout_for_value_layout_(layout: Layout) -> Layout {
        // Calculate layout using the given value layout.
        // Previously, layout was calculated on the expression
        // `&*(ptr as *const ArcInner<T>)`, but this created a misaligned
        // reference (see #54908).
        Layout::new::<Inner<A, I, ()>>()
            .extend(layout)
            .unwrap()
            .0
            .pad_to_align()
    }
}

impl<A, I, T> AllocForLayout<A, I, [T]>
where
    A: TrMalloc + Clone,
    I: Sized,
    T: Sized,
{
    pub unsafe fn allocate_for_inner_with_slice(
        len: usize,
        alloc: A,
    ) -> NonNull<Inner<A, I, [T]>> {
        let mem_to_inner = |mem: *mut u8| {
            let p = mem.cast::<T>();
            ptr::slice_from_raw_parts_mut(p, len) as *mut Inner<A, I, [T]>
        };
        Self::allocate_for_layout(
            alloc,
            Layout::array::<T>(len).unwrap(),
            mem_to_inner,
        )
    }
}
