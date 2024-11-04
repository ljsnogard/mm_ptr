use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp,
    fmt,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::{self, NonNull}
};

use abs_mm::{
    mem_alloc::TrMalloc,
    res_man::{TrBoxed, TrUnique},
};

use crate::alloc_for_layout_::{AllocForLayout, Inner};

type OwnedInner<T, A> = Inner<A, (), T>;

/// A smart pointer with specified allocator.
#[derive(Debug)]
pub struct Owned<T, A>(NonNull<OwnedInner<T, A>>)
where
    T: ?Sized,
    A: TrMalloc + Clone;

impl<T, A> Owned<T, A>
where
    A: TrMalloc + Clone,
{
    pub fn new(data: T, alloc: A) -> Self {
        let mem_to_inner = |mem| { mem as *mut Inner<A, (), T> };
        unsafe {
            let mut inner = AllocForLayout::allocate_for_layout(
                alloc,
                Layout::for_value(&data),
                mem_to_inner
            );
            let inner = inner.as_mut();
            inner.data().as_ptr().write(data);
            Self::from_owned_inner_(inner)
        }
    }

    pub fn pin(data: T, alloc: A) -> Pin<Self> {
        unsafe { Pin::new_unchecked(Self::new(data, alloc)) }
    }
}

impl<T, A> Owned<[T], A>
where
    A: TrMalloc + Clone,
{
    pub fn new_slice(
        len: usize,
        mut init_each: impl FnMut(usize) -> T,
        alloc: A,
    ) -> Self {
        unsafe {
            let mut a = AllocForLayout::<A, (), [T]>
                ::allocate_for_inner_with_slice(len, alloc);
            let inner = a.as_mut();
            for (i, x) in inner.data().as_mut().iter_mut().enumerate() {
                ptr::write(x, init_each(i))
            }
            Self::from_owned_inner_(inner)
        }
    }
}

impl<T, A> Owned<[MaybeUninit<T>], A>
where
    A: TrMalloc + Clone,
{
    pub fn new_uninit_slice(len: usize, alloc: A) -> Self {
        unsafe {
            let mut a = AllocForLayout::<A, (), [MaybeUninit<T>]>
                ::allocate_for_inner_with_slice(len, alloc);
            let inner = a.as_mut();
            Self::from_owned_inner_(inner)
        }
    }
}

impl<'a, T, A> Owned<T, A>
where
    T: ?Sized,
    A: 'a + TrMalloc + Clone,
{
    pub fn leak(owned: Owned<T, A>) -> &'a mut T {
        let inner = unsafe { owned.0.as_ref() };
        core::mem::forget(owned);
        unsafe { inner.data().as_mut() }
    }
}

impl<T, A> Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    /// Constructs an `Owned<T, A>` from a raw resource.
    ///
    /// After calling this function, the raw resource is owned by the resulting
    /// `Owned<T, A>`. Specifically, the `Owned` destructor will call the 
    /// destructor of `T` and free the allocated memory. For this to be safe,
    /// the memory must have been allocated in accordance with the memory 
    /// layout used by `Owned<T, A>` .
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use core_malloc::CoreAlloc;
    /// use mm_ptr::{Owned, XtMallocOwned};
    ///
    /// let x = CoreAlloc::default().owned("hello".to_owned());
    /// let x_ptr = Owned::leak(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Owned` to prevent leak.
    ///     let x = Owned::<String, CoreAlloc>::from_raw(x_ptr);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Owned::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is
    /// // now dangling!
    /// ```
    pub unsafe fn from_raw(raw: *mut T) -> Owned<T, A> {
        let offset = Self::data_offset_from_inner_base_(raw);

        // Reverse the offset to find the original ArcInner.
        let inner_ptr = raw.byte_sub(offset) as *mut OwnedInner<T, A>;
        let x = inner_ptr.as_mut();
        let Option::Some(inner) = x else {
            panic!();
        };
        Self::from_owned_inner_(inner)
    }

    pub const fn malloc(&self) -> &A {
        unsafe { self.0.as_ref().allocator() }
    }

    pub const fn as_ptr(&self) -> *const T {
        unsafe { self.0.as_ref().data().as_ptr() }
    }

    /// Get the offset within an `OwnedInner` for the payload behind a pointer.
    ///
    /// # Safety
    ///
    /// The pointer must point to (and have valid metadata for) a previously
    /// valid instance of T, but the T is allowed to be dropped.
    unsafe fn data_offset_from_inner_base_(ptr: *const T) -> usize {
        // Align the unsized value to the end of the ArcInner.
        // Because RcBox is repr(C), it will always be the last field in memory.
        // SAFETY: since the only unsized types possible are slices, trait objects,
        // and extern types, the input safety requirement is currently enough to
        // satisfy the requirements of align_of_val_raw; this is an implementation
        // detail of the language that must not be relied upon outside of std.
        unsafe { Self::data_offset_align(mem::align_of_val_raw(ptr)) }
    }

    #[inline]
    fn data_offset_align(align: usize) -> usize {
        let layout = Layout::new::<OwnedInner<(), A>>();
        layout.size() + layout.padding_needed_for(align)
    }

    unsafe fn from_owned_inner_(inner: &mut OwnedInner<T, A>) -> Self {
        Self(NonNull::new(inner).expect("ptr cannot be null"))
    }
}

impl<T, A> Drop for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn drop(&mut self) {
        let inner = unsafe { self.0.as_mut() };
        inner.release(|p|
            unsafe { ptr::drop_in_place(p.as_ptr())}
        );
    }
}

impl<T, A> Deref for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().data().as_ref() }
    }
}

impl<T, A> DerefMut for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut().data().as_mut() }
    }
}

impl<T, A> Borrow<T> for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T, A> BorrowMut<T> for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn borrow_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T, A> cmp::PartialEq for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl<T, A> cmp::Eq for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{}

impl<T, A> fmt::Pointer for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = self.as_ptr();
        ptr.fmt(f)
    }
}

impl<T, A> TrBoxed for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Malloc = A;

    #[inline(always)]
    fn malloc(&self) -> &Self::Malloc {
        Owned::malloc(self)
    }
}

impl<T, A> TrUnique for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{}

unsafe impl<T, A> Send for Owned<T, A>
where
    T: ?Sized + Send,
    A: TrMalloc + Clone,
{}

unsafe impl<T, A> Sync for Owned<T, A>
where
    T: ?Sized + Sync,
    A: TrMalloc + Clone,
{}

pub trait XtMallocOwned: TrMalloc + Clone {
    fn owned<T>(&self, t: T) -> Owned<T, Self>;
}

impl<A> XtMallocOwned for A
where
    A: TrMalloc + Clone,
{
    fn owned<T>(
        &self,
        t: T,
    ) -> Owned<T, Self> {
        Owned::<T, A>::new(t, self.clone())
    }
}

#[cfg(test)]
mod tests_ {
    use std::{
        ops::Deref,
        sync::{Arc, Weak},
        vec::Vec,
    };
    use core_malloc::CoreAlloc;
    use super::{Owned, XtMallocOwned};

    #[test]
    fn drop_owned_should_drop_item() {
        let u = CoreAlloc::new().owned(Arc::new(0usize));
        let w = Arc::downgrade(u.deref());
        drop(u);
        assert!(w.upgrade().is_none());
    }

    #[test]
    fn leak_owned_should_not_drop_item() {
        let u = CoreAlloc::new().owned(Arc::new(0usize));
        let w = Arc::downgrade(u.deref());
        let a = Owned::leak(u);
        assert_eq!(w.strong_count(), 1);
        unsafe {
            drop(Owned::<Arc<usize>, CoreAlloc>::from_raw(a));
        }
        assert!(w.upgrade().is_none());
    }

    #[test]
    fn drop_owned_slice_should_drop_all_items() {
        let u_slice = Owned::new_slice(
            16usize,
            |_| Arc::new(0usize),
            CoreAlloc::new()
        );
        let w_slice: Vec<Weak<usize>> = u_slice
            .deref()
            .iter()
            .map(Arc::downgrade)
            .collect();
        drop(u_slice);
        assert!(w_slice.iter().all(|w| w.upgrade().is_none()));
    }
}
