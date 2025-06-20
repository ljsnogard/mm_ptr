use core::{
    alloc::{Layout, LayoutError},
    borrow::{Borrow, BorrowMut},
    cmp,
    fmt,
    marker::{PhantomPinned, Unsize},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{CoerceUnsized, Deref, DerefMut},
    pin::Pin,
    ptr::{self, NonNull}
};

use abs_mm::{
    as_pinned::{TrAsPinned, TrAsPinnedMut},
    mem_alloc::TrMalloc,
    res_man::{TrBoxed, TrUnique},
};
use anylr::Either;

use crate::alloc_utils_::{handle_try_alloc_error_, TryAllocError};

/// A smart pointer that owns the resource managed by the specified allocator,
/// like a `Box<T>`, and resource can be emplaced during construction.
#[derive(Debug)]
pub struct Owned<T, A>(NonNull<OwnedInner<T, A>>)
where
    T: ?Sized,
    A: TrMalloc + Clone;

impl<T, A> Owned<T, A>
where
    T: Sized,
    A: TrMalloc + Clone,
{
    #[inline]
    pub fn new(data: T, alloc: A) -> Self {
        let emplace = |m: &mut MaybeUninit<T>| {
            m.write(data);
        };
        Self::try_emplace(emplace, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<T, A>(e))
    }

    #[inline]
    pub fn try_new(data: T, alloc: A) -> Result<Self, TryAllocError<A>> {
        let emplace = |m: &mut MaybeUninit<T>| {
            m.write(data);
        };
        Self::try_emplace(emplace, alloc)
    }

    #[inline]
    pub fn emplace<F>(emplace: F, alloc: A) -> Self
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        Self::try_emplace(emplace, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<T, A>(e))
    }

    #[inline]
    pub fn pin(data: T, alloc: A) -> Pin<Self> {
        let emplace = |m: &mut MaybeUninit<T>| {
            m.write(data);
        };
        Self::pin_emplace(emplace, alloc)
    }

    #[inline]
    pub fn pin_emplace<F>(emplace: F, alloc: A) -> Pin<Self>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        Self::try_pin_emplace(emplace, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<T, A>(e))
    }

    #[inline]
    pub fn try_pin_emplace<F>(
        emplace: F,
        alloc: A,
    ) -> Result<Pin<Self>, TryAllocError<A>>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        let x = Self::try_emplace(emplace, alloc)?;
        Result::Ok(unsafe { Pin::new_unchecked(x) } )
    }

    pub fn try_emplace<F>(emplace: F, alloc: A) -> Result<Self, TryAllocError<A>>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        let mem_to_inner = |mem| { mem as *mut OwnedInner<T, A> };
        let value_layout = Layout::new::<T>();
        unsafe {
            let inner = Self::try_allocate_for_layout(alloc, value_layout, mem_to_inner)?;
            let inner = &mut *inner;
            let data = &mut inner.data_ as *mut T as *mut MaybeUninit<T>;
            emplace(&mut *data);
            Result::Ok(Self::from_owned_inner_(inner))
        }
    }

    pub fn into_inner(self) -> T {
        unsafe {
            let mut this = ManuallyDrop::new(self);
            let owned_inner = this.0.as_mut();
            let data = owned_inner.data_ptr().as_ptr().read();
            owned_inner.release(|_| ());
            data
        }
    }
}

impl<T, A> Owned<[T], A>
where
    T: Sized,
    A: TrMalloc + Clone,
{
    pub fn new_slice<F>(len: usize, emplace_each: F, alloc: A) -> Self
    where
        F: FnMut(usize, &mut MaybeUninit<T>),
    {
        Self::try_new_slice(len, emplace_each, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<[T], A>(e))
    }

    pub fn try_new_slice<F>(
        len: usize,
        mut emplace_each: F,
        alloc: A,
    ) -> Result<Self, TryAllocError<A>>
    where
        F: FnMut(usize, &mut MaybeUninit<T>),
    {
        let array_layout = match Layout::array::<T>(len) {
            Result::Ok(x) => x,
            Result::Err(layout_err) =>
                return Result::Err(Either::Left(layout_err)),
        };
        unsafe {
            let mut a = Self::try_allocate_for_inner_with_slice(alloc, len, array_layout)?;
            let inner = a.as_mut();
            for (u, x) in inner.data_ptr().as_mut().iter_mut().enumerate() {
                let m = x as *mut T as *mut MaybeUninit<T>;
                emplace_each(u, &mut *m);
            }
            Result::Ok(Self::from_owned_inner_(inner))
        }
    }

    unsafe fn try_allocate_for_inner_with_slice(
        alloc: A,
        len: usize,
        array_layout: Layout,
    ) -> Result<NonNull<OwnedInner<[T], A>>, Either<LayoutError, (Layout, A::Err)>> {
        let mem_to_inner = |mem: *mut u8| {
            let p = mem.cast::<T>();
            ptr::slice_from_raw_parts_mut(p, len) as *mut OwnedInner<[T], A>
        };
        let value_layout = array_layout;
        let p = unsafe {
            Self::try_allocate_for_layout(alloc, value_layout, mem_to_inner)?
        };
        Result::Ok(unsafe { NonNull::new_unchecked(p) })
    }
}

impl<T, A> Owned<[MaybeUninit<T>], A>
where
    T: Sized,
    A: TrMalloc + Clone,
{
    pub fn new_uninit_slice(len: usize, alloc: A) -> Self {
        Self::try_new_uninit_slice(len, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<[MaybeUninit<T>], A>(e))
    }

    pub fn try_new_uninit_slice(
        len: usize,
        alloc: A,
    ) -> Result<Self, TryAllocError<A>> {
        let array_layout = Layout::array::<T>(len).unwrap();
        unsafe {
            let mut a = Self::try_allocate_for_inner_with_slice(alloc, len, array_layout)?;
            let inner = a.as_mut();
            Result::Ok(Self::from_owned_inner_(inner))
        }
    }

    pub fn new_zeroed_slice(len: usize, alloc: A) -> Self {
        Self::try_new_zeroed_slice(len, alloc)
            .unwrap_or_else(|e| handle_try_alloc_error_::<[MaybeUninit<T>], A>(e))
    }

    pub fn try_new_zeroed_slice(
        len: usize,
        alloc: A,
    ) -> Result<Self, Either<LayoutError, (Layout, A::Err)>> {
        let mut x= Self::try_new_uninit_slice(len, alloc)?;
        x.iter_mut().for_each(|m| *m = MaybeUninit::zeroed());
        Result::Ok(x)
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
        unsafe { inner.data_ptr().as_mut() }
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
    /// use abs_mm::mem_alloc::CoreAlloc;
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
        unsafe {
            let offset = Self::data_offset_from_inner_base_(raw);

            // Reverse the offset to find the original ArcInner.
            let inner_ptr = raw.byte_sub(offset) as *mut OwnedInner<T, A>;
            Self::from_owned_inner_(&mut *inner_ptr)
        }
    }

    pub const fn malloc(&self) -> &A {
        unsafe { self.0.as_ref().allocator() }
    }

    pub const fn as_ptr(&self) -> *const T {
        unsafe { self.0.as_ref().data_ptr().as_ptr() }
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

    // unsafe fn allocate_for_layout(
    //     alloc: A,
    //     value_layout: Layout,
    //     mem_to_inner: impl FnOnce(*mut u8) -> *mut OwnedInner<T, A>,
    // ) -> *mut OwnedInner<T, A> {
    //     let layout = inner_layout_for_value_layout(&alloc, value_layout);
    //     let ptr = alloc
    //         .allocate(layout)
    //         .unwrap_or_else(|e|
    //             handle_alloc_error_::<T, _>(e, layout));
    //     unsafe { Self::initialize_inner(ptr, layout, mem_to_inner, alloc) }
    // }

    /// Allocates an `OwnedInner<T, A>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_arcinner` is called with the data pointer
    /// and must return back a (potentially fat)-pointer for the `ArcInner<T>`.
    unsafe fn try_allocate_for_layout(
        alloc: A,
        value_layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut OwnedInner<T, A>,
    ) -> Result<*mut OwnedInner<T, A>, Either<LayoutError, (Layout, A::Err)>> {
        let layout = inner_layout_for_value_layout(&alloc, value_layout)
            .map_err(|e| Either::Left(e))?;
        let ptr = alloc
            .allocate(layout)
            .map_err(|e| Either::Right((layout, e)))?;
        let inner = unsafe {
            Self::initialize_inner(ptr, layout, mem_to_inner, alloc)
        };
        Result::Ok(inner)
    }

    unsafe fn initialize_inner(
        ptr: NonNull<[u8]>,
        layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut OwnedInner<T, A>,
        alloc: A,
    ) -> *mut OwnedInner<T, A> {
        let inner = mem_to_inner(ptr.as_non_null_ptr().as_ptr());
        debug_assert_eq!(unsafe { Layout::for_value_raw(inner) }, layout);

        unsafe { (&raw mut (*inner).alloc_).write(alloc); }
        inner
    }
}

/// Calculate layout for `OwnedInner<T>` using the inner value's layout
fn inner_layout_for_value_layout<A: TrMalloc>(
    _alloc_hint_: &A,
    layout: Layout,
) -> Result<Layout, LayoutError> {
    // Calculate layout using the given value layout.
    // Previously, layout was calculated on the expression
    // `&*(ptr as *const ArcInner<T>)`, but this created a misaligned
    // reference (see #54908).
    let (layout, _) = Layout::new::<OwnedInner<(), A>>().extend(layout)?;
    Result::Ok(layout.pad_to_align())
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

impl<T, U, A> CoerceUnsized<Owned<U, A>> for Owned<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
    A: TrMalloc + Clone,
{}

impl<T, A> Deref for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().data_ptr().as_ref() }
    }
}

impl<T, A> DerefMut for Owned<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut().data_ptr().as_mut() }
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
{
    type Item = T;
}

impl<'a, T, A> TrAsPinned<'a, T> for Pin<Owned<T, A>>
where
    Self: 'a,
    T: 'a + ?Sized,
    A: 'a + TrMalloc + Clone,
{
    fn as_pinned(self) -> Pin<&'a T> {
        unsafe {
            let t = self.as_ref().get_ref() as *const T;
            Pin::new_unchecked(&*t)
        }
    }
}

impl<'a, T, A> TrAsPinnedMut<'a, T> for Pin<Owned<T, A>>
where
    Self: 'a,
    T: 'a + ?Sized,
    A: 'a + TrMalloc + Clone,
{
    fn as_pinned_mut(mut self) -> Pin<&'a mut T> {
        unsafe {
            let t = self.as_mut().get_unchecked_mut() as *mut T;
            Pin::new_unchecked(&mut *t)
        }
    }
}

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

#[derive(Debug)]
#[repr(C)]
struct OwnedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc,
{
    _pin_: PhantomPinned,
    alloc_: A,
    data_: T,
}

impl<T, A> OwnedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc,
{
    pub const fn allocator(&self) -> &A {
        &self.alloc_
    }

    pub const fn data_ptr(&self) -> NonNull<T> {
        unsafe {
            NonNull::new_unchecked(&self.data_ as *const T as *mut _)
        }
    }
}

impl<T, A> OwnedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub fn release(
        &mut self,
        may_drop: impl FnOnce(NonNull<T>),
    ) {
        let alloc = self.allocator().clone();
        unsafe {
            let layout = Layout::for_value(self);
            may_drop(self.data_ptr());
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

#[cfg(test)]
mod tests_ {
    use std::{
        ops::Deref,
        sync::{Arc, Weak},
        vec::Vec,
    };
    use abs_mm::mem_alloc::{CoreAlloc, CoreAllocError, TrMalloc};

    use super::{Owned, XtMallocOwned};

    #[test]
    fn coercion_should_work() {
        let p: Owned<dyn TrMalloc<Err = CoreAllocError>, CoreAlloc> = 
            Owned::new(CoreAlloc::new(), CoreAlloc::new());
        drop(p);
        let p: Owned<[usize], CoreAlloc> =
            Owned::new([0usize; 10], CoreAlloc::new());
        drop(p);
    }

    #[test]
    fn drop_owned_should_drop_item() {
        let u = CoreAlloc::new().owned(Arc::new(0usize));
        let w = Arc::downgrade(u.deref());
        drop(u);
        assert!(w.upgrade().is_none());
    }

    #[test]
    fn into_inner_should_not_drop_item() {
        let u = CoreAlloc::new().owned(Arc::new(0usize));
        let w = Arc::downgrade(u.deref());
        assert_eq!(w.strong_count(), 1);
        let x = u.into_inner();
        assert_eq!(w.strong_count(), 1);
        drop(x);
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
            |_, m| { m.write(Arc::new(0usize)); },
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
