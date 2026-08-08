use core::{
    alloc::{Layout, LayoutError},
    borrow::Borrow,
    cmp,
    convert::Infallible,
    error,
    fmt,
    marker::{PhantomData, PhantomPinned, Unsize},
    mem::{self, Alignment, ManuallyDrop, MaybeUninit},
    ops::{CoerceUnsized, Deref, Try},
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize},
};

use abs_mm::{
    mem_alloc::{AllocError, TrMalloc},
    res_man::{TrBoxed, TrStrongShared, TrWeakShared},
};
use atomex::{AtomexPtr, AtomicCount, CmpxchResult, TrAtomicFlags};

use crate::{
    alloc_utils_::handle_alloc_error_,
    owned_::{Owned, XtMallocOwned},
};

/// A atomic-reference-counted smart pointer for sharing the ownership of
/// resource in a multi-threaded environment.
///
/// This is not a drop-in replacement for `Arc<T>``. `Shared<T, A>` explicitly
/// separate the strong control block and the weak ones. So that when `T` drops,
/// its memory is earlier reclaimed than the case in `Arc<T>`.
///
/// This is better when a lot of observer and few of them tries to upgrade.
///
/// | Concept             | Shared          | `Arc`           |
/// | ------------------- | --------------- | --------------- |
/// | Object lifetime     | owned by strong | owned by strong |
/// | Allocation lifetime | independent     | tied to weak    |
/// | Observation         | weak-side       | strong-side     |
/// | Upgrade guarantee   | best-effort     | linearizable    |
#[derive(Debug)]
pub struct Shared<T, A>(NonNull<SharedInner<T, A>>)
where
    T: ?Sized,
    A: TrMalloc + Clone;

impl<T, A> Shared<T, A>
where
    T: Sized,
    A: TrMalloc + Clone,
{
    #[inline]
    pub fn new(data: T, alloc: A) -> Self {
        Self::try_new(data, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<T>(e))
    }

    #[inline]
    pub fn try_new(data: T, alloc: A) -> Result<Self, AllocError> {
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
            .unwrap_or_else(|e| handle_alloc_error_::<T>(e))
    }

    #[inline]
    pub fn pin(data: T, alloc: A) -> Pin<Self> {
        let emplace = |m: &mut MaybeUninit<T>| {
            m.write(data);
        };
        Self::try_pin_emplace(emplace, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<T>(e))
    }

    #[inline]
    pub fn pin_emplace<F>(emplace: F, alloc: A) -> Pin<Self>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        Self::try_pin_emplace(emplace, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<T>(e))
    }

    #[inline]
    pub fn try_pin_emplace<F>(
        emplace: F,
        alloc: A,
    ) -> Result<Pin<Self>, AllocError>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        let x = Self::try_emplace(emplace, alloc)?;
        Result::Ok(unsafe { Pin::new_unchecked(x) } )
    }

    pub fn try_emplace<F>(emplace: F, alloc: A) -> Result<Self, AllocError>
    where
        F: FnOnce(&mut MaybeUninit<T>),
    {
        let mem_to_inner = |mem| mem as *mut SharedInner<T, A>;
        let value_layout = Layout::new::<T>();
        unsafe {
            let inner = Self::try_allocate_for_layout(alloc, value_layout, mem_to_inner)?;
            let mut inner = NonNull::new_unchecked(inner);
            let data = (&mut inner.as_mut().data_) as *mut T as *mut MaybeUninit<T>;
            emplace(&mut *data);
            Result::Ok(Self::from_shared_inner_(inner.as_ref()))
        }
    }

    pub fn try_into_inner(self) -> Result<T, Option<Self>> {
        const LAST_COUNT: usize = 1usize;
        if Shared::strong_count(&self) > LAST_COUNT {
            return Result::Err(Option::Some(self));
        }
        let mut this = ManuallyDrop::new(self);
        let shared_inner = unsafe { this.0.as_mut() };
        let data = unsafe { shared_inner.data_ptr().as_ptr().read() };
        let strong_count = shared_inner.decrease_strong_count();
        debug_assert_eq!(strong_count, LAST_COUNT);
        let may_err = shared_inner.try_release_(|_| ());
        if may_err.is_none() {
            Result::Ok(data)
        } else {
            Result::Err(Option::None)
        }
    }
}

impl<T, A> Shared<[T], A>
where
    T: Sized,
    A: TrMalloc + Clone,
{
    pub fn new_slice<F>(len: usize, emplace_each: F, alloc: A) -> Self
    where
        F: FnMut(usize, &mut MaybeUninit<T>),
    {
        Self::try_new_slice(len, emplace_each, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<[T]>(e))
    }

    pub fn try_new_slice<F>(
        len: usize,
        mut emplace_each: F,
        alloc: A,
    ) -> Result<Self, AllocError>
    where
        F: FnMut(usize, &mut MaybeUninit<T>),
    {
        unsafe {
            let p = Self::try_allocate_for_inner_with_slice(alloc, len)?;
            let inner = p.as_ref();
            for (i, x) in inner.data_ptr().as_mut().iter_mut().enumerate() {
                let m = x as *mut T as *mut MaybeUninit<T>;
                emplace_each(i, &mut *m);
            }
            Result::Ok(Self::from_shared_inner_(inner))
        }
    }

    unsafe fn try_allocate_for_inner_with_slice(
        alloc: A,
        len: usize,
    ) -> Result<NonNull<SharedInner<[T], A>>, AllocError> {
        let mem_to_inner = |mem: *mut u8| {
            let p = mem.cast::<T>();
            ptr::slice_from_raw_parts_mut(p, len) as *mut SharedInner<[T], A>
        };
        let value_layout = match Layout::array::<T>(len) {
            Result::Err(layout_err) =>
                return Result::Err(AllocError::LayoutErr(layout_err)),
            Result::Ok(layout) => layout,
        };
        let p = unsafe {
            Self::try_allocate_for_layout(alloc, value_layout, mem_to_inner)?
        };
        Result::Ok(unsafe { NonNull::new_unchecked(p) })
    }
}

impl<T, A> Shared<[MaybeUninit<T>], A>
where
    A: TrMalloc + Clone,
{
    pub fn new_uninit_slice(len: usize, alloc: A) -> Self {
        Self::try_new_uninit_slice(len, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<[MaybeUninit<T>]>(e))
    }

    pub fn try_new_uninit_slice(
        len: usize,
        alloc: A,
    ) -> Result<Self, AllocError> {
        unsafe {
            let mut a = Self::try_allocate_for_inner_with_slice(alloc, len)?;
            let inner = a.as_mut();
            Result::Ok(Self::from_shared_inner_(inner))
        }
    }

    pub fn new_zeroed_slice(len: usize, alloc: A) -> Self {
        Self::try_new_zeroed_slice(len, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<[MaybeUninit<T>]>(e))
    }

    pub fn try_new_zeroed_slice(
        len: usize,
        alloc: A,
    ) -> Result<Self, AllocError> {
        let x= Self::try_new_uninit_slice(len, alloc)?;
        let p = unsafe {
            // Safe because x is the only owner of the memory allocated
            &mut *(x.as_ptr() as *mut [MaybeUninit<T>])
        };
        p.iter_mut().for_each(|m| *m = MaybeUninit::zeroed());
        Result::Ok(x)
    }
}

impl<T, A> Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    /// Allocates an `SharedInner<T, A>` with sufficient space for
    /// a possibly-unsized inner value where the value has the layout provided,
    /// returning an error if allocation fails.
    ///
    /// The function `mem_to_arcinner` is called with the data pointer
    /// and must return back a (maybe fat)-pointer for the `SharedInner<T>`.
    unsafe fn try_allocate_for_layout(
        alloc: A,
        value_layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut SharedInner<T, A>,
    ) -> Result<*mut SharedInner<T, A>, AllocError> {
        let layout = inner_layout_for_value_layout(&alloc, value_layout)
            .map_err(AllocError::LayoutErr)?;
        let Result::Ok(ptr) = alloc.allocate(layout) else {
            return Result::Err(AllocError::Unknown);
        };
        let inner = unsafe {
            Self::initialize_inner(ptr, layout, mem_to_inner, alloc)
        };
        Result::Ok(inner)
    }

    unsafe fn initialize_inner(
        ptr: NonNull<[u8]>,
        layout: Layout,
        mem_to_inner: impl FnOnce(*mut u8) -> *mut SharedInner<T, A>,
        alloc: A,
    ) -> *mut SharedInner<T, A> {
        let inner = mem_to_inner(ptr.as_non_null_ptr().as_ptr());
        debug_assert_eq!(unsafe { Layout::for_value_raw(inner) }, layout);
        unsafe {
            (&raw mut (*inner).alloc_).write(alloc);
            (&raw mut (*inner).refc_).write(AtomicUsize::new(0usize));
            (&raw mut (*inner).weak_).write(AtomicPtr::new(ptr::null_mut()));
        }
        inner
    }
}

/// Calculate layout for `ArcInner<T>` using the inner value's layout
fn inner_layout_for_value_layout<A>(
    _alloc_hint_: &A,
    layout: Layout,
) -> Result<Layout, LayoutError>
where
    A: TrMalloc + Clone,
{
    // Calculate layout using the given value layout.
    // Previously, layout was calculated on the expression
    // `&*(ptr as *const ArcInner<T>)`, but this created a misaligned
    // reference (see #54908).
    let (layout, _) = Layout::new::<SharedInner<(), A>>().extend(layout)?;
    Result::Ok(layout.pad_to_align())
}

impl<'a, T, A> Shared<T, A>
where
    T: ?Sized,
    A: 'a + TrMalloc + Clone,
{
    /// Try to leak resource without deallocating the memory.
    ///
    /// As an alias of resource may occur when a weak pointer upgrades during
    /// the call to `try_leak`.
    /// Thus `try_leak` will return `Err` as long as weak count is greater
    /// than zero.
    pub fn try_leak(shared: Shared<T, A>) -> Result<&'a mut T, Self> {
        if Self::strong_count(&shared) != 1 || Self::weak_count(&shared) > 0 {
            return Result::Err(shared);
        };
        // Safe because there is only one strong count
        let opt_shared_inner = unsafe { shared.0.as_ptr().as_mut() };
        let Option::Some(shared_inner) = opt_shared_inner else {
            // should panic, but we don't have to.
            return Result::Err(shared);
        };
        if let Option::Some(mut p_weak_inner) = shared_inner.weak_().load() {
            let weak_inner = unsafe { p_weak_inner.as_mut() };
            weak_inner.back_track_.pointer_.store(ptr::null_mut());
        };
        let inner = unsafe { shared.0.as_ref() };
        mem::forget(shared);
        unsafe { Result::Ok(inner.data_ptr().as_mut()) }
    }

    /// Construct a `Shared<T, A>` from raw resource.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    pub unsafe fn try_from_raw(raw: *mut T) -> Option<Self> {
        unsafe {
            let offset = Self::data_offset_from_inner_base_(raw);

            // Reverse the offset to find the original ArcInner.
            let inner_ptr = raw.byte_sub(offset) as *mut SharedInner<T, A>;
            let inner = inner_ptr.as_mut()?;
            if inner.strong_count() != 1 || inner.weak_count() > 0 {
                return Option::None;
            }
            let inner_ptr = NonNull::new_unchecked(inner);
            Option::Some(Self(inner_ptr))
        }
    }

    /// Get the offset within an `SharedInner` for the payload behind a pointer.
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
        unsafe {
            let opt_offset = Self::data_offset_align(mem::align_of_val_raw(ptr));
            let Option::Some(offset) = opt_offset else {
                panic!("Failed getting valid align for pointer {ptr:?}")
            };
            offset
        }
    }

    #[inline]
    fn data_offset_align(align: usize) -> Option<usize> {
        let layout = Layout::new::<SharedInner<(), A>>();
        let alignment = Alignment::new(align)?;
        let offset = layout.size() + layout.padding_needed_for(alignment);
        Option::Some(offset)
    }
}

impl<T, A> Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    /// Allocate a header for weak pointers if it is not allocated, and then
    /// create a `Weak<T, A>` associated with the header.
    pub fn downgrade(shared: &Shared<T, A>) -> Weak<T, A> {
        let shared_inner = unsafe { shared.0.as_ref() };
        let atom_weak = shared_inner.weak_();
        let p_weak_inner: * mut WeakInner<T, A>;
        let atom_guard = unsafe {
            let ptr = shared_inner
                as *const _
                as *mut SharedInner<T, A>
                as *mut WeakInner<T, A>;
            NonNull::new_unchecked(ptr)
        };
        loop {
            let r = atom_weak.try_spin_init(atom_guard);
            if let Result::Err(p) = r {
                if !ptr::eq(p.as_ptr(), atom_guard.as_ptr()) {
                    p_weak_inner = p.as_ptr();
                    break;
                }
            } else {
                p_weak_inner = WeakInner::allocate_weak_inner(shared_inner);
                atom_weak.store(p_weak_inner);
                break;
            }
        }
        assert!(!p_weak_inner.is_null());
        Weak::<T, A>::from_weak_inner_(unsafe { &*p_weak_inner })
    }

    pub fn strong_count(shared: &Shared<T, A>) -> usize {
        shared.shared_inner_().strong_count()
    }

    pub fn weak_count(shared: &Shared<T, A>) -> usize {
        shared.shared_inner_().weak_count()
    }

    pub const fn as_ptr(&self) -> *const T {
        self.shared_inner_().data_ptr().as_ptr()
    }

    #[inline(always)]
    pub fn malloc(&self) -> &A {
        self.shared_inner_().allocator()
    }

    fn from_shared_inner_(inner: &SharedInner<T, A>) -> Self {
        inner.increase_strong_count();
        let ptr = inner as *const _ as *mut _;
        let Option::Some(p) = NonNull::new(ptr) else {
            unreachable!("[Shared::from_inner_ptr_]")
        };
        Self(p)
    }

    const fn shared_inner_(&self) -> &SharedInner<T, A> {
        // Safe because it is from allocated.
        unsafe { self.0.as_ref() }
    }
}

impl<T, A> Drop for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn drop(&mut self) {
        let shared_inner = unsafe { self.0.as_mut() };
        if shared_inner.decrease_strong_count() > 1 {
            return;
        }
        let may_drop = |p: NonNull<T>|
            unsafe { ptr::drop_in_place(p.as_ptr()) };
        shared_inner.try_release_(may_drop);
    }
}

impl<T, A> Clone for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn clone(&self) -> Self {
        Shared::from_shared_inner_(self.shared_inner_())
    }
}

impl<T, U, A> CoerceUnsized<Shared<U, A>> for Shared<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
    A: TrMalloc + Clone,
{}

impl<T, A> Deref for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().data_ptr().as_ref() }
    }
}

impl<T, A> Borrow<T> for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T, A> PartialEq for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(
            self.shared_inner_() as *const _,
            other.shared_inner_() as *const _,
        )
    }
}

impl<T, A> Eq for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{}

impl<T, A> PartialEq<Weak<T, A>> for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn eq(&self, other: &Weak<T, A>) -> bool {
        let Option::Some(w) = other.opt_weak_inner_() else {
            return false;
        };
        let Option::Some(s) = self.shared_inner_().weak_().load() else {
            return false;
        };
        let s = unsafe { s.as_ref() };
        ptr::eq(w, s)
    }
}

impl<T, A> fmt::Pointer for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ptr().fmt(f)
    }
}

impl<T, A> TrBoxed for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Malloc = A;
    type Item = T;

    #[inline(always)]
    fn try_get_mem_alloc(&self) -> impl Try<Output = &Self::Malloc> {
        Result::<&A, Infallible>::Ok(self.malloc())
    }
}

impl<T, A> TrStrongShared for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Downgraded = self::Weak<T, A>;

    #[inline]
    fn strong_count(&self) -> usize {
        Shared::strong_count(self)
    }

    #[inline]
    fn try_get_weak_count(&self) -> impl Try<Output = usize> {
        Result::<usize, Infallible>::Ok(Shared::weak_count(self))
    }

    #[inline]
    fn downgrade(&self) -> Self::Downgraded {
        Shared::downgrade(self)
    }
}

unsafe impl<T, A> Send for Shared<T, A>
where
    T: ?Sized + Send + Sync,
    A: TrMalloc + Clone,
{}

unsafe impl<T, A> Sync for Shared<T, A>
where
    T: ?Sized + Send + Sync,
    A: TrMalloc + Clone,
{}

/// The weak pointer of Shared<T, A>. This is a different implementation from
/// `Arc<T>`'s weak. `Weak<T, A>` has its independent control block, so that the
/// existence of weak pointers will not prevent reclaiming memory used by `T` in
/// `Shared<T, A>`.
#[derive(Debug)]
pub struct Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    inner_: Option<NonNull<WeakInner<T, A>>>,
}

#[derive(Debug, Clone, Copy)]
pub enum WeakUpgradeError {
    /// The instance owned by the `Shared<T>` pointer is destructing.
    Destructing,

    /// The instance owned by the `Shared<T>` is memory freed.
    Destroyed,

    /// Fatal error, should panic.
    Unknown,
}

impl fmt::Display for WeakUpgradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeakUpgradeError::Destructing => write!(f, "WeakUpgradeError::Destructing"),
            WeakUpgradeError::Destroyed => write!(f, "WeakUpgradeError::Destroyed"),
            WeakUpgradeError::Unknown => write!(f, "WeakUpgradeError::Unknown"),
        }
    }
}

impl error::Error for WeakUpgradeError
{ }

impl<T, A> Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new() -> Self {
        Weak { inner_: Option::None }
    }

    pub fn try_upgrade(&self) -> Result<Shared<T, A>, WeakUpgradeError> {
        const MAX_ATTEMPT: usize = 2;

        let Option::Some(weak_inner) = self.opt_weak_inner_() else {
            return Result::Err(WeakUpgradeError::Unknown);
        };
        let Option::Some(p_shared_inner) = weak_inner.ptr_shared_ else {
            return Result::Err(WeakUpgradeError::Destroyed);
        };
        let shared_inner = unsafe { p_shared_inner.as_ref() };
        let Result::Ok(guard) = weak_inner.back_track_.try_lock(shared_inner, MAX_ATTEMPT) else {
            return Result::Err(WeakUpgradeError::Destructing);
        };
        if !ptr::eq(guard.shared_inner(), shared_inner) {
            return Result::Err(WeakUpgradeError::Unknown)
        };
        Result::Ok(Shared::from_shared_inner_(shared_inner))
    }

    /// Try to get the strong count. Sometimes this call returns `Option::None`
    /// because it can't be sure if the existence of the strong count.
    pub fn try_get_strong_count(&self) -> Option<usize> {
        const MAX_ATTEMPT: usize = 1;

        let Option::Some(weak_inner) = self.opt_weak_inner_() else {
            return Option::None; // This is not an normal memory state
        };
        let Option::Some(p_shared_inner) = weak_inner.ptr_shared_ else {
            return Option::Some(0usize);
        };
        let shared_inner = unsafe { p_shared_inner.as_ref() };
        let Result::Ok(guard) = weak_inner.back_track_.try_lock(shared_inner, MAX_ATTEMPT) else {
            return Option::None;
        };
        if !ptr::eq(guard.shared_inner(), shared_inner) {
            return Option::None;
        };
        Option::Some(shared_inner.strong_count())
    }

    pub fn weak_count(&self) -> usize {
        self.opt_weak_inner_()
            .map_or(0usize, |w| w.weak_count_usize())
    }

    pub fn allocator(&self) -> Option<&A> {
        let mut inner = self.inner_?;
        unsafe {
            let owned = Owned::from_raw(inner.as_mut());
            let p = owned.malloc() as *const A as *mut _;
            let a = NonNull::new_unchecked(p);
            mem::forget(owned);
            Option::Some(a.as_ref())
        }
    }

    fn from_weak_inner_(weak_inner: &WeakInner<T, A>) -> Self {
        weak_inner.increase_weak_count();
        let p = unsafe {
            let ptr = weak_inner as *const WeakInner<T, A> as *mut _;
            NonNull::new_unchecked(ptr)
        };
        Weak {
            inner_: Option::Some(p),
        }
    }

    fn opt_weak_inner_(&self) -> Option<&WeakInner<T, A>> {
        unsafe { Option::Some(self.inner_?.as_ref()) }
    }
}

impl<T, A> Drop for Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn drop(&mut self) {
        let Option::Some(mut p_inner) = self.inner_ else {
            return;
        };
        let weak_inner = unsafe { p_inner.as_mut() };
        let weak_count = weak_inner.decrease_weak_count();
        if weak_count != 1 || !weak_inner.back_track_.is_empty() {
            return;
        }
        WeakInner::release_weak_inner(weak_inner);
    }
}

impl<T, A> Clone for Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn clone(&self) -> Self {
        if let Option::Some(weak_inner) = self.opt_weak_inner_() {
            Weak::from_weak_inner_(weak_inner)
        } else {
            Weak::default()
        }
    }
}

impl<T, A> TrWeakShared for Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Item = T;
    type Upgraded = self::Shared<T, A>;

    #[inline]
    fn try_get_strong_count(&self) -> impl Try<Output = usize> {
        Weak::try_get_strong_count(self)
    }

    #[inline]
    fn weak_count(&self) -> usize {
        Weak::weak_count(self)
    }

    #[inline]
    fn try_upgrade(&self) -> impl Try<Output = Self::Upgraded> {
        Weak::try_upgrade(self)
    }
}

impl<T, A> Default for Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> cmp::PartialEq for Weak<T, A> {
    fn eq(&self, other: &Self) -> bool {
        match (self.inner_, other.inner_) {
            (Option::Some(this), Option::Some(that))
                => ptr::eq(this.as_ptr(), that.as_ptr()),
            (Option::None, Option::None) => true,
            _ => false,
        }
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> cmp::Eq for Weak<T, A> {}

impl<T: ?Sized, A: TrMalloc + Clone> cmp::PartialEq<Shared<T, A>> for Weak<T, A> {
    #[inline]
    fn eq(&self, other: &Shared<T, A>) -> bool {
        <Shared<T, A> as PartialEq<Weak<T, A>>>::eq(other, self)
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> fmt::Pointer for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Option::Some(p) = self.inner_ else {
            return ptr::null::<WeakInner<T, A>>().fmt(f);
        };
        p.as_ptr().fmt(f)
    }
}

unsafe impl<T, A> Send for Weak<T, A>
where
    T: ?Sized + Send + Sync,
    A: TrMalloc + Clone,
{}

unsafe impl<T, A> Sync for Weak<T, A>
where
    T: ?Sized + Send + Sync,
    A: TrMalloc + Clone,
{}

pub trait XtMallocShared: TrMalloc + Clone {
    fn shared<T>(&self, t: T) -> Shared<T, Self>;
}

impl<A> XtMallocShared for A
where
    A: TrMalloc + Clone,
{
    fn shared<T>(&self, t: T) -> Shared<T, Self> {
        Shared::<T, A>::new(t, self.clone())
    }
}

#[repr(C)]
struct SharedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    _pin_: PhantomPinned,
    alloc_: A,
    refc_: AtomicUsize,
    weak_: AtomicPtr<u8>,
    data_: T,
}

impl<T, A> SharedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn refc_(&self) -> AtomicCount<usize, &mut AtomicUsize> {
        let this_mut = self as *const _ as *mut Self;
        unsafe { AtomicCount::new(&mut (*this_mut).refc_) }
    }

    fn weak_(
        &self,
    ) -> AtomexPtr<WeakInner<T, A>, &mut AtomicPtr<WeakInner<T, A>>> {
        let this_mut = self as *const _ as *mut Self;
        unsafe {
            let weak = &mut (*this_mut).weak_ as *mut AtomicPtr<u8>;
            let weak = weak as *mut AtomicPtr<WeakInner<T, A>>;
            AtomexPtr::new(&mut *weak)
        }
    }

    const fn allocator(&self) -> &A {
        &self.alloc_
    }

    const fn data_ptr(&self) -> NonNull<T> {
        let this_mut = self as *const _ as *mut Self;
        unsafe { NonNull::new_unchecked(&mut (*this_mut).data_) }
    }

    #[inline]
    unsafe fn from_p_alloc_(
        p_alloc: *const A,
        data: *const T,
    ) -> *const SharedInner<T, A> {
        let offset_alloc = mem::offset_of!(SharedInner<T, A>, alloc_);
        let offset_data = mem::offset_of!(SharedInner<(), A>, data_);

        debug_assert!(offset_data >= offset_alloc);
        debug_assert_eq!(offset_alloc, 0usize);
        debug_assert_eq!(mem::size_of::<SharedInner<(), A>>(), offset_data);

        let p_inner = unsafe { p_alloc.byte_sub(offset_alloc) };
        // let data = unsafe { p_alloc.byte_add(offset_data - offset_alloc) };
        let metadata = ptr::metadata(data);
        ptr::from_raw_parts(p_inner, metadata)
    }

    fn try_release_(
        &mut self,
        may_drop: impl FnOnce(NonNull<T>),
    ) -> Option<ReleaseFailure> {
        const MAX_ATTEMPT: usize = 4;
        let this = self as *mut _ as *mut u8;
        loop {
            // rc here may differ as its atomic.
            let rc = self.refc_().val();
            if rc > 0 {
                break Option::Some(ReleaseFailure::Resurrected(rc));
            }
            let opt_weak_inner = unsafe { self.weak_().value().as_mut() };
            let mut weak_inner_existing = false;
            let mut opt_guard: Option<BackTrackGuard<'_, T, A>> = Option::None;
            if let Option::Some(weak_inner) = opt_weak_inner {
                weak_inner_existing = true;
                if weak_inner.weak_count_usize() == 0 {
                    WeakInner::release_weak_inner(weak_inner);
                } else {
                    opt_guard = weak_inner
                        .back_track_
                        .try_lock(self, MAX_ATTEMPT)
                        .ok();
                }
            };
            if weak_inner_existing && opt_guard.is_none() {
                break Option::Some(ReleaseFailure::BacktrackBusy);
            }
            let alloc = self.allocator().clone();
            let layout = Layout::for_value(self);
            may_drop(self.data_ptr());

            let ptr = unsafe { NonNull::new_unchecked(this) };
            let res = unsafe { alloc.deallocate(ptr, layout)};
            assert!(res.is_ok());

            if let Option::Some(guard) = opt_guard {
                guard.reset_back_track();
            }
            break Option::None;
        }
    }

    fn increase_strong_count(&self) -> usize {
        self.refc_().inc()
    }

    fn decrease_strong_count(&self) -> usize {
        self.refc_().dec()
    }

    pub fn strong_count(&self) -> usize {
        self.refc_().val()
    }

    pub fn weak_count(&self) -> usize {
        unsafe {
            self.weak_()
                .pointer()
                .as_ref()
                .map_or(0, |w| w.weak_count_usize())
        }
    }
}

/// The reason why the attempt to release SharedInner has failed.
enum ReleaseFailure {
    /// During attempt to release, an upgrade has succeeded and the strong count
    /// has increased.
    Resurrected(usize),

    /// Cannot acquire back track guard
    BacktrackBusy,
}

/// Wrapping an atomic pointer, as a field of WeakInner to trace back to the
/// SharedInner.
///
/// However, a pointer to SharedInner could be a fat pointer so it can't fit
/// into an atomic pointer directly. So the work around here is to trace back
/// to the allocator in the SharedInner, and then calculate the pointer to the
/// desired SharedInner.
struct BackTrackSharedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pointer_: AtomexPtr<A>,
    _mark_t_: PhantomData<SharedInner<T, A>>,
}

impl<T, A> BackTrackSharedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new(p: &SharedInner<T, A>) -> Self {
        let pointer = AtomicPtr::new(&p.alloc_ as *const A as *mut A);
        BackTrackSharedInner {
            pointer_: AtomexPtr::new(pointer),
            _mark_t_: PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pointer_.load().is_none()
    }

    // fn weak_inner_(&self) -> &WeakInner<T, A> {
    //     let offset = mem::offset_of!(WeakInner<T, A>, back_track_);
    //     let this = self as *const Self;
    //     let inner = unsafe { this.byte_sub(offset) as *const WeakInner<T, A> };
    //     unsafe { inner.as_ref().unwrap() }
    // }

    // pub fn try_read(
    //     &self,
    //     max_attempt: usize,
    // ) -> Result<&SharedInner<T, A>, *mut A> {
    //     let offset_alloc = mem::offset_of!(SharedInner<T, A>, alloc_);
    //     let locked_val = self.dangling_val_();
    //     let weak_inner = unsafe { self.weak_inner_() };
    //     let opt_p_shared_inner = weak_inner
    //         .ptr_shared_
    //         .map(|p| p.as_ptr());
    //     let Option::Some(p_shared_inner) = opt_p_shared_inner else {
    //         return Result::Err(ptr::null_mut())
    //     };
    //     let p_shared_inner_thin = p_shared_inner as *mut u8;

    //     let mut current;
    //     let mut c = 0usize;
    //     loop {
    //         current = self.pointer_.pointer();
    //         if current.is_null() {
    //             break;
    //         }
    //         c += 1usize;
    //         if current == locked_val {
    //             continue;
    //         }
    //         let deduced_p = unsafe { current.byte_sub(offset_alloc) as usize };
    //         if deduced_p == p_shared_inner_thin as usize {
    //             let shared_inner = unsafe { &*p_shared_inner };
    //             return Result::Ok(shared_inner)
    //         }
    //         if c >= max_attempt {
    //             break;
    //         }
    //     }
    //     Result::Err(current)
    // }

    pub fn try_lock<'a>(
        &'a self,
        shared_inner: &'a SharedInner<T, A>,
        max_attempt: usize,
    ) -> Result<BackTrackGuard<'a, T, A>, Option<CmpxchResult<*mut A>>> {
        let current = &shared_inner.alloc_ as *const A as *mut A;
        let locked_val = self.dangling_val_();
        assert_ne!(current, locked_val);

        let expect = |p| p == current;
        let desire = |_| locked_val;
        let mut result = Option::<CmpxchResult<*mut A>>::None;
        let mut c = 0usize;
        while c < max_attempt {
            let cmp_res = self
                .pointer_
                .try_once_compare_exchange_weak(current, expect, desire);
            if cmp_res.is_succ() {
                let tk = BackTrackGuard::new(self, shared_inner);
                return Result::Ok(tk)
            }
            c += 1usize;
            result = Option::Some(cmp_res);
        }
        Result::Err(result)
    }

    fn dangling_val_(&self) -> *mut A {
        self as *const Self as *const A as *mut A
    }
}

/// A short living token to keep the BackTrackSharedInner at the locked state
/// and to retrieve the pointer to the SharedInner.
///
/// This will not affect the strong count nor the weak count. You SHOULD NOT use
/// it anywhere else but only when releasing SharedInner.
struct BackTrackGuard<'a, T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    back_track_: &'a BackTrackSharedInner<T, A>,
    shared_inner_: Option<NonNull<SharedInner<T, A>>>,
}

impl<'a, T, A> BackTrackGuard<'a, T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new(
        back_track: &'a BackTrackSharedInner<T, A>,
        shared_inner: &'a SharedInner<T, A>,
    ) -> Self {
        let p = unsafe {
            let p = shared_inner as *const _ as *mut SharedInner<T, A>;
            NonNull::new_unchecked(p)
        };
        BackTrackGuard {
            back_track_: back_track,
            shared_inner_: Option::Some(p),
        }
    }

    pub fn shared_inner(&self) -> &SharedInner<T, A> {
        // This is safe because shared_inner is initialized by reference. And
        // once it's changed to null, the guard will drop.
        unsafe { self.shared_inner_.unwrap().as_ref() }
    }

    /// Consumes the guard and the back track will set to null pointer on drop.
    pub fn reset_back_track(mut self) {
        self.shared_inner_ = Option::None;
    }
}

impl<'a, T, A> Drop for BackTrackGuard<'a, T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn drop(&mut self) {
        let map_non_null = |p: NonNull<SharedInner<T, A>>| {
            unsafe { &p.as_ref().alloc_ as *const _ as *mut A }
        };
        let current = self.back_track_ as *const _ as *const A as *mut A;
        let target = self
            .shared_inner_
            .map(map_non_null)
            .unwrap_or(ptr::null_mut());
        assert_ne!(current, target);

        let expect = |p| p == current;
        let desire = |_| target;
        let result = self
            .back_track_
            .pointer_
            .try_spin_compare_exchange_weak(expect, desire);
        assert!(result.is_succ());
    }
}

#[repr(C)]
struct WeakInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    _pinned_: PhantomPinned,

    /// A back track pointer for upgrading a Weak<T>. This pointer could be
    /// mutated to null when the back-tracking SharedInner instance is
    /// dropping.
    back_track_: BackTrackSharedInner<T, A>,
    weak_count_: AtomicCount<usize>,
    ptr_shared_: Option<NonNull<SharedInner<T, A>>>,
}

impl<T, A> WeakInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new(shared_inner: &SharedInner<T, A>) -> Self {
        let p = shared_inner as *const _ as *mut SharedInner<T, A>;
        let p = unsafe { NonNull::new_unchecked(p) };
        WeakInner {
            _pinned_: PhantomPinned,
            back_track_: BackTrackSharedInner::new(shared_inner),
            weak_count_: AtomicCount::new(AtomicUsize::new(0usize)),
            ptr_shared_: Option::Some(p),
        }
    }

    fn allocate_weak_inner(
        shared_inner: &SharedInner<T, A>,
    ) -> *mut WeakInner<T, A> {
        let alloc = shared_inner.alloc_.clone();
        Owned::leak(alloc.owned(WeakInner::new(shared_inner)))
    }

    fn release_weak_inner(weak_inner: &mut WeakInner<T, A>) {
        let p = unsafe { Owned::<WeakInner<T, A>, A>::from_raw(weak_inner) };
        drop(p)
    }

    #[inline]
    fn increase_weak_count(&self) -> usize {
        self.weak_count_.inc()
    }

    #[inline]
    fn decrease_weak_count(&self) -> usize {
        self.weak_count_.dec()
    }

    #[inline]
    fn weak_count_usize(&self) -> usize {
        self.weak_count_.val()
    }
}

#[cfg(test)]
mod tests_ {
    extern crate alloc;

    use alloc::{sync::Arc, vec::Vec};
    use core::ops::Deref;

    use abs_mm::mem_alloc::{AllocError, CoreAlloc, DeallocError, TrMalloc};

    use super::{Shared, XtMallocShared};

    #[test]
    fn smoke() {
        let shared = CoreAlloc::new().shared(Arc::new(u128::MAX));
        let arc_weak = Arc::downgrade(shared.deref());
        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 0);

        let weak = Shared::downgrade(&shared);
        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 1);
        let upgrade_result = weak.try_upgrade();
        assert!(upgrade_result.is_ok());
        assert_eq!(Shared::strong_count(&shared), 2);

        let Result::Ok(upgraded_shared) = upgrade_result else {
            unreachable!()
        };
        drop(shared);
        assert_eq!(Shared::strong_count(&upgraded_shared), 1);
        drop(upgraded_shared);
        assert!(arc_weak.upgrade().is_none());
        assert!(weak.try_upgrade().is_err())
    }

    #[test]
    fn coercion_should_work() {
        let p: Shared<dyn TrMalloc<AllocErr = AllocError, DeallocErr = DeallocError>, CoreAlloc> =
            Shared::new(CoreAlloc::new(), CoreAlloc::new());
        drop(p);
        let p: Shared<[usize], CoreAlloc> =
            Shared::new([0usize; 10], CoreAlloc::new());
        drop(p);
    }

    #[test]
    fn leak_shared_should_not_drop_item() {
        let s = CoreAlloc::new().shared(Arc::new(0usize));
        let w = Arc::downgrade(s.deref());
        let a = Shared::try_leak(s).unwrap();
        assert_eq!(w.strong_count(), 1);
        let upgraded = w.upgrade().unwrap();
        let s = unsafe {
            Shared::<Arc<usize>, CoreAlloc>::try_from_raw(a).unwrap()
        };
        assert_eq!(s.deref(), &upgraded);
    }

    #[test]
    fn new_slice_smoke() {
        const LEN: usize = 1024;
        let shared = Shared::<[Arc<usize>], CoreAlloc>::new_slice(
            LEN,
            |u, m| { m.write(Arc::new(u)); },
            CoreAlloc::new(),
        );
        let arc_clone: Vec<_> = shared.iter().cloned().collect();
        let arc_weak: Vec<_> = shared.iter().map(Arc::downgrade).collect();

        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 0);

        let weak = Shared::downgrade(&shared);
        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 1);

        let upgrade_result = weak.try_upgrade();
        assert!(upgrade_result.is_ok());
        assert_eq!(Shared::strong_count(&shared), 2);

        drop(shared);
        drop(upgrade_result);

        let x = arc_clone
            .into_iter()
            .enumerate()
            .all(|(u, a)| u == *a);
        assert!(x, "all equal");

        let x = arc_weak
            .into_iter()
            .all(|w| w.upgrade().is_none());
        assert!(x, "all upgrade is none");
    }

    #[test]
    fn into_inner_smoke() {
        let shared = CoreAlloc::new().shared(Arc::new(u128::MAX));
        let arc_weak = Arc::downgrade(shared.deref());
        assert_eq!(Arc::strong_count(shared.deref()), 1);
        let r = shared.try_into_inner();
        assert!(r.is_ok(), "Shared::try_into_inner failed.");
        let arc = r.ok().unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(*arc, u128::MAX);
        let upgrade_result = arc_weak.upgrade();
        assert!(upgrade_result.is_some());
        assert_eq!(upgrade_result.unwrap(), arc);
    }
}
