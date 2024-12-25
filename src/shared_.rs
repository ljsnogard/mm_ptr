use core::{
    alloc::Layout,
    borrow::Borrow,
    cell::UnsafeCell,
    cmp,
    fmt,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize},
};

use abs_mm::{
    mem_alloc::TrMalloc,
    res_man::{TrBoxed, TrShared},
};
use atomex::{AtomicCount, AtomexPtr};
use atomic_sync::x_deps::atomex;

use crate::{
    owned_::{Owned, XtMallocOwned},
    alloc_for_layout_::{AllocForLayout, Inner},
};

/// A atomic-reference-counted smart pointer for sharing the ownership of
/// resource in multi-threaded environment.
#[derive(Debug)]
pub struct Shared<T, A>(NonNull<SharedInner<T, A>>)
where
    T: ?Sized,
    A: TrMalloc + Clone;

impl<T, A> Shared<T, A>
where
    A: TrMalloc + Clone,
{
    pub fn new(data: T, alloc: A) -> Self {
        unsafe {
            let inner = AllocForLayout::allocate_for_layout(
                alloc,
                Layout::for_value(&data),
                Self::mem_to_inner_,
            );
            let inner = SharedInner::from_inner_ptr(inner).as_ref();
            inner.data_().as_ptr().write(data);
            Self::from_shared_inner_(inner)
        }
    }

    pub fn pin(data: T, alloc: A) -> Pin<Self> {
        unsafe { Pin::new_unchecked(Self::new(data, alloc)) }
    }

    #[inline(always)]
    pub fn into_inner(self) -> Option<T> {
        self.try_into_inner().ok()
    }

    pub fn try_into_inner(self) -> Result<T, Self> {
        if Shared::strong_count(&self) > 1usize {
            return Result::Err(self);
        }
        unsafe {
            let mut this = ManuallyDrop::new(self);
            let shared_inner = this.0.as_mut();
            let data = shared_inner.data_().as_ptr().read();
            shared_inner.release_(|_| ());
            Result::Ok(data)
        }
    }

    fn mem_to_inner_(mem: *mut u8) -> *mut Inner<A, SharedInfo<T, A>, T> {
        mem as *mut Inner<A, SharedInfo<T, A>, T>
    }
}

impl<T, A> Shared<[T], A>
where
    A: TrMalloc + Clone,
{
    pub fn new_slice(
        len: usize,
        mut init_each: impl FnMut(usize, &mut MaybeUninit<T>) -> &mut T,
        alloc: A,
    ) -> Self {
        unsafe {
            let a = AllocForLayout::<A, SharedInfo<[T], A>, [T]>
                ::allocate_for_inner_with_slice(len, alloc);
            let mut p = SharedInner::from_inner_with_slice(a, len);
            let inner = p.as_mut();
            for (i, x) in inner.data_().as_mut().iter_mut().enumerate() {
                let m = x as *mut T as *mut MaybeUninit<T>;
                let _ = init_each(i, &mut *m);
            }
            Self::from_shared_inner_(inner)
        }
    }
}

impl<T, A> Shared<[MaybeUninit<T>], A>
where
    A: TrMalloc + Clone,
{
    pub fn new_uninit_slice(len: usize, alloc: A) -> Self {
        unsafe {
            let a = AllocForLayout::<
                A,
                SharedInfo<[MaybeUninit<T>], A>,
                [MaybeUninit<T>],
            >::allocate_for_inner_with_slice(len, alloc);
            let mut p = SharedInner::from_inner_with_slice(a, len);
            let inner = p.as_mut();
            Self::from_shared_inner_(inner)
        }
    }
}

impl<'a, T, A> Shared<T, A>
where
    T: ?Sized,
    A: 'a + TrMalloc + Clone
{
    /// Try to leak resource without deallocating the memory.
    ///
    /// An alias of resource may occur when a weak pointer upgrades during
    /// the call to `try_leak`.
    /// For now `try_leak` will return `Err` even though weak count is greater
    /// than zero.
    pub fn try_leak(shared: Shared<T, A>) -> Result<&'a mut T, Self> {
        if Self::strong_count(&shared) != 1 || Self::weak_count(&shared) > 0 {
            return Result::Err(shared);
        };
        // Safe because there is only one strong count
        let opt_shared_inner = unsafe { shared.0.as_ptr().as_mut() };
        let Option::Some(shared_inner) = opt_shared_inner else {
            return Result::Err(shared);
        };
        if let Option::Some(mut p_weak_inner) = shared_inner.weak_().load() {
            let weak_inner = unsafe { p_weak_inner.as_mut() };
            weak_inner.reset_back_track();
        };
        let inner = unsafe { shared.0.as_ref() };
        mem::forget(shared);
        unsafe { Result::Ok(inner.data_().as_mut()) }
    }

    /// Construct a `Shared<T, A>` from raw resource.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    pub unsafe fn try_from_raw(raw: *mut T) -> Option<Self> {
        let offset = Self::data_offset_from_inner_base_(raw);

        // Reverse the offset to find the original ArcInner.
        let inner_ptr = raw.byte_sub(offset) as *mut SharedInner<T, A>;
        let inner = inner_ptr.as_mut()?;
        if inner.strong_count() != 1 || inner.weak_count() > 0 {
            return Option::None;
        }
        let inner_ptr = unsafe { NonNull::new_unchecked(inner) };
        Option::Some(Self(inner_ptr))
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
        unsafe { Self::data_offset_align(mem::align_of_val_raw(ptr)) }
    }

    #[inline]
    fn data_offset_align(align: usize) -> usize {
        let layout = Layout::new::<SharedInner<(), A>>();
        layout.size() + layout.padding_needed_for(align)
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
                let back_track = unsafe {
                    let p = shared_inner as *const _ as *mut _;
                    let p = NonNull::new_unchecked(p);
                    Option::Some(p)
                };
                p_weak_inner = unsafe {
                    WeakInner::allocate_weak_inner(
                        back_track,
                        shared_inner.allocator(),
                    )
                };
                atom_weak.store(p_weak_inner);
                break;
            }
        }
        assert!(!p_weak_inner.is_null());
        Weak::from_weak_inner_(unsafe { &*p_weak_inner })
    }

    pub fn strong_count(shared: &Shared<T, A>) -> usize {
        shared.shared_inner_().strong_count()
    }

    pub fn weak_count(shared: &Shared<T, A>) -> usize {
        shared.shared_inner_().weak_count()
    }

    pub const fn as_ptr(&self) -> *const T {
        self.shared_inner_().data_().as_ptr()
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
        shared_inner.release_(|p|
            unsafe { ptr::drop_in_place(p.as_ptr()) }
        );
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> Clone for Shared<T, A> {
    fn clone(&self) -> Self {
        Shared::from_shared_inner_(self.shared_inner_())
    }
}

impl<T, A> Deref for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().data_().as_ref() }
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
        let Option::Some(back_track) = w.current_back_track() else {
            return false;
        };
        ptr::eq(back_track.as_ptr(), self.shared_inner_())
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

    #[inline(always)]
    fn malloc(&self) -> &Self::Malloc {
        Shared::malloc(self)
    }
}

impl<T, A> TrShared for Shared<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    #[inline(always)]
    fn strong_count(&self) -> usize {
        Shared::strong_count(self)
    }

    #[inline(always)]
    fn weak_count(&self) -> usize {
        Shared::weak_count(self)
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

#[derive(Debug)]
pub struct Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    inner_: Option<NonNull<WeakInner<T, A>>>,
}

impl<T, A> Weak<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new() -> Self {
        Weak { inner_: Option::None }
    }

    pub fn upgrade(&self) -> Option<Shared<T, A>> {
        Option::Some(Shared::from_shared_inner_(unsafe {
            self.opt_weak_inner_()?.current_back_track()?.as_ref()
        }))
    }

    pub fn strong_count(&self) -> usize {
        self.opt_weak_inner_()
            .and_then(|w| w.current_back_track())
            .map(|p| unsafe { p.as_ref() })
            .map_or(0usize, |inner| inner.strong_count())
    }

    pub fn weak_count(&self) -> usize {
        self.opt_weak_inner_()
            .map_or(0usize, |w| w.weak_count_usize())
    }

    pub fn as_ptr(&self) -> Option<*const T> {
        self.opt_weak_inner_()
            .and_then(|w| w.current_back_track())
            .map(|p| unsafe { p.as_ref().data_().as_ptr() as *const T })
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
        if let Option::Some(weak_inner) = self
            .inner_
            .map(|mut p| unsafe { p.as_mut() })
            .filter(|w| w.decrease_weak_count() == 1
                    && w.current_back_track().is_none())
        {
            WeakInner::release_weak_inner(weak_inner);
        }
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
        match (self.as_ptr(), other.as_ptr()) {
            (Option::Some(this), Option::Some(that)) => ptr::eq(this, that),
            (Option::None, Option::None) => true,
            _ => false,
        }
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> cmp::Eq for Weak<T, A> {}

impl<T: ?Sized, A: TrMalloc + Clone> cmp::PartialEq<Shared<T, A>> for Weak<T, A> {
    fn eq(&self, other: &Shared<T, A>) -> bool {
        let Option::Some(w) = self.opt_weak_inner_() else {
            return false;
        };
        let Option::Some(p) = w.current_back_track() else {
            return false;
        };
        ptr::eq(p.as_ptr(), other.shared_inner_())
    }
}

impl<T: ?Sized, A: TrMalloc + Clone> fmt::Pointer for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Option::Some(p) = self.as_ptr() {
            return p.fmt(f);
        }
        if let Option::Some(x) = self.opt_weak_inner_() {
            let p = x as *const WeakInner<T, A>;
            return p.fmt(f);
        }
        ptr::null::<()>().fmt(f)
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

struct SharedInfo<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    refc_: AtomicCount<usize>,
    weak_: AtomexPtr<WeakInner<T, A>>,
}

impl<T, A> SharedInfo<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub const fn new() -> Self {
        SharedInfo {
            refc_: AtomicCount::new(AtomicUsize::new(0usize)),
            weak_: AtomexPtr::new(AtomicPtr::new(ptr::null_mut()) ),
        }
    }
}

#[repr(C)]
struct SharedInner<T, A>(Inner<A, SharedInfo<T, A>, T>)
where
    T: ?Sized,
    A: TrMalloc + Clone;

impl<T, A> SharedInner<T, A>
where
    A: TrMalloc + Clone,
{
    pub fn from_inner_ptr(
        inner: NonNull<Inner<A, SharedInfo<T, A>, T>>,
    ) -> NonNull<Self> {
        unsafe {
            let x = inner.as_ref();
            x.info().as_ptr().write(SharedInfo::new());
            let p = inner.as_ptr().cast::<SharedInner<T, A>>();
            NonNull::new_unchecked(p)
        }
    }
}

type PtrInnerWithSlice<T, A> = NonNull<Inner<A, SharedInfo<[T], A>, [T]>>;

impl<T, A> SharedInner<[T], A>
where
    A: TrMalloc + Clone,
{
    pub fn from_inner_with_slice(
        inner: PtrInnerWithSlice<T, A>,
        len: usize,
    ) -> NonNull<SharedInner<[T], A>> {
        unsafe {
            let x = inner.as_ref();
            x.info().as_ptr().write(SharedInfo::new());
            let p = NonNull::new_unchecked(inner.as_ptr() as *mut [u8]);
            let p = p.as_non_null_ptr().as_ptr();
            let p = ptr::slice_from_raw_parts_mut(p, len);
            NonNull::new_unchecked(p as *mut SharedInner<[T], A>)
        }
    }
}

impl<T, A> SharedInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    fn refc_(&self) -> &AtomicCount<usize> {
        unsafe { &self.0.info().as_ref().refc_ }
    }

    fn weak_(&self) -> &AtomexPtr<WeakInner<T, A>> {
        unsafe { &self.0.info().as_ref().weak_ }
    }

    #[inline(always)]
    const fn allocator(&self) -> &A {
        self.0.allocator()
    }

    #[inline(always)]
    const fn data_(&self) -> NonNull<T> {
        self.0.data()
    }

    fn release_(
        &mut self,
        may_drop: impl FnOnce(NonNull<T>),
    ) {
        let p_weak_inner = self.weak_().pointer();
        let opt_weak_inner = unsafe { p_weak_inner.as_mut() };
        if let Option::Some(weak_inner) = opt_weak_inner {
            if weak_inner.weak_count_usize() == 0 {
                WeakInner::release_weak_inner(weak_inner);
            } else {
                weak_inner.reset_back_track();
            }
        }
        self.0.release(may_drop);
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

type BackTrackSharedInner<T, A> = Option<NonNull<SharedInner<T, A>>>;

#[repr(C)]
struct WeakInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    /// A back track pointer for upgrading a Weak<T>. This pointer could be
    /// mutated to null when the back-tracking SharedInner instance is
    /// dropping.
    back_track_: UnsafeCell<BackTrackSharedInner<T, A>>,

    weak_count_: AtomicCount<usize>,
}

impl<T, A> WeakInner<T, A>
where
    T: ?Sized,
    A: TrMalloc + Clone,
{
    pub fn new(opt_shared_inner: BackTrackSharedInner<T, A>) -> Self {
        WeakInner {
            back_track_: UnsafeCell::new(opt_shared_inner),
            weak_count_: AtomicCount::default(),
        }
    }

    fn current_back_track(&self) -> BackTrackSharedInner<T, A> {
        unsafe { self.back_track_.get().read() }
    }

    fn reset_back_track(&mut self) {
        unsafe { self.back_track_.get().write(Option::None) }
    }

    unsafe fn allocate_weak_inner(
        shared_inner: BackTrackSharedInner<T, A>,
        alloc: &A,
    ) -> *mut WeakInner<T, A> {
        Owned::leak(alloc.owned(WeakInner::new(shared_inner)))
    }

    fn release_weak_inner(weak_inner: &mut WeakInner<T, A>) {
        unsafe { drop(Owned::<WeakInner<T, A>, A>::from_raw(weak_inner)) }
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
    use std::{ops::Deref, sync::Arc, vec::Vec};

    use core_malloc::CoreAlloc;
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
        let upgrade_result = weak.upgrade();
        assert!(upgrade_result.is_some());
        assert_eq!(Shared::strong_count(&shared), 2);

        drop(shared);
        drop(upgrade_result);
        assert!(arc_weak.upgrade().is_none());
        assert!(weak.upgrade().is_none())
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
            |u, m| m.write(Arc::new(u)),
            CoreAlloc::new(),
        );
        let arc_clone: Vec<_> = shared.iter().cloned().collect();
        let arc_weak: Vec<_> = shared.iter().map(Arc::downgrade).collect();

        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 0);

        let weak = Shared::downgrade(&shared);
        assert_eq!(Shared::strong_count(&shared), 1);
        assert_eq!(Shared::weak_count(&shared), 1);

        let upgrade_result = weak.upgrade();
        assert!(upgrade_result.is_some());
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
        assert!(r.is_ok());
        let arc = r.ok().unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(*arc, u128::MAX);
        let upgrade_result = arc_weak.upgrade();
        assert!(upgrade_result.is_some());
        assert_eq!(upgrade_result.unwrap(), arc);
    }
}
