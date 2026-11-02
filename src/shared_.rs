use core::{
    alloc::{AllocatorClone, AllocError, Layout, LayoutError},
    borrow::Borrow,
    cmp,
    convert::Infallible,
    error,
    fmt,
    marker::{PhantomData, PhantomPinned, Unsize},
    mem::{self, Alignment, MaybeUninit},
    ops::{CoerceUnsized, Deref, Try},
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize},
};

use abs_mm::res_man::{TrAllocBy, TrBoxed, TrStrongShared, TrWeakShared};
use atomex::{AtomexPtr, AtomicCount, CmpxchResult, TrAtomicFlags};

use crate::{
    alloc_utils_::handle_alloc_error_,
    owned_::{Owned, XtMallocOwned},
};

/// A atomic-reference-counted smart pointer for sharing the ownership of
/// resource in a multi-threaded environment.
///
/// This is not a drop-in replacement for `Arc<T>`. `Shared<T, A>` explicitly
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
pub struct Shared<T, A>(NonNull<SharedInner<T, A>>)
where
    T: ?Sized,
    A: AllocatorClone;

impl<T, A> Shared<T, A>
where
    T: Sized,
    A: AllocatorClone,
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

    pub fn try_into_inner(self) -> Result<T, Self> {
        const LAST_COUNT: usize = 1usize;
        const MAX_ATTEMPT: usize = 2usize;
        let mut opt_guard = Option::None;
        let mut i = 0usize;
        let racing_weak = self.racing_weak_().as_ptr();
        // Though `Shared::downgrade` accepts `&Shared<T>`, it is not possible when
        // strong count is 1 while other alias are still alive.
        // This is prevented by the compiler.
        // So it is not neccessary to acquire a guard from WeakInner backtrack.
        loop {
            if Shared::strong_count(&self) > LAST_COUNT {
                break;
            }
            let mut weak_inner_alive = false;
            let shared_inner = unsafe { self.0.as_ref() };
            if let Option::Some(p_weak_inner) = shared_inner.weak_().load() {
                // very unlikely in racing condition, only when this is poisoned.
                if p_weak_inner.as_ptr() == racing_weak {
                    if i > MAX_ATTEMPT {
                        break;
                    } else {
                        i += 1usize;
                        continue;
                    }
                }
                weak_inner_alive = true;
                let weak_inner = unsafe { p_weak_inner.as_ref() };
                if weak_inner.weak_count_usize() > 0usize {
                    opt_guard = weak_inner
                        .back_track_
                        .try_lock(shared_inner, MAX_ATTEMPT)
                        .ok();
                }
            }
            if opt_guard.is_some() || !weak_inner_alive {
                let mut this = MaybeUninit::new(self);
                let shared_inner = unsafe { &mut this.assume_init_mut().0.as_mut() };
                let data = unsafe {
                    let data = shared_inner.data_ptr().as_ptr().read();
                    MaybeUninit::new(data)
                };
                let strong_count = shared_inner.decrease_strong_count();
                debug_assert_eq!(strong_count, LAST_COUNT);
                let may_err = shared_inner.try_release_(|_| (), opt_guard);
                if may_err.is_none() {
                    return Result::Ok(unsafe { data.assume_init() });
                } else {
                    return Result::Err(unsafe { this.assume_init() });
                }
            }
        }
        Result::Err(self)
    }
}

impl<T, A> Shared<[T], A>
where
    T: Sized,
    A: AllocatorClone,
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
            Result::Err(_) => return Result::Err(AllocError),
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
    A: AllocatorClone,
{
    /// 构造一个长度为 `len` 的未初始化切片指针，元素类型为 `MaybeUninit<T>`。
    ///
    /// 对外形状对齐 `Arc::<[T]>::new_uninit_slice`：返回的指针拥有 `len` 个
    /// 未初始化的槽位，调用方必须逐个写入后才能把内容当作 `[T]` 使用。
    /// 注意 `Shared<T, A>` 只实现 `Deref` 而不实现 `DerefMut`，因此无法就地
    /// 写入槽位：需要写入时应改用带 emplace 闭包的
    /// [`new_slice`](Shared::new_slice)，或先用 [`try_leak`](Shared::try_leak)
    /// 取回可变引用。
    ///
    /// # Panics
    ///
    /// 当 `len` 个 `MaybeUninit<T>` 的布局计算溢出，或分配器无法满足该布局时，
    /// 经 `handle_alloc_error_` panic。
    ///
    /// # Examples
    ///
    /// ```
    /// use abs_mm::CoreAlloc;
    /// use core::mem::MaybeUninit;
    /// use mm_ptr::Shared;
    ///
    /// let x: Shared<[MaybeUninit<u32>], CoreAlloc> =
    ///     Shared::new_uninit_slice(3, CoreAlloc::default());
    /// assert_eq!(x.len(), 3);
    /// ```
    pub fn new_uninit_slice(len: usize, alloc: A) -> Self {
        Self::try_new_uninit_slice(len, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<[MaybeUninit<T>]>(e))
    }

    /// [`new_uninit_slice`](Self::new_uninit_slice) 的可失败版本。
    ///
    /// 对外形状对齐 `Arc::<[T]>::try_new_uninit_slice`：失败时返回
    /// `Err(AllocError)` 而不是 panic。
    ///
    /// # Errors
    ///
    /// 当 `len` 个 `MaybeUninit<T>` 的布局计算溢出（即 `Layout::array` 失败），
    /// 或分配器无法满足该布局时，返回 `Err(AllocError)`。
    ///
    /// # Examples
    ///
    /// ```
    /// use abs_mm::CoreAlloc;
    /// use core::mem::MaybeUninit;
    /// use mm_ptr::Shared;
    ///
    /// let x: Shared<[MaybeUninit<u32>], CoreAlloc> =
    ///     Shared::try_new_uninit_slice(3, CoreAlloc::default()).unwrap();
    /// assert_eq!(x.len(), 3);
    /// ```
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

    /// 构造一个长度为 `len` 的已零初始化切片指针，元素类型为 `MaybeUninit<T>`。
    ///
    /// 对外形状对齐 `Arc::<[T]>::new_zeroed_slice`：返回的指针拥有 `len` 个
    /// 全零字节的槽位；由于元素类型是 `MaybeUninit<T>`，不要求 `T` 具备任何
    /// “可零初始化”约束，某个零值对 `T` 是否合法仍由调用方判定。
    ///
    /// # Panics
    ///
    /// 当 `len` 个 `MaybeUninit<T>` 的布局计算溢出，或分配器无法满足该布局时，
    /// 经 `handle_alloc_error_` panic。
    ///
    /// # Examples
    ///
    /// ```
    /// use abs_mm::CoreAlloc;
    /// use core::mem::MaybeUninit;
    /// use mm_ptr::Shared;
    ///
    /// let x: Shared<[MaybeUninit<u32>], CoreAlloc> =
    ///     Shared::new_zeroed_slice(3, CoreAlloc::default());
    /// let read: Vec<u32> = x.iter().map(|m| unsafe { *m.assume_init_ref() }).collect();
    /// assert_eq!(read, [0, 0, 0]);
    /// ```
    pub fn new_zeroed_slice(len: usize, alloc: A) -> Self {
        Self::try_new_zeroed_slice(len, alloc)
            .unwrap_or_else(|e| handle_alloc_error_::<[MaybeUninit<T>]>(e))
    }

    /// [`new_zeroed_slice`](Self::new_zeroed_slice) 的可失败版本。
    ///
    /// 对外形状对齐 `Arc::<[T]>::try_new_zeroed_slice`：失败时返回
    /// `Err(AllocError)` 而不是 panic。
    ///
    /// # Errors
    ///
    /// 当 `len` 个 `MaybeUninit<T>` 的布局计算溢出（即 `Layout::array` 失败），
    /// 或分配器无法满足该布局时，返回 `Err(AllocError)`。
    ///
    /// # Examples
    ///
    /// ```
    /// use abs_mm::CoreAlloc;
    /// use core::mem::MaybeUninit;
    /// use mm_ptr::Shared;
    ///
    /// let x: Shared<[MaybeUninit<u32>], CoreAlloc> =
    ///     Shared::try_new_zeroed_slice(3, CoreAlloc::default()).unwrap();
    /// let read: Vec<u32> = x.iter().map(|m| unsafe { *m.assume_init_ref() }).collect();
    /// assert_eq!(read, [0, 0, 0]);
    /// ```
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
    A: AllocatorClone,
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
            .map_err(|_| AllocError)?;
        let Result::Ok(ptr) = alloc.allocate(layout) else {
            return Result::Err(AllocError);
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
    A: AllocatorClone,
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
    A: 'a + AllocatorClone,
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
    A: AllocatorClone,
{
    /// Allocate a header for weak pointers if it is not allocated, and then
    /// create a `Weak<T, A>` associated with the header.
    pub fn downgrade(shared: &Shared<T, A>) -> Weak<T, A> {
        let shared_inner = unsafe { shared.0.as_ref() };
        let atom_weak = shared_inner.weak_();
        let p_weak_inner: * mut WeakInner<T, A>;
        let racing_weak = shared.racing_weak_();
        loop {
            let r = atom_weak.try_spin_init(racing_weak);
            if let Result::Err(p) = r {
                // Judge if the WeakInner is locked or initialized by other.
                // If initialized by other, it is a pointer other than dangling
                if !ptr::eq(p.as_ptr(), racing_weak.as_ptr()) {
                    p_weak_inner = p.as_ptr();
                    break;
                }
            } else {
                p_weak_inner = WeakInner::allocate_weak_inner(shared_inner);
                atom_weak.store(p_weak_inner);
                break;
            }
        }
        assert!(!p_weak_inner.is_null() && p_weak_inner != racing_weak.as_ptr());
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

    /// Create an impossible WeakInner pointer to indicate that the WeakInner
    /// pointer is under racing conditioni.
    fn racing_weak_(&self) -> NonNull<WeakInner<T, A>> {
        unsafe {
            NonNull::new_unchecked(self.shared_inner_()
                as *const _
                as *mut SharedInner<T, A>
                as *mut WeakInner<T, A>
            )
        }
    }
}

impl<T, A> Drop for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    fn drop(&mut self) {
        let shared_inner = unsafe { self.0.as_mut() };
        if shared_inner.decrease_strong_count() > 1 {
            return;
        }
        let may_drop = |p: NonNull<T>| unsafe { ptr::drop_in_place(p.as_ptr()) };
        loop {
            let opt_failure = shared_inner.try_release_(may_drop, Option::None);
            let Option::Some(failure) = opt_failure else {
                return;
            };
            if matches!(failure, ReleaseFailure::Resurrected(_)) {
                return;
            }
        }
    }
}

impl<T, A> Clone for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    fn clone(&self) -> Self {
        Shared::from_shared_inner_(self.shared_inner_())
    }
}

impl<T, U, A> CoerceUnsized<Shared<U, A>> for Shared<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
    A: AllocatorClone,
{}

impl<T, A> Deref for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().data_ptr().as_ref() }
    }
}

impl<T, A> Borrow<T> for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T, A> PartialEq for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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
    A: AllocatorClone,
{}

impl<T, A> PartialEq<Weak<T, A>> for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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
    A: AllocatorClone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ptr().fmt(f)
    }
}

/// 为 `Shared<T, A>` 提供不依赖类型参数的调试输出。
///
/// 该实现不要求 `T: Debug`，也不要求 `A: Debug`，任何 `Shared<T, A>` 自身
/// 都无条件地实现 `fmt::Debug`；输出形式为元组结构体风格，字段是内部指针地址。
///
/// # Examples
///
/// ```
/// use abs_mm::CoreAlloc;
/// use mm_ptr::{Shared, XtMallocShared};
///
/// // 刻意没有实现 `Debug` 的类型。
/// struct NotDebug;
///
/// let x: Shared<NotDebug, CoreAlloc> = CoreAlloc::default().shared(NotDebug);
/// assert!(format!("{x:?}").starts_with("Shared(0x"));
/// ```
impl<T, A> fmt::Debug for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Shared").field(&self.0).finish()
    }
}

impl<T, A> fmt::Display for Shared<T, A>
where
    T: ?Sized + fmt::Display,
    A: AllocatorClone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T, A> TrAllocBy for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    type Alloc = A;

    fn mem_alloc(&self) -> &Self::Alloc {
        self.malloc()
    }
}

impl<T, A> TrBoxed for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    type Item = T;
}

impl<T, A> TrStrongShared for Shared<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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

// SAFETY: `Shared<T, A>` 允许在多线程环境中共享，因此它与 `Arc<T, A>` 的约束
// 必须一致。
// - `T: Send + Sync`：`T` 的析构可能发生在任意线程（`Shared::drop` 最终会
//   `drop_in_place` 载荷），故需要 `T: Send`；而 `Deref` 与 `Clone` 允许并发地
//   由 `&Shared` 取得 `&T`，故需要 `T: Sync`。
// - `A: Send`：`SharedInner<T, A>` 内嵌了一份 `A`，`Shared` 被移动到其他线程后
//   会在那个线程调用 `A::deallocate`，`Shared::downgrade` 也会在那个线程
//   `clone` 出 `A` 来分配 `WeakInner`，故需要 `A: Send`。
// - `A: Sync`：`Shared::malloc` 会从 `&self` 交出 `&A`，而 `Shared` 可在多个线程
//   上被同时 `clone`/`downgrade`，故需要 `A: Sync` 才能并发共享 `&A`。
unsafe impl<T, A> Send for Shared<T, A>
where
    T: ?Sized + Send + Sync,
    A: AllocatorClone + Send + Sync,
{}

// SAFETY: 与 `Send` 实现同理；`Sync` 意味着 `&Shared<T, A>` 本身可跨线程共享，
// 于是 `&T` 与 `&A` 都可能被多个线程并发访问，同样要求 `T: Send + Sync` 与
// `A: Send + Sync`。
unsafe impl<T, A> Sync for Shared<T, A>
where
    T: ?Sized + Send + Sync,
    A: AllocatorClone + Send + Sync,
{}

/// The weak pointer of Shared<T, A>. This is a different implementation from
/// `Arc<T>`'s weak. `Weak<T, A>` has its independent control block, so that the
/// existence of weak pointers will not prevent reclaiming memory used by `T` in
/// `Shared<T, A>`.
pub struct Weak<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    inner_: Option<NonNull<WeakInner<T, A>>>,
}

/// 为 `Weak<T, A>` 提供不依赖类型参数的调试输出。
///
/// 该实现不要求 `T: Debug`，也不要求 `A: Debug`，任何 `Weak<T, A>` 自身
/// 都无条件地实现 `fmt::Debug`；输出形式与派生实现一致，为结构体字段形式
/// `Weak { inner_: ... }`。
///
/// # Examples
///
/// ```
/// use abs_mm::CoreAlloc;
/// use mm_ptr::{Shared, XtMallocShared};
///
/// // 刻意没有实现 `Debug` 的类型。
/// struct NotDebug;
///
/// let shared: Shared<NotDebug, CoreAlloc> = CoreAlloc::default().shared(NotDebug);
/// let weak = Shared::downgrade(&shared);
/// assert!(format!("{weak:?}").starts_with("Weak { inner_: Some(0x"));
/// ```
impl<T, A> fmt::Debug for Weak<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weak")
            .field("inner_", &self.inner_)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WeakUpgradeError {
    /// The instance owned by the `Shared<T>` pointer is destructing.
    Destructing,

    /// The instance owned by the `Shared<T>` is memory freed.
    Destroyed,

    /// Racing condition
    Contention,

    /// Fatal error, should panic.
    Unknown,
}

impl fmt::Display for WeakUpgradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeakUpgradeError::Destructing => write!(f, "WeakUpgradeError::Destructing"),
            WeakUpgradeError::Destroyed => write!(f, "WeakUpgradeError::Destroyed"),
            WeakUpgradeError::Contention => write!(f, "WeakUpgradeError::Contention"),
            WeakUpgradeError::Unknown => write!(f, "WeakUpgradeError::Unknown"),
        }
    }
}

impl error::Error for WeakUpgradeError
{ }

impl<T, A> Weak<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    pub const fn new() -> Self {
        Weak { inner_: Option::None }
    }

    pub fn try_upgrade(&self) -> Result<Shared<T, A>, WeakUpgradeError> {
        const MAX_ATTEMPT: usize = 2;

        let Option::Some(weak_inner) = self.opt_weak_inner_() else {
            return Result::Err(WeakUpgradeError::Unknown);
        };
        let p_shared_pointer = weak_inner.ptr_shared_.as_ptr();
        let res = weak_inner
            .back_track_
            .try_lock(p_shared_pointer, MAX_ATTEMPT);
        let Result::Ok(guard) = res else {
            return Result::Err(WeakUpgradeError::Contention);
        };
        let shared_inner = guard.shared_inner();
        if !ptr::eq(shared_inner, p_shared_pointer) {
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
        let p_shared_pointer = weak_inner.ptr_shared_.as_ptr();
        let res = weak_inner
            .back_track_
            .try_lock(p_shared_pointer, MAX_ATTEMPT);
        let Result::Ok(guard) = res else {
            return Option::None;
        };
        // critical section is guarded from here
        let shared_inner = guard.shared_inner();
        if !ptr::eq(shared_inner, p_shared_pointer) {
            return Option::None;
        };
        let c = Option::Some(shared_inner.strong_count());
        drop(guard);
        // critical section ends
        c
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
    A: AllocatorClone,
{
    fn drop(&mut self) {
        let Option::Some(mut p_inner) = self.inner_ else {
            return;
        };
        const LAST_COUNT: usize = 1usize;
        let weak_inner = unsafe { p_inner.as_mut() };
        if weak_inner.weak_count_usize() != LAST_COUNT {
            weak_inner.decrease_weak_count();
            return;
        }
        // 极端情况下，当 SharedInner 和 WeakInner 同时引用数降为 0 时有可能导致
        // WeakInner 泄露，所以必须等待一个确定信号来释放 WeakInner 即 backtrack
        // 被 reset。
        {
            let re = weak_inner.back_track_.try_read(usize::MAX);
            let Result::Err(p) = re else {
                return; // SharedInner 还是存活的，所以由它来释放
            };
            if weak_inner.weak_count_usize() != LAST_COUNT {
                return; // 可能有其他 Shared::downgrade()
            }
            debug_assert!(
                p.is_null(),
                "[Weak<T, A>::Drop] assumes WeakInner should drop after SharedInner is released.",
            );
            weak_inner.decrease_weak_count();
            WeakInner::release_weak_inner(weak_inner);
            // return;
        }
    }
}

impl<T, A> Clone for Weak<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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
    A: AllocatorClone,
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
    A: AllocatorClone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized, A: AllocatorClone> cmp::PartialEq for Weak<T, A> {
    fn eq(&self, other: &Self) -> bool {
        match (self.inner_, other.inner_) {
            (Option::Some(this), Option::Some(that))
                => ptr::eq(this.as_ptr(), that.as_ptr()),
            (Option::None, Option::None) => true,
            _ => false,
        }
    }
}

impl<T: ?Sized, A: AllocatorClone> cmp::Eq for Weak<T, A> {}

impl<T: ?Sized, A: AllocatorClone> cmp::PartialEq<Shared<T, A>> for Weak<T, A> {
    #[inline]
    fn eq(&self, other: &Shared<T, A>) -> bool {
        <Shared<T, A> as PartialEq<Weak<T, A>>>::eq(other, self)
    }
}

impl<T: ?Sized, A: AllocatorClone> fmt::Pointer for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Option::Some(p) = self.inner_ else {
            return ptr::null::<WeakInner<T, A>>().fmt(f);
        };
        p.as_ptr().fmt(f)
    }
}

// SAFETY: `Weak<T, A>` 与 `Shared<T, A>` 一样允许在多线程环境中共享，其约束与
// `Arc<T, A>::Weak` 一致。
// - `T: Send + Sync`：升级/降级不访问 `T` 本身，但 `Weak` 可被跨线程持有，
//   且它与 `Shared<T, A>` 相互转换，故沿用 `Shared<T, A>` 对 `T` 的要求。
// - `A: Send`：`WeakInner` 由 `A` 分配，并在 `Weak` 的最后一次析构时经
//   `WeakInner::release_weak_inner`（`Owned<WeakInner<T, A>, A>`）在该线程释放，
//   `Weak::allocator` 也会交出内部的 `A`，故需要 `A: Send`。
// - `A: Sync`：`Weak::allocator` 从 `&self` 交出 `&A`，`&Weak` 可跨线程共享，
//   故需要 `A: Sync`。
unsafe impl<T, A> Send for Weak<T, A>
where
    T: ?Sized + Send + Sync,
    A: AllocatorClone + Send + Sync,
{}

// SAFETY: 与 `Send` 实现同理；`Sync` 意味着 `&Weak<T, A>` 可跨线程共享，
// `&A` 可能被并发访问，故同样要求 `T: Send + Sync` 与 `A: Send + Sync`。
unsafe impl<T, A> Sync for Weak<T, A>
where
    T: ?Sized + Send + Sync,
    A: AllocatorClone + Send + Sync,
{}

pub trait XtMallocShared
where
    Self: AllocatorClone,
{
    fn shared<T>(&self, t: T) -> Shared<T, Self>;
}

impl<A> XtMallocShared for A
where
    A: AllocatorClone,
{
    fn shared<T>(&self, t: T) -> Shared<T, Self> {
        Shared::<T, A>::new(t, self.clone())
    }
}

#[repr(C)]
struct SharedInner<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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
    A: AllocatorClone,
{
    #[allow(clippy::mut_from_ref)]
    fn refc_(&self) -> AtomicCount<usize, &mut AtomicUsize> {
        let this_mut = self as *const _ as *mut Self;
        unsafe { AtomicCount::new(&mut (*this_mut).refc_) }
    }

    #[allow(clippy::mut_from_ref)]
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

    #[inline]
    fn sentinel_(shared_inner_ptr: *const Self) -> *mut A {
        let offset = mem::offset_of!(SharedInner<T, A>, alloc_);
        let p = unsafe { shared_inner_ptr.byte_add(offset) };
        p as *const A as *mut A
    }

    const fn allocator(&self) -> &A {
        &self.alloc_
    }

    const fn data_ptr(&self) -> NonNull<T> {
        let this_mut = self as *const _ as *mut Self;
        unsafe { NonNull::new_unchecked(&mut (*this_mut).data_) }
    }

    // /// Calculate the address of SharedInner<T, A> from a pointer to its `alloc_` field
    // #[inline]
    // unsafe fn from_p_alloc_(
    //     p_alloc: *const A,
    //     data: *const T,
    // ) -> *const SharedInner<T, A> {
    //     let offset_alloc = mem::offset_of!(SharedInner<T, A>, alloc_);
    //     let offset_data = mem::offset_of!(SharedInner<(), A>, data_);

    //     debug_assert!(offset_data >= offset_alloc);
    //     debug_assert_eq!(offset_alloc, 0usize);
    //     debug_assert_eq!(mem::size_of::<SharedInner<(), A>>(), offset_data);

    //     let p_inner = unsafe { p_alloc.byte_sub(offset_alloc) };
    //     // let data = unsafe { p_alloc.byte_add(offset_data - offset_alloc) };
    //     let metadata = ptr::metadata(data);
    //     ptr::from_raw_parts(p_inner, metadata)
    // }

    fn try_release_(
        &mut self,
        may_drop: impl FnOnce(NonNull<T>),
        mut opt_guard: Option<BackTrackGuard<'_, T, A>>,
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
            let mut weak_inner_alive = false;
            if let Option::Some(weak_inner) = opt_weak_inner {
                weak_inner_alive = true;
                if weak_inner.weak_count_usize() == 0 {
                    WeakInner::release_weak_inner(weak_inner);
                } else {
                    if opt_guard.is_none() {
                        opt_guard = weak_inner
                            .back_track_
                            .try_lock(self, MAX_ATTEMPT)
                            .ok();
                        continue;
                    }
                }
            };
            if weak_inner_alive && opt_guard.is_none() {
                return Option::Some(ReleaseFailure::BacktrackBusy);
            }
            let alloc = self.allocator().clone();
            let layout = Layout::for_value(self);
            may_drop(self.data_ptr());

            let ptr = unsafe { NonNull::new_unchecked(this) };
            unsafe { alloc.deallocate(ptr, layout) };

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
#[allow(dead_code)]
enum ReleaseFailure {
    /// During attempt to release, an upgrade has succeeded and the strong count
    /// has increased.
    Resurrected(usize),

    /// Cannot acquire back track guard
    BacktrackBusy,
}

/// 在 WeakInner 中反向追踪 SharedInner 的原子指针。有三种可能的状态，
/// 空值：SharedInner 已释放
/// dangling_val_：指向自身地址 `BackTrackSharedInner`，用于表示锁已被获取。
/// 以及真正的 `SharedInner` 地址
///
/// 由于原子指针不能是胖指针，所以变通地使用 `&shared_inner_.alloc_` 来作为
/// 声明类型值
struct BackTrackSharedInner<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    pointer_: AtomexPtr<A>,
    _mark_t_: PhantomData<SharedInner<T, A>>,
}

impl<T, A> BackTrackSharedInner<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    pub const fn new(p: &SharedInner<T, A>) -> Self {
        let pointer = AtomicPtr::new(&p.alloc_ as *const A as *mut A);
        BackTrackSharedInner {
            pointer_: AtomexPtr::new(pointer),
            _mark_t_: PhantomData,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.pointer_.load().is_none()
    }

    /// To calculate the address of WeakInner from the backtrack.
    ///
    /// # Safety
    /// This will only work when ALL backtrack instances are the field of WeakInner.
    unsafe fn weak_inner_(&self) -> &WeakInner<T, A> {
        let offset = mem::offset_of!(WeakInner<T, A>, back_track_);
        let this = self as *const Self;
        let inner = unsafe { this.byte_sub(offset) as *const WeakInner<T, A> };
        unsafe { inner.as_ref().unwrap() }
    }

    /// 尝试读取 backtrack 中的地址，如果是正常的 SharedInner 地址就返回 Ok
    /// 超出最大尝试或者是其他的值是返回 Err
    pub fn try_read(
        &self,
        max_attempt: usize,
    ) -> Result<&SharedInner<T, A>, *mut A> {
        let offset_alloc = mem::offset_of!(SharedInner<T, A>, alloc_);
        let locked_val = self.dangling_val_();
        let weak_inner = unsafe { self.weak_inner_() };
        let p_shared_inner = weak_inner
            .ptr_shared_
            .as_ptr();
        let p_shared_inner_thin = p_shared_inner as *mut u8;

        let mut current;
        let mut c = 0usize;
        loop {
            current = self.pointer_.pointer();
            if current.is_null() {
                break;
            }
            c += 1usize;
            if current == locked_val {
                continue;
            }
            let deduced_p = unsafe { current.byte_sub(offset_alloc) as usize };
            if deduced_p == p_shared_inner_thin as usize {
                let shared_inner = unsafe { &*p_shared_inner };
                return Result::Ok(shared_inner)
            }
            if c >= max_attempt {
                break;
            }
        }
        Result::Err(current)
    }

    /// Try to acquire the backtrack guard by offering a correct reference to
    /// the SharedInner. This would work because when the guard is available
    /// the backtrack pointer stores the address of the `alloc_` field of the
    /// `SharedInner` instance.
    pub fn try_lock<'a>(
        &'a self,
        p_shared_inner: *const SharedInner<T, A>,
        max_attempt: usize,
    ) -> Result<BackTrackGuard<'a, T, A>, Option<CmpxchResult<*mut A>>> {
        let current = SharedInner::sentinel_(p_shared_inner);
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
                unsafe {
                    let shared_inner = &*p_shared_inner;
                    let tk = BackTrackGuard::new(self, shared_inner);
                    return Result::Ok(tk);
                }
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
    A: AllocatorClone,
{
    back_track_: &'a BackTrackSharedInner<T, A>,
    shared_inner_: Option<NonNull<SharedInner<T, A>>>,
}

impl<'a, T, A> BackTrackGuard<'a, T, A>
where
    T: ?Sized,
    A: AllocatorClone,
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
    A: AllocatorClone,
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
    A: AllocatorClone,
{
    _pinned_: PhantomPinned,

    /// A back track pointer for upgrading a Weak<T>. This pointer could be
    /// mutated to null when the back-tracking SharedInner instance is
    /// dropping.
    back_track_: BackTrackSharedInner<T, A>,
    weak_count_: AtomicCount<usize>,
    ptr_shared_: NonNull<SharedInner<T, A>>,
}

impl<T, A> WeakInner<T, A>
where
    T: ?Sized,
    A: AllocatorClone,
{
    pub const fn new(shared_inner: &SharedInner<T, A>) -> Self {
        let p = shared_inner as *const _ as *mut SharedInner<T, A>;
        let p = unsafe { NonNull::new_unchecked(p) };
        WeakInner {
            _pinned_: PhantomPinned,
            back_track_: BackTrackSharedInner::new(shared_inner),
            weak_count_: AtomicCount::new(AtomicUsize::new(0usize)),
            ptr_shared_: p,
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
    use core::{
        alloc::{Allocator, AllocatorClone},
        ops::Deref,
    };

    use abs_mm::CoreAlloc;

    use super::{Shared, Weak, XtMallocShared};

    /// 用于验证 `Debug` 实现不依赖 `T: Debug` 的载荷类型，刻意不派生 `Debug`。
    struct NotDebug;

    /// 测试 `Shared<T, A>` 在载荷类型未实现 `Debug` 时依然实现 `fmt::Debug`。
    /// - 手段：以刻意未实现 `Debug` 的 `NotDebug` 作为载荷、`CoreAlloc` 作为分配器，
    ///   构造 `Shared<NotDebug, CoreAlloc>`，再用 `format!("{x:?}")` 格式化该指针；
    ///   同时在测试体内定义一个只声明 `T: ?Sized`、`A: AllocatorClone` 的泛型函数，
    ///   在其函数体中把 `Shared<T, A>` 交给仅接受 `Debug` 类型的断言函数。
    /// - 判断：格式化结果以 `"Shared(0x"` 开头，说明输出内容是内部指针地址；
    ///   泛型函数体在编译期仅依据已声明约束检查，能通过即证明 `fmt::Debug`
    ///   实现未隐藏 `T: Debug` / `A: Debug` 约束，若将来被加上则此处编译失败。
    #[test]
    fn debug_should_work_without_t_debug_() {
        fn assert_debug<T: ?Sized + core::fmt::Debug>() {}
        fn assert_debug_bounds_are_unconditional<T: ?Sized, A: AllocatorClone>() {
            assert_debug::<Shared<T, A>>();
        }
        assert_debug_bounds_are_unconditional::<NotDebug, CoreAlloc>();

        let x: Shared<NotDebug, CoreAlloc> = CoreAlloc::new().shared(NotDebug);
        let displayed = std::format!("{x:?}");
        assert!(
            displayed.starts_with("Shared(0x"),
            "unexpected debug output: {displayed}",
        );
    }

    /// 测试 `Weak<T, A>` 在载荷类型未实现 `Debug` 时依然实现 `fmt::Debug`。
    /// - 手段：以刻意未实现 `Debug` 的 `NotDebug` 作为载荷、`CoreAlloc` 作为分配器，
    ///   构造 `Shared<NotDebug, CoreAlloc>` 并降级得到 `Weak<NotDebug, CoreAlloc>`，
    ///   再用 `format!("{w:?}")` 格式化该弱指针；同时在测试体内定义一个只声明
    ///   `T: ?Sized`、`A: AllocatorClone` 的泛型函数，在其函数体中把 `Weak<T, A>`
    ///   交给仅接受 `Debug` 类型的断言函数。
    /// - 判断：格式化结果以 `"Weak { inner_: Some(0x"` 开头，即与派生实现输出一致；
    ///   泛型函数体在编译期仅依据已声明约束检查，能通过即证明 `fmt::Debug`
    ///   实现未隐藏 `T: Debug` / `A: Debug` 约束，若将来被加上则此处编译失败。
    #[test]
    fn debug_weak_should_work_without_t_debug_() {
        fn assert_debug<T: ?Sized + core::fmt::Debug>() {}
        fn assert_debug_bounds_are_unconditional<T: ?Sized, A: AllocatorClone>() {
            assert_debug::<Weak<T, A>>();
        }
        assert_debug_bounds_are_unconditional::<NotDebug, CoreAlloc>();

        let x: Shared<NotDebug, CoreAlloc> = CoreAlloc::new().shared(NotDebug);
        let weak = Shared::downgrade(&x);
        let displayed = std::format!("{weak:?}");
        assert!(
            displayed.starts_with("Weak { inner_: Some(0x"),
            "unexpected debug output: {displayed}",
        );
    }

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
        let p: Shared<dyn Allocator, CoreAlloc> =
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

    /// 测试 `Shared<T, A>` 与 `Weak<T, A>` 的 `Send`/`Sync` 约束形状与 `Arc<T, A>` 对齐。
    /// - 手段：定义只声明 `T: ?Sized + Send + Sync`、`A: AllocatorClone + Send + Sync`
    ///   的泛型函数，在其中把 `Shared<T, A>` 与 `Weak<T, A>` 交给只接受 `Send`
    ///   或只接受 `Sync` 的断言函数；再以 `CoreAlloc` 实例化该泛型函数。
    /// - 判断：该泛型函数体仅凭上述边界即可通过编译，说明这两个 impl 的约束不严于
    ///   `Arc<T, A>` 的对应 impl（没有诸如 `A: 'static` 之类的额外要求）；若将来
    ///   有人给 impl 添加超出此处的约束，本测试会编译失败。约束的下界（即
    ///   `A: !Send` 或 `A: !Sync` 时不得实现 `Send`/`Sync`）无法用正向断言表达，
    ///   由 impl 的 `where` 子句与 `// SAFETY:` 注释共同保证。
    #[test]
    fn send_sync_bounds_should_match_arc_() {
        fn assert_send<T: ?Sized + Send>() {}
        fn assert_sync<T: ?Sized + Sync>() {}
        fn check<T: ?Sized + Send + Sync, A: AllocatorClone + Send + Sync>() {
            assert_send::<Shared<T, A>>();
            assert_sync::<Shared<T, A>>();
            assert_send::<Weak<T, A>>();
            assert_sync::<Weak<T, A>>();
        }
        check::<u8, CoreAlloc>();
    }
}
