use core::{marker::PhantomData, ops::Deref};

use abs_mm::res_man::TrReclaim;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoReclaim<P, R>(PhantomData<P>, PhantomData<R>)
where
    P: Deref<Target = R>,
    R: ?Sized;

impl<P, R> TrReclaim<R> for NoReclaim<P, R>
where
    P: Deref<Target = R>,
    R: ?Sized,
{
    type Reclaimable = P;

    fn reclaim(self, reclaimable: &mut Self::Reclaimable) {
        let _ = reclaimable;
    }
}

pub struct Reclaim<F, P, R>
where
    F: FnOnce(&mut P),
    P: Deref<Target = R>,
    R: ?Sized,
{
    _use_ptr_: PhantomData<P>,
    _use_res_: PhantomData<R>,
    reclaim_: Option<F>,
}

impl<F, P, R> Reclaim<F, P, R>
where
    F: FnOnce(&mut P),
    P: Deref<Target = R>,
    R: ?Sized,
{
    pub const fn new(reclaim: F) -> Self {
        Reclaim {
            reclaim_: Option::Some(reclaim),
            _use_ptr_: PhantomData,
            _use_res_: PhantomData,
        }
    }

    #[inline]
    pub fn reclaim(mut self, reclaimable: &mut P) {
        Self::reclaim_(&mut self.reclaim_, reclaimable);
    }

    fn reclaim_(r: &mut Option<F>, p: &mut P) {
        if let Option::Some(recl) = self.reclaim_.take() {
            recl(p)
        }
    }
}

impl<F, P, R> TrReclaim<R> for Reclaim<F, P, R>
where
    F: FnOnce(&mut P),
    P: Deref<Target = R>,
    R: ?Sized,
{
    type Reclaimable = P;

    #[inline]
    fn reclaim(self, reclaimable: &mut Self::Reclaimable) {
        Reclaim::reclaim(self, reclaimable);
    }
}
