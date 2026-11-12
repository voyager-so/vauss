//! Packed geometry/opacity Adam moments: one u32 word per value holding a
//! 16-bit normalized first moment `m / (sqrt(v) + e)` and a 16-bit log-scaled
//! `ln(1 + sqrt(v) / e)` (codec in `kernels::geo_moments`), 4 bytes instead
//! of two f32 moments. [`project_update`] steps them in place; these helpers
//! convert to and from f32 and reset rows for refinement. A zero word is
//! exactly `m = v = 0`.
//!
//! [`project_update`]: super::project_update::project_update

use brush_cube::create_tensor;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Dispatch, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Device, Int, IntDType, Shape, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;

use super::kernels::geo_moments::{decode_geo_kernel, encode_geo_kernel, zero_geo_rows_kernel};

/// Word dtype; any 32-bit integer works for the kernels.
pub const GEO_WORD_DTYPE: DType = DType::U32;

mod ops {
    #![allow(unused_variables)]
    use super::{DType, GEO_WORD_DTYPE, Shape, TensorSpec};
    use burn::backend::{
        Backend,
        tensor::{FloatTensor, IntTensor},
    };

    #[burn::backend::backend_extension(Cube, Fusion)]
    pub trait GeoMomentOps: Backend {
        /// Pack f32 `(m, v)` into one word per value, rounding to nearest.
        #[fusion(meta = |m, _v, _epsilon| TensorSpec::new(Shape::new([m.shape.num_elements()]), GEO_WORD_DTYPE))]
        fn encode_geo(
            m: FloatTensor<Self>,
            v: FloatTensor<Self>,
            #[fusion(scalar)] epsilon: f32,
        ) -> IntTensor<Self>;

        #[fusion(meta = |words, _epsilon| {
            let spec = TensorSpec::new(Shape::new([words.shape.num_elements()]), DType::F32);
            (spec.clone(), spec)
        })]
        fn decode_geo(
            words: IntTensor<Self>,
            #[fusion(scalar)] epsilon: f32,
        ) -> (FloatTensor<Self>, FloatTensor<Self>);

        /// Zero the unique `rows` (`row_width` words each) in place.
        #[fusion(meta = |words, _rows, _row_width| words.clone())]
        fn zero_geo_rows(
            words: IntTensor<Self>,
            rows: IntTensor<Self>,
            row_width: u32,
        ) -> IntTensor<Self>;
    }
}
use ops::GeoMomentOps;

impl GeoMomentOps for CubeBackend {
    fn encode_geo(m: FloatTensor<Self>, v: FloatTensor<Self>, epsilon: f32) -> IntTensor<Self> {
        let (m, v) = (into_contiguous(m), into_contiguous(v));
        for t in [&m, &v] {
            assert_eq!(t.dtype, DType::F32, "moments decode to f32");
        }
        let n = m.shape().num_elements();
        assert_eq!(v.shape().num_elements(), n);
        let words = create_tensor([n], &m.device, GEO_WORD_DTYPE);
        if n == 0 {
            return words;
        }
        let dim = CubeDim::new_1d(256);
        let count = calculate_cube_count_elemwise(&m.client, n, dim);
        encode_geo_kernel::launch(
            &m.client,
            count,
            dim,
            m.clone().into_tensor_arg(),
            v.into_tensor_arg(),
            words.clone().into_tensor_arg(),
            epsilon,
        );
        words
    }

    fn decode_geo(words: IntTensor<Self>, epsilon: f32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let words = into_contiguous(words);
        let n = words.shape().num_elements();
        let m = create_tensor([n], &words.device, DType::F32);
        let v = create_tensor([n], &words.device, DType::F32);
        if n == 0 {
            return (m, v);
        }
        let dim = CubeDim::new_1d(256);
        let count = calculate_cube_count_elemwise(&words.client, n, dim);
        decode_geo_kernel::launch(
            &words.client,
            count,
            dim,
            words.clone().into_tensor_arg(),
            m.clone().into_tensor_arg(),
            v.clone().into_tensor_arg(),
            epsilon,
        );
        (m, v)
    }

    fn zero_geo_rows(
        words: IntTensor<Self>,
        rows: IntTensor<Self>,
        row_width: u32,
    ) -> IntTensor<Self> {
        let words = into_contiguous(words);
        let words = if words.can_mut() { words } else { words.copy() };
        let rows = into_contiguous(rows);
        assert_eq!(rows.dtype, DType::U32, "row indices must be u32");
        let n = rows.shape().num_elements() * row_width as usize;
        if n == 0 {
            return words;
        }
        let dim = CubeDim::new_1d(256);
        let count = calculate_cube_count_elemwise(&words.client, n, dim);
        zero_geo_rows_kernel::launch(
            &words.client,
            count,
            dim,
            words.clone().into_tensor_arg(),
            rows.into_tensor_arg(),
            row_width,
        );
        words
    }
}

/// All-zero words (`m = v = 0`).
pub fn zeros<const D: usize>(shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    Tensor::zeros(shape, (device, GEO_WORD_DTYPE))
}

/// Round-to-nearest encode of f32 moments; decode-then-encode is idempotent.
pub fn encode<const D: usize>(m: Tensor<D>, v: Tensor<D>, epsilon: f32) -> Tensor<D, Int> {
    let shape = m.dims();
    assert_eq!(v.dims(), shape);
    let words = <Dispatch as GeoMomentOps>::encode_geo(m.into_dispatch(), v.into_dispatch(), epsilon);
    Tensor::<1, Int>::from_dispatch(words).reshape(shape)
}

/// f32 `(m, v)` of packed words.
pub fn decode<const D: usize>(words: Tensor<D, Int>, epsilon: f32) -> (Tensor<D>, Tensor<D>) {
    let shape = words.dims();
    let (m, v) = <Dispatch as GeoMomentOps>::decode_geo(words.into_dispatch(), epsilon);
    (
        Tensor::<1>::from_dispatch(m).reshape(shape),
        Tensor::<1>::from_dispatch(v).reshape(shape),
    )
}

/// Reset the unique `rows` to zero moments in place.
pub fn zero_rows<const D: usize>(words: Tensor<D, Int>, rows: Tensor<1, Int>) -> Tensor<D, Int> {
    let shape = words.dims();
    let row_width = u32::try_from(shape[1..].iter().product::<usize>()).expect("row width exceeds u32");
    let rows = if rows.dtype() == DType::U32 { rows } else { rows.cast(IntDType::U32) };
    let out = <Dispatch as GeoMomentOps>::zero_geo_rows(words.into_dispatch(), rows.into_dispatch(), row_width);
    Tensor::<D, Int>::from_dispatch(out)
}
