use std::ptr::NonNull;

use datafusion::arrow::{array::{BooleanArray, ArrayData}, buffer::Buffer, datatypes::DataType};


#[inline]
pub(crate) fn _build_boolean_array(mut res: Vec<u8>, array_len: usize) -> BooleanArray {
    let value_buffer = unsafe {
        let buf = Buffer::from_raw_parts(NonNull::new_unchecked(res.as_mut_ptr()), res.len(), res.capacity());
        std::mem::forget(res);
        buf
    };
    let builder = ArrayData::builder(DataType::Boolean)
        .len(array_len)
        .add_buffer(value_buffer);
    let array_data = builder.build().unwrap();
    BooleanArray::from(array_data)
}