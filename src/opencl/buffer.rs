use std::marker::PhantomData;
use std::mem;

use crate::opencl::error::*;

pub struct Buffer<T: Sized> {
    pub(crate) buffer: ocl::Buffer<u8>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: Sized> Buffer<T> {
    pub fn length(&self) -> usize {
        self.buffer.len() / mem::size_of::<T>()
    }

    pub fn write_from(&mut self, offset: usize, data: &[T]) -> GPUResult<()> {
        let size_of_t = mem::size_of::<T>();
        let data_size = data.len() * size_of_t;

        assert_eq!(data_size, mem::size_of_val(data));
        assert!(offset + data.len() <= self.length());

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const _, data.len() * mem::size_of::<T>())
        };
        self.buffer
            .create_sub_buffer(None, offset * size_of_t, data_size)?
            .write(bytes)
            .enq()?;
        Ok(())
    }

    pub fn read_into(&self, offset: usize, data: &mut [T]) -> GPUResult<()> {
        let size_of_t = mem::size_of::<T>();
        let data_size = data.len() * size_of_t;

        assert_eq!(data_size, mem::size_of_val(data));
        assert!(offset + data.len() <= self.length());

        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut _, data_size) };
        self.buffer
            .create_sub_buffer(None, offset * size_of_t, data_size)?
            .read(bytes)
            .enq()?;
        Ok(())
    }
}
