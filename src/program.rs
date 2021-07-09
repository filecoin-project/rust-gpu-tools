use crate::cuda;
use crate::error::GPUError;
use crate::opencl;

pub enum Program {
    Opencl(opencl::Program),
    Cuda(cuda::Program),
}

impl Program {
    /// Run some code in the context of the program
    ///
    /// There is an implementation for OpenCL and for CUDA. Both use different Rust types, but
    /// [`opencl::Program`] and [`cuda::Program`] implement the same API. This means that same
    /// code code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in two closures, one for OpenCL, one for CUDA, both get their
    /// corresponding program type as parameter. For convenience there is the [`define_closures`]
    /// macro defined, which can help reducing code duplication by creating two closures out of
    /// a single one.
    pub fn run<F1, F2, R, E>(&self, fun: (F1, F2)) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&opencl::Program) -> Result<R, E>,
        F2: FnOnce(&cuda::Program) -> Result<R, E>,
    {
        match self {
            Self::Opencl(program) => program.run(fun.0),
            Self::Cuda(program) => program.run(fun.1),
        }
    }

    pub fn device_name(&self) -> &str {
        match self {
            Self::Opencl(program) => program.device_name(),
            Self::Cuda(program) => program.device_name(),
        }
    }
}

/// Creates a tuple where each element is a closure where the parameter type is one of the given
/// ones. The parameter name and the closure body is the same.
///
/// Example:
///
/// ```
/// use rust_gpu_tools::define_closures;
///
/// let closures = define_closures!(|program: String | u8| -> bool {
///     true
/// });
///
/// // Generates
/// let closures = (
///     |program: String| { true },
///     |program: u8| { true },
/// );
/// ```
#[macro_export]
macro_rules! define_closures {
    (| $var:ident: $($type:ty)|+ | -> $ret:ty $body:block) => {
       (
           $(
               |$var: $type| -> $ret { $body },
           )+
       )
    };
}
