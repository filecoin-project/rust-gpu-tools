#[cfg(feature = "cuda")]
use crate::cuda;
use crate::error::GPUError;
#[cfg(feature = "opencl")]
use crate::opencl;

/// Abstraction for running programs on CUDA or OpenCL.
pub enum Program {
    /// CUDA program.
    #[cfg(feature = "cuda")]
    Cuda(cuda::Program),
    /// OpenCL program.
    #[cfg(feature = "opencl")]
    Opencl(opencl::Program),
}

impl Program {
    /// Run some code in the context of the program.
    ///
    /// There is an implementation for OpenCL and for CUDA. Both use different Rust types, but
    /// [`opencl::Program`] and [`cuda::Program`] implement the same API. This means that same
    /// code code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in two closures, one for OpenCL, one for CUDA, both get their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating two closures out of a single one.
    ///
    /// CUDA and OpenCL support can be enabled/disabled by the `opencl` and `cuda` features. If
    /// one of them is disabled, you still need to pass in two closures. This way the API stays,
    /// the same, but you can disable it things at compile-time.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", feature = "opencl"))]
    pub fn run<F1, F2, R, E, A>(&self, fun: (F1, F2), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&cuda::Program, A) -> Result<R, E>,
        F2: FnOnce(&opencl::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Cuda(program) => program.run(fun.0, arg),
            Self::Opencl(program) => program.run(fun.1, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There is an implementation for OpenCL and for CUDA. Both use different Rust types, but
    /// [`opencl::Program`] and [`cuda::Program`] implement the same API. This means that same
    /// code code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in two closures, one for OpenCL, one for CUDA, both get their
    /// corresponding program type as parameter. For convenience there is the [`program_closures`]
    /// macro defined, which can help reducing code duplication by creating two closures out of
    /// a single one.
    ///
    /// CUDA and OpenCL support can be enabled/disabled by the `opencl` and `cuda` features. If
    /// one of them is disabled, you still need to pass in two closures. This way the API stays,
    /// the same, but you can disable it things at compile-time.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", not(feature = "opencl")))]
    pub fn run<F1, F2, R, E, A>(&self, fun: (F1, F2), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&cuda::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Cuda(program) => program.run(fun.0, arg),
        }
    }

    /// Run some code in the context of the program.
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
    ///
    /// CUDA and OpenCL support can be enabled/disabled by the `opencl` and `cuda` features. If
    /// one of them is disabled, you still need to pass in two closures. This way the API stays,
    /// the same, but you can disable it things at compile-time.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(not(feature = "cuda"), feature = "opencl"))]
    pub fn run<F1, F2, R, E, A>(&self, fun: (F1, F2), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F2: FnOnce(&opencl::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Opencl(program) => program.run(fun.1, arg),
        }
    }

    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn device_name(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(program) => program.device_name(),
            #[cfg(feature = "opencl")]
            Self::Opencl(program) => program.device_name(),
        }
    }
}

/// Creates two closures, one for CUDA, one for OpenCL for the given one.
///
/// This macro is used to be able to interact with rust-gpu-tools with unified code for both,
/// CUDA and OpenCL, without the need to repeat the code. The input parameter is a `program` and
/// it will be mapped to &[`cuda::Program`] and &[`opencl::Program`].
///
/// The second parameter is a single arbitrary argument, which will be passed on into the closure.
/// This is useful when you e.g. need to pass in a mutable reference. Such a reference cannot be
/// shared between closures, hence we pass it on, so that the compiler knows that it is used at
/// most once.
///
/// Depending on whether the `cuda` and/or `opencl` feature is enabled, it will do the correct
/// thing and not specify one of them if it is appropriate.
///
/// ### Example
///
/// ```
/// use rust_gpu_tools::{cuda, opencl, program_closures};
///
/// let closures = program_closures!(|program, arg: u8| -> bool {
///     true
/// });
///
/// // Generates
/// let closures = (
///     |program: &cuda::Program, arg: u8| { true },
///     |program: &opencl::Program, arg: u8| { true },
/// );
///
/// // If e.g. the `cuda` feature is disabled, it would generate
/// let closures_without_cuda = (
///     (),
///     |program: &opencl::Program, arg: u8| { true },
/// );
/// ```
#[cfg(all(feature = "cuda", feature = "opencl"))]
#[macro_export]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg| -> $ret { $body },
            |$program: &$crate::opencl::Program, $arg| -> $ret { $body },
        )
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg: $arg_type| -> $ret { $body },
            |$program: &$crate::opencl::Program, $arg: $arg_type| -> $ret { $body },
        )
    };
}

/// Creates two closures, one for CUDA, one for OpenCL for the given one.
///
/// This macro is used to be able to interact with rust-gpu-tools with unified code for both,
/// CUDA and OpenCL, without the need to repeat the code. The input parameter is a `program` and
/// it will be mapped to [`&cuda::Program`] and [`&opencl::Program`].
///
/// The second parameter is a single arbitrary argument, which will be passed on into the closure.
/// This is useful when you e.g. need to pass in a mutable reference. Such a reference cannot be
/// shared between closures, hence we pass it on, so that the compiler knows that it is used at
/// most once.
///
/// Depending on whether the `cuda` and/or `opencl` feature is enabled, it will do the correct
/// thing and not specify one of them if it is appropriate.
///
/// ### Example
///
/// ```
/// use rust_gpu_tools::{cuda, opencl, program_closures};
///
/// let closures = program_closures!(|program, arg: u8| -> bool {
///     true
/// });
///
/// // Generates
/// let closures = (
///     |program: &cuda::Program, arg: u8| { true },
///     |program: &opencl::Program, arg: u8| { true },
/// );
///
/// // If e.g. the `cuda` feature is disabled, it would generate
/// let closures_without_cuda = (
///     (),
///     |program: &opencl::Program, arg: u8| { true },
/// );
/// ```
#[macro_export]
#[cfg(all(feature = "cuda", not(feature = "opencl")))]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg| -> $ret { $body },
            (),
        )
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg: $arg_type| -> $ret { $body },
            (),
        )
    };
}

/// Creates two closures, one for CUDA, one for OpenCL for the given one.
///
/// This macro is used to be able to interact with rust-gpu-tools with unified code for both,
/// CUDA and OpenCL, without the need to repeat the code. The input parameter is a `program` and
/// it will be mapped to [`&cuda::Program`] and [`&opencl::Program`].
///
/// The second parameter is a single arbitrary argument, which will be passed on into the closure.
/// This is useful when you e.g. need to pass in a mutable reference. Such a reference cannot be
/// shared between closures, hence we pass it on, so that the compiler knows that it is used at
/// most once.
///
/// Depending on whether the `cuda` and/or `opencl` feature is enabled, it will do the correct
/// thing and not specify one of them if it is appropriate.
///
/// ### Example
///
/// ```
/// use rust_gpu_tools::{cuda, opencl, program_closures};
///
/// let closures = program_closures!(|program, arg: u8| -> bool {
///     true
/// });
///
/// // Generates
/// let closures = (
///     |program: &cuda::Program, arg: u8| { true },
///     |program: &opencl::Program, arg: u8| { true },
/// );
///
/// // If e.g. the `cuda` feature is disabled, it would generate
/// let closures_without_cuda = (
///     (),
///     |program: &opencl::Program, arg: u8| { true },
/// );
/// ```
#[macro_export]
#[cfg(all(not(feature = "cuda"), feature = "opencl"))]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        ((), |$program: &$crate::opencl::Program, $arg| -> $ret {
            $body
        })
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            (),
            |$program: &$crate::opencl::Program, $arg: $arg_type| -> $ret { $body },
        )
    };
}
