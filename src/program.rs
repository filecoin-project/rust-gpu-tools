#[cfg(feature = "cuda")]
use crate::cuda;
use crate::error::GPUError;
#[cfg(feature = "metal")]
use crate::metal;
#[cfg(feature = "opencl")]
use crate::opencl;

/// Abstraction for running programs on CUDA, OpenCL, or Metal.
pub enum Program {
    /// CUDA program.
    #[cfg(feature = "cuda")]
    Cuda(cuda::Program),
    /// OpenCL program.
    #[cfg(feature = "opencl")]
    Opencl(opencl::Program),
    /// Metal program.
    #[cfg(feature = "metal")]
    Metal(metal::Program),
}

impl Program {
    /// Run some code in the context of the program.
    ///
    /// There are implementations for OpenCL, CUDA, and Metal. All three use different Rust types, but
    /// [`opencl::Program`], [`cuda::Program`], and [`metal::Program`] implement the same API. This means that same
    /// code can be used to run on any of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// CUDA, OpenCL, and Metal support can be enabled/disabled by the respective features. If
    /// one of them is disabled, you still need to pass in the correct tuple of closures. This way the API stays
    /// the same, but you can disable things at compile-time.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", feature = "opencl", feature = "metal"))]
    pub fn run<F1, F2, F3, R, E, A>(&self, fun: (F1, F2, F3), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&cuda::Program, A) -> Result<R, E>,
        F2: FnOnce(&opencl::Program, A) -> Result<R, E>,
        F3: FnOnce(&metal::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Cuda(program) => program.run(fun.0, arg),
            Self::Opencl(program) => program.run(fun.1, arg),
            Self::Metal(program) => program.run(fun.2, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There are implementations for OpenCL and CUDA. Both use different Rust types, but
    /// [`opencl::Program`] and [`cuda::Program`] implement the same API. This means that same
    /// code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", feature = "opencl", not(feature = "metal")))]
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
    /// There are implementations for CUDA and Metal. Both use different Rust types, but
    /// [`cuda::Program`] and [`metal::Program`] implement the same API. This means that same
    /// code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", feature = "metal", not(feature = "opencl")))]
    pub fn run<F1, F3, R, E, A>(&self, fun: (F1, F3), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&cuda::Program, A) -> Result<R, E>,
        F3: FnOnce(&metal::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Cuda(program) => program.run(fun.0, arg),
            Self::Metal(program) => program.run(fun.1, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There are implementations for OpenCL and Metal. Both use different Rust types, but
    /// [`opencl::Program`] and [`metal::Program`] implement the same API. This means that same
    /// code can be used to run on either of them. The only difference is the type of the
    /// `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "opencl", feature = "metal", not(feature = "cuda")))]
    pub fn run<F2, F3, R, E, A>(&self, fun: (F2, F3), arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F2: FnOnce(&opencl::Program, A) -> Result<R, E>,
        F3: FnOnce(&metal::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Opencl(program) => program.run(fun.0, arg),
            Self::Metal(program) => program.run(fun.1, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There is an implementation for CUDA. This means that same code can be used to run on it.
    /// The only difference is the type of the `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(feature = "cuda", not(feature = "opencl"), not(feature = "metal")))]
    pub fn run<F1, R, E, A>(&self, fun: F1, arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F1: FnOnce(&cuda::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Cuda(program) => program.run(fun, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There is an implementation for OpenCL. This means that same code can be used to run on it.
    /// The only difference is the type of the `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(not(feature = "cuda"), feature = "opencl", not(feature = "metal")))]
    pub fn run<F2, R, E, A>(&self, fun: F2, arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F2: FnOnce(&opencl::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Opencl(program) => program.run(fun, arg),
        }
    }

    /// Run some code in the context of the program.
    ///
    /// There is an implementation for Metal. This means that same code can be used to run on it.
    /// The only difference is the type of the `Program`.
    ///
    /// You need to pass in a tuple of closures, one for each supported backend, each getting their
    /// corresponding program type as parameter. For convenience there is the
    /// [`crate::program_closures`] macro defined, which can help reducing code duplication by
    /// creating closures out of a single one.
    ///
    /// The second parameter is a single arbitrary argument, which will be passed on into the
    /// closure. This is useful when you e.g. need to pass in a mutable reference. Such a reference
    /// cannot be shared between closures, hence we pass it on, so that the compiler knows that it
    /// is used at most once.
    #[cfg(all(not(feature = "cuda"), not(feature = "opencl"), feature = "metal"))]
    pub fn run<F3, R, E, A>(&self, fun: F3, arg: A) -> Result<R, E>
    where
        E: From<GPUError>,
        F3: FnOnce(&metal::Program, A) -> Result<R, E>,
    {
        match self {
            Self::Metal(program) => program.run(fun, arg),
        }
    }

    /// Returns the name of the GPU, e.g. "GeForce RTX 3090" or "Apple M1 Pro".
    pub fn device_name(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(program) => program.device_name(),
            #[cfg(feature = "opencl")]
            Self::Opencl(program) => program.device_name(),
            #[cfg(feature = "metal")]
            Self::Metal(program) => program.device_name(),
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
#[cfg(all(feature = "cuda", feature = "opencl", feature = "metal"))]
#[macro_export]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg| -> $ret { $body },
            |$program: &$crate::opencl::Program, $arg| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg| -> $ret { $body },
        )
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg: $arg_type| -> $ret { $body },
            |$program: &$crate::opencl::Program, $arg: $arg_type| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg: $arg_type| -> $ret { $body },
        )
    };
}

#[cfg(all(feature = "cuda", feature = "opencl", not(feature = "metal")))]
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
#[cfg(all(feature = "cuda", feature = "metal", not(feature = "opencl")))]
#[macro_export]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg| -> $ret { $body },
        )
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::cuda::Program, $arg: $arg_type| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg: $arg_type| -> $ret { $body },
        )
    };
}

#[cfg(all(feature = "opencl", feature = "metal", not(feature = "cuda")))]
#[macro_export]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::opencl::Program, $arg| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg| -> $ret { $body },
        )
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        (
            |$program: &$crate::opencl::Program, $arg: $arg_type| -> $ret { $body },
            |$program: &$crate::metal::Program, $arg: $arg_type| -> $ret { $body },
        )
    };
}

#[macro_export]
#[cfg(all(feature = "cuda", not(feature = "opencl"), not(feature = "metal")))]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        |$program: &$crate::cuda::Program, $arg| -> $ret { $body }
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        |$program: &$crate::cuda::Program, $arg: $arg_type| -> $ret { $body }
    };
}

/// Creates a closure for OpenCL.
///
/// This macro is used to be able to interact with rust-gpu-tools with unified code,
/// without the need to repeat the code. The input parameter is a `program` and
/// it will be mapped to [`&opencl::Program`].
///
/// The second parameter is a single arbitrary argument, which will be passed on into the closure.
/// This is useful when you e.g. need to pass in a mutable reference.
///
/// Depending on which features are enabled, this macro will generate the appropriate closures.
///
/// ### Example
///
/// ```
/// use rust_gpu_tools::{cuda, opencl, metal, program_closures};
///
/// let closures = program_closures!(|program, arg: u8| -> bool {
///     true
/// });
/// ```
#[macro_export]
#[cfg(all(not(feature = "cuda"), feature = "opencl", not(feature = "metal")))]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        |$program: &$crate::opencl::Program, $arg| -> $ret { $body }
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        |$program: &$crate::opencl::Program, $arg: $arg_type| -> $ret { $body }
    };
}

/// Creates a closure for Metal.
///
/// This macro is used to be able to interact with rust-gpu-tools with unified code,
/// without the need to repeat the code. The input parameter is a `program` and
/// it will be mapped to [`&metal::Program`].
///
/// The second parameter is a single arbitrary argument, which will be passed on into the closure.
/// This is useful when you e.g. need to pass in a mutable reference.
///
/// Depending on which features are enabled, this macro will generate the appropriate closures.
///
/// ### Example
///
/// ```
/// use rust_gpu_tools::{cuda, opencl, metal, program_closures};
///
/// let closures = program_closures!(|program, arg: u8| -> bool {
///     true
/// });
/// ```
#[macro_export]
#[cfg(all(not(feature = "cuda"), not(feature = "opencl"), feature = "metal"))]
macro_rules! program_closures {
    // Additional argument without a type
    (|$program:ident, $arg:ident| -> $ret:ty $body:block) => {
        |$program: &$crate::metal::Program, $arg| -> $ret { $body }
    };
    // Additional argument with a type
    (|$program:ident, $arg:ident: $arg_type:ty| -> $ret:ty $body:block) => {
        |$program: &$crate::metal::Program, $arg: $arg_type| -> $ret { $body }
    };
}
