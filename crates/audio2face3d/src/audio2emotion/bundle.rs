//! Closed, non-generic owning bundles for completed emotion facades.

use std::sync::Arc;

use crate::Result;
#[cfg(feature = "tensorrt")]
use crate::audio2emotion::classifier::{
    ClassifierEmotionExecutor, ClassifierEmotionExecutorCreationParameters,
    ClassifierEmotionExecutorFactory,
};
use crate::audio2emotion::post_process::{
    PostProcessEmotionExecutor, PostProcessEmotionExecutorCreationParameters,
    PostProcessEmotionExecutorFactory,
};
use crate::audio2x::{AudioAccumulator, EmotionAccumulator, ExecutorFuture};
use crate::cuda::CudaStream;

/// Completed emotion executor borrowed from an owning bundle.
pub enum EmotionExecutorRef<'a> {
    #[cfg(feature = "tensorrt")]
    Classifier(&'a ClassifierEmotionExecutor),
    PostProcess(&'a PostProcessEmotionExecutor),
}

/// Mutably borrowed completed emotion executor.
pub enum EmotionExecutorMut<'a> {
    #[cfg(feature = "tensorrt")]
    Classifier(&'a mut ClassifierEmotionExecutor),
    PostProcess(&'a mut PostProcessEmotionExecutor),
}

/// Model-specific parameters accepted by [`EmotionExecutorBundleFactory::load`].
pub enum EmotionExecutorBundleCreationParameters {
    #[cfg(feature = "tensorrt")]
    Classifier(ClassifierEmotionExecutorCreationParameters),
    PostProcess(PostProcessEmotionExecutorCreationParameters),
}

/// A closed, non-generic owning emotion bundle.
pub enum EmotionExecutorBundle {
    #[cfg(feature = "tensorrt")]
    Classifier(Box<ClassifierEmotionExecutor>),
    PostProcess(Box<PostProcessEmotionExecutor>),
}

impl EmotionExecutorBundle {
    pub fn executor(&self) -> EmotionExecutorRef<'_> {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::Classifier(executor) => EmotionExecutorRef::Classifier(executor),
            Self::PostProcess(executor) => EmotionExecutorRef::PostProcess(executor),
        }
    }

    pub fn executor_mut(&mut self) -> EmotionExecutorMut<'_> {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::Classifier(executor) => EmotionExecutorMut::Classifier(executor),
            Self::PostProcess(executor) => EmotionExecutorMut::PostProcess(executor),
        }
    }

    pub fn cuda_stream(&self) -> &CudaStream {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::Classifier(executor) => executor.cuda_stream(),
            Self::PostProcess(executor) => executor.cuda_stream(),
        }
    }

    pub fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::Classifier(executor) => executor.audio_accumulator(track),
            Self::PostProcess(executor) => executor.audio_accumulator(track),
        }
    }

    pub fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::Classifier(executor) => executor.emotion_accumulator(track),
            Self::PostProcess(executor) => executor.emotion_accumulator(track),
        }
    }
}

/// Runtime-independent factory for completed emotion bundles.
pub struct EmotionExecutorBundleFactory;

impl EmotionExecutorBundleFactory {
    pub fn load(
        parameters: EmotionExecutorBundleCreationParameters,
    ) -> ExecutorFuture<'static, EmotionExecutorBundle> {
        Box::pin(async move {
            match parameters {
                #[cfg(feature = "tensorrt")]
                EmotionExecutorBundleCreationParameters::Classifier(parameters) => {
                    ClassifierEmotionExecutorFactory::load(parameters)
                        .await
                        .map(Box::new)
                        .map(EmotionExecutorBundle::Classifier)
                }
                EmotionExecutorBundleCreationParameters::PostProcess(parameters) => {
                    PostProcessEmotionExecutorFactory::load(parameters)
                        .await
                        .map(Box::new)
                        .map(EmotionExecutorBundle::PostProcess)
                }
            }
        })
    }

    #[cfg(feature = "tensorrt")]
    pub fn classifier(
        parameters: ClassifierEmotionExecutorCreationParameters,
    ) -> ExecutorFuture<'static, EmotionExecutorBundle> {
        Self::load(EmotionExecutorBundleCreationParameters::Classifier(
            parameters,
        ))
    }

    pub fn post_process(
        parameters: PostProcessEmotionExecutorCreationParameters,
    ) -> ExecutorFuture<'static, EmotionExecutorBundle> {
        Self::load(EmotionExecutorBundleCreationParameters::PostProcess(
            parameters,
        ))
    }
}

#[cfg(feature = "tensorrt")]
pub fn create_classifier_bundle(
    parameters: ClassifierEmotionExecutorCreationParameters,
) -> ExecutorFuture<'static, EmotionExecutorBundle> {
    EmotionExecutorBundleFactory::classifier(parameters)
}

pub fn create_post_process_bundle(
    parameters: PostProcessEmotionExecutorCreationParameters,
) -> ExecutorFuture<'static, EmotionExecutorBundle> {
    EmotionExecutorBundleFactory::post_process(parameters)
}

#[cfg(feature = "cuda")]
impl EmotionExecutorBundleFactory {
    pub fn load_with_context(
        parameters: EmotionExecutorBundleCreationParameters,
        context: crate::Audio2Face3DContext,
    ) -> ExecutorFuture<'static, EmotionExecutorBundle> {
        let scope = crate::logging::integration::LogScope::new(context);
        Box::pin(scope.wrap_future(scope.in_scope(|| Self::load(parameters))))
    }
}
