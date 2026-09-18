//! Budget units include owned Vec/String capacities. Shared layouts are charged per event.
//! Map nodes use a conservative per-entry allowance, not an allocator/RSS measurement.
use crate::client::types::*;
use std::mem::size_of;
fn map(values: &std::collections::BTreeMap<String, f32>) -> usize {
    values.keys().fold(0usize, |sum, name| {
        sum.saturating_add(name.capacity()).saturating_add(1024)
    })
}
fn keys(values: &Vec<EmotionKeyframe>) -> usize {
    values.iter().fold(
        values
            .capacity()
            .saturating_mul(size_of::<EmotionKeyframe>()),
        |n, k| n.saturating_add(map(k.values())),
    )
}
fn layout(layout: &CurveLayout) -> usize {
    layout.storage_bytes()
}
pub(crate) fn request(options: &RequestOptions) -> usize {
    let mut n = size_of::<RequestOptions>();
    if let Some(b) = &options.blendshapes {
        n = n
            .saturating_add(map(&b.multipliers))
            .saturating_add(map(&b.offsets));
    }
    if let Some(e) = &options.emotion {
        n = n.saturating_add(map(&e.beginning));
    }
    n
}
pub(crate) fn input(chunk: InputChunk) -> Result<(InputChunk, usize)> {
    let (pcm, emotions) = chunk.into_parts();
    let pcm = pcm.into_vec();
    let n = size_of::<InputChunk>()
        .saturating_add(pcm.capacity())
        .saturating_add(keys(&emotions));
    Ok((InputChunk::new(PcmBuffer::from_vec(pcm)?, emotions), n))
}
pub(crate) fn output(event: OutputEvent) -> Result<(OutputEvent, usize)> {
    let (event, heap) = match event {
        OutputEvent::Audio(audio) => {
            let (format, position, pcm) = audio.into_parts();
            let pcm = pcm.into_vec();
            let n = pcm.capacity();
            (
                OutputEvent::Audio(AudioBlock::new(
                    format,
                    position,
                    PcmBuffer::from_vec(pcm)?,
                )?),
                n,
            )
        }
        OutputEvent::Curves(curve) => {
            let (names, time, values) = curve.into_parts();
            let n = values
                .capacity()
                .saturating_mul(size_of::<f32>())
                .saturating_add(layout(&names));
            (
                OutputEvent::Curves(CurveFrame::new(names, time, values)?),
                n,
            )
        }
        OutputEvent::Emotion(trace) => {
            let n = keys(&trace.input)
                .saturating_add(keys(&trace.mixed))
                .saturating_add(keys(&trace.smoothed));
            (OutputEvent::Emotion(trace), n)
        }
        OutputEvent::StreamInfo(info) => {
            let n = info.curves.as_ref().map_or(0, |v| layout(v));
            (OutputEvent::StreamInfo(info), n)
        }
        OutputEvent::Diagnostic(d) => {
            let n = d.message.capacity();
            (OutputEvent::Diagnostic(d), n)
        }
        OutputEvent::Completed(_) => {
            return Err(Error::new(
                ErrorKind::Protocol,
                "Completed is owned by the session core",
            ));
        }
        OutputEvent::ProcessingFinished => (OutputEvent::ProcessingFinished, 0),
    };
    Ok((event, size_of::<OutputEvent>().saturating_add(heap)))
}
