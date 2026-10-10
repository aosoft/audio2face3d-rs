//! Immutable PCM blocks shared by the controller and the audio player.
use super::{Error, Result};
use std::sync::Arc;

pub const MAX_AUDIO_BLOCKS: usize = 65_536;

#[derive(Debug)]
pub(crate) struct AudioBlock {
    start: usize,
    samples: Box<[f32]>,
}
impl AudioBlock {
    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct AudioTrack {
    blocks: Vec<Arc<AudioBlock>>,
    samples: usize,
}

impl AudioTrack {
    pub fn len(&self) -> usize {
        self.samples
    }
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.blocks.iter().flat_map(|block| block.samples.iter())
    }
    pub fn to_vec(&self) -> Vec<f32> {
        self.iter().copied().collect()
    }

    pub fn get(&self, index: usize) -> Option<&f32> {
        let mut cursor = self
            .blocks
            .partition_point(|block| block.start <= index)
            .checked_sub(1)?;
        self.sample_at(index, &mut cursor)
    }

    pub(crate) fn sample_at(&self, index: usize, cursor: &mut usize) -> Option<&f32> {
        if index >= self.samples {
            return None;
        }
        let contains =
            |block: &AudioBlock| index >= block.start && index - block.start < block.samples.len();
        if !self
            .blocks
            .get(*cursor)
            .is_some_and(|block| contains(block))
        {
            if self
                .blocks
                .get(cursor.saturating_add(1))
                .is_some_and(|block| contains(block))
            {
                *cursor += 1;
            } else {
                *cursor = self
                    .blocks
                    .partition_point(|block| block.start <= index)
                    .saturating_sub(1);
            }
        }
        let block = self.blocks.get(*cursor)?;
        block.samples.get(index.checked_sub(block.start)?)
    }

    pub(crate) fn block_cost(samples: usize) -> usize {
        if samples == 0 {
            return 0;
        }
        // Payload once; include both owners' handle arrays and the Arc allocation.
        samples * std::mem::size_of::<f32>()
            + std::mem::size_of::<AudioBlock>()
            + 2 * std::mem::size_of::<Arc<AudioBlock>>()
            + 2 * std::mem::size_of::<usize>()
    }

    pub(crate) fn push(&mut self, samples: Box<[f32]>) {
        if samples.is_empty() {
            return;
        }
        let block = Arc::new(AudioBlock {
            start: self.samples,
            samples,
        });
        self.samples += block.samples.len();
        self.blocks.push(block);
    }

    pub(crate) fn shared_from(&self, first: usize) -> Result<Vec<Arc<AudioBlock>>> {
        self.blocks
            .get(first..)
            .map(|blocks| blocks.to_vec())
            .ok_or_else(|| Error("invalid audio block cursor".into()))
    }

    pub(crate) fn append_shared(&mut self, blocks: Vec<Arc<AudioBlock>>) -> Result<()> {
        let mut end = self.samples;
        if self.blocks.len().saturating_add(blocks.len()) > MAX_AUDIO_BLOCKS {
            return Err(Error("audio block limit exceeded".into()));
        }
        for block in &blocks {
            if block.start != end {
                return Err(Error("noncontiguous playback audio blocks".into()));
            }
            end += block.samples.len();
        }
        self.blocks.extend(blocks);
        self.samples = end;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn shares_payload_with(&self, other: &Self) -> bool {
        self.blocks.len() == other.blocks.len()
            && self
                .blocks
                .iter()
                .zip(&other.blocks)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

impl PartialEq for AudioTrack {
    fn eq(&self, other: &Self) -> bool {
        self.samples == other.samples && self.iter().eq(other.iter())
    }
}

impl serde::Serialize for AudioTrack {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(self.samples))?;
        for sample in self.iter() {
            sequence.serialize_element(sample)?;
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_handles_keep_one_payload_and_flat_serialization() {
        let mut audio = AudioTrack::default();
        audio.push(vec![0.1, 0.2].into_boxed_slice());
        audio.push(vec![0.3].into_boxed_slice());
        let mut playback = AudioTrack::default();
        playback
            .append_shared(audio.shared_from(0).unwrap())
            .unwrap();
        assert!(audio.shares_payload_with(&playback));
        assert_eq!(playback.get(1), Some(&0.2));
        assert_eq!(playback.get(2), Some(&0.3));
        assert_eq!(playback.get(3), None);
        assert_eq!(
            serde_json::to_value(&playback).unwrap(),
            serde_json::to_value(vec![0.1_f32, 0.2, 0.3]).unwrap()
        );
    }
}
