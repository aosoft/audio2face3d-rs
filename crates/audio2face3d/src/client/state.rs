//! Request transitions under the request lock. Returned buffers and effects
//! must be released or executed only after that lock has been dropped.
use super::{Limits, core::Buffered, types::*};
use std::{collections::VecDeque, task::Poll};

#[derive(Default, PartialEq, Eq)]
enum InputPhase {
    #[default]
    Open,
    FinishRequested,
    EofObserved,
}
#[derive(Default)]
enum Outcome {
    #[default]
    Running,
    Success,
    Failed(Error),
}
#[derive(Default)]
enum Cleanup {
    #[default]
    Pending,
    Complete(Result<()>),
}
#[derive(Default, PartialEq, Eq)]
enum Delivery {
    #[default]
    Receiving,
    TerminalConsumed,
}
#[derive(Default, PartialEq, Eq)]
enum Registration {
    #[default]
    Registered,
    Retired,
}
#[derive(Default, PartialEq, Eq)]
enum ProcessingNotice {
    #[default]
    NotSeen,
    Seen,
}

#[derive(Default)]
pub(super) struct Effects {
    // Ownership, rather than a deferred closure, keeps Lease destruction out
    // of the state lock. The request applies notifications separately.
    pub discarded_input: VecDeque<Buffered<InputChunk>>,
    pub pending_input: Option<Buffered<InputChunk>>,
    pub discarded_output: VecDeque<Buffered<OutputEvent>>,
    pub pending_output: Option<Buffered<OutputEvent>>,
    pub retire: bool,
    pub notify: bool,
    pub poke: bool,
}

pub(super) enum Received {
    Event(Buffered<OutputEvent>),
    Complete(Summary),
}

#[derive(Default)]
pub(super) struct State {
    input: VecDeque<Buffered<InputChunk>>,
    pending_input: Option<Buffered<InputChunk>>,
    input_bytes: usize,
    output: VecDeque<Buffered<OutputEvent>>,
    pending_output: Option<Buffered<OutputEvent>>,
    output_bytes: usize,
    input_phase: InputPhase,
    outcome: Outcome,
    cleanup: Cleanup,
    delivery: Delivery,
    registration: Registration,
    progress: Progress,
    processing_notice: ProcessingNotice,
}

impl State {
    pub fn error(&self) -> Option<Error> {
        match &self.outcome {
            Outcome::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }
    pub fn progress(&self) -> Progress {
        self.progress
    }
    pub fn closed(&self) -> Poll<Result<()>> {
        match &self.cleanup {
            Cleanup::Pending => Poll::Pending,
            Cleanup::Complete(result) => Poll::Ready(result.clone()),
        }
    }
    fn ensure_open(&self) -> Result<()> {
        if let Some(error) = self.error() {
            return Err(error);
        }
        if self.input_phase != InputPhase::Open {
            return Err(Error::invalid("input already finished"));
        }
        Ok(())
    }
    pub fn input_capacity(&self, bytes: usize, limits: &Limits) -> bool {
        self.input.len() < limits.input_queue_items
            && bytes <= limits.input_queue_bytes - self.input_bytes
    }
    pub fn check_input(&self, bytes: usize, limits: &Limits) -> Result<()> {
        self.ensure_open()?;
        if !self.input_capacity(bytes, limits) {
            return Err(Error::new(ErrorKind::QueueFull, "input queue full"));
        }
        Ok(())
    }
    pub fn stage_input(&mut self, item: &mut Option<Buffered<InputChunk>>) -> Result<()> {
        self.ensure_open()?;
        assert!(self.pending_input.is_none());
        self.pending_input = item.take();
        Ok(())
    }
    pub fn enqueue_input(&mut self, item: Buffered<InputChunk>) {
        self.input_bytes += item.bytes();
        self.input.push_back(item);
    }
    pub fn accept_input(&mut self, limits: &Limits) -> Poll<Result<()>> {
        if let Some(error) = self.error() {
            return Poll::Ready(Err(error));
        }
        let bytes = self
            .pending_input
            .as_ref()
            .expect("send polled after completion")
            .bytes();
        if !self.input_capacity(bytes, limits) {
            return Poll::Pending;
        }
        let item = self.pending_input.take().unwrap();
        self.enqueue_input(item);
        Poll::Ready(Ok(()))
    }
    pub fn abandon_input(&mut self) -> Option<Buffered<InputChunk>> {
        self.pending_input.take()
    }
    pub fn finish_input(&mut self) -> Result<()> {
        if let Some(error) = self.error() {
            return Err(error);
        }
        if self.input_phase == InputPhase::Open {
            self.input_phase = InputPhase::FinishRequested;
        }
        Ok(())
    }
    pub fn receive_input(&mut self) -> Poll<Result<Option<Buffered<InputChunk>>>> {
        if let Some(error) = self.error() {
            return Poll::Ready(Err(error));
        }
        if let Some(item) = self.input.pop_front() {
            self.input_bytes -= item.bytes();
            return Poll::Ready(Ok(Some(item)));
        }
        if self.input_phase != InputPhase::Open {
            self.input_phase = InputPhase::EofObserved;
            return Poll::Ready(Ok(None));
        }
        Poll::Pending
    }
    pub fn stage_output(&mut self, item: &mut Option<Buffered<OutputEvent>>) -> Result<()> {
        if let Some(error) = self.error() {
            return Err(error);
        }
        if !matches!(self.outcome, Outcome::Running) {
            return Err(Error::new(
                ErrorKind::Protocol,
                "output after backend completion",
            ));
        }
        let event = &item.as_ref().unwrap().value;
        if matches!(event, OutputEvent::ProcessingFinished) {
            if self.processing_notice == ProcessingNotice::Seen {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "duplicate processing-finished event",
                ));
            }
            self.processing_notice = ProcessingNotice::Seen;
        }
        match event {
            OutputEvent::Audio(audio) => {
                self.progress.received_audio_bytes += audio.pcm().as_bytes().len() as u64
            }
            OutputEvent::Curves(_) => self.progress.received_curve_frames += 1,
            _ => {}
        }
        assert!(self.pending_output.is_none());
        self.pending_output = item.take();
        Ok(())
    }
    pub fn accept_output(&mut self, limits: &Limits) -> Poll<Result<()>> {
        if let Some(error) = self.error() {
            return Poll::Ready(Err(error));
        }
        let bytes = self
            .pending_output
            .as_ref()
            .expect("emit polled after completion")
            .bytes();
        if self.output.len() >= limits.output_queue_items
            || bytes > limits.output_queue_bytes - self.output_bytes
        {
            return Poll::Pending;
        }
        let item = self.pending_output.take().unwrap();
        self.output_bytes += bytes;
        self.output.push_back(item);
        Poll::Ready(Ok(()))
    }
    pub fn receive(&mut self) -> Poll<Result<Received>> {
        if let Some(error) = self.error() {
            self.delivery = Delivery::TerminalConsumed;
            return Poll::Ready(Err(error));
        }
        if let Some(item) = self.output.pop_front() {
            self.output_bytes -= item.bytes();
            match &item.value {
                OutputEvent::Audio(audio) => {
                    self.progress.delivered_audio_bytes += audio.pcm().as_bytes().len() as u64
                }
                OutputEvent::Curves(_) => self.progress.delivered_curve_frames += 1,
                _ => {}
            }
            return Poll::Ready(Ok(Received::Event(item)));
        }
        if matches!(self.outcome, Outcome::Success) {
            self.delivery = Delivery::TerminalConsumed;
            return Poll::Ready(Ok(Received::Complete(Summary {
                progress: self.progress,
            })));
        }
        Poll::Pending
    }
    pub fn fail(&mut self, error: Error, id: RequestId) -> Effects {
        if self.error().is_some() || self.delivery == Delivery::TerminalConsumed {
            return Effects::default();
        }
        self.outcome = Outcome::Failed(error.with_request(id).with_progress(self.progress));
        self.input_bytes = 0;
        self.output_bytes = 0;
        let mut effects = Effects {
            discarded_input: std::mem::take(&mut self.input),
            pending_input: self.pending_input.take(),
            discarded_output: std::mem::take(&mut self.output),
            pending_output: self.pending_output.take(),
            notify: true,
            poke: true,
            ..Effects::default()
        };
        effects.retire = self.retire();
        effects
    }
    pub fn complete_cleanup(&mut self, result: Result<()>, id: RequestId) -> Effects {
        // An acknowledged cleanup result is immutable, independently of later
        // delivery failures and deadlines.
        if matches!(self.cleanup, Cleanup::Complete(_)) {
            return Effects::default();
        }
        let mut effects = match result {
            Err(error) => self.fail(error, id),
            Ok(())
                if self.input_phase != InputPhase::EofObserved || self.pending_output.is_some() =>
            {
                self.fail(
                    Error::new(
                        ErrorKind::IncompleteResponse,
                        "backend ended before input/output handoff completed",
                    ),
                    id,
                )
            }
            _ => Effects::default(),
        };
        self.cleanup = Cleanup::Complete(self.error().map_or(Ok(()), Err));
        if self.error().is_none() {
            self.outcome = Outcome::Success;
        }
        effects.retire |= self.retire();
        effects.notify = true;
        effects
    }
    fn retire(&mut self) -> bool {
        let ready = matches!(self.cleanup, Cleanup::Complete(_))
            && (self.error().is_some() || self.delivery == Delivery::TerminalConsumed);
        if ready && self.registration == Registration::Registered {
            self.registration = Registration::Retired;
            true
        } else {
            false
        }
    }
    pub fn changed(&mut self) -> Effects {
        Effects {
            retire: self.retire(),
            notify: true,
            ..Effects::default()
        }
    }
}
