use crate::{
    Audio2Face3DContext,
    logging::{LogLevel, Logger},
};
use std::{
    cell::Cell,
    fmt::{self, Write},
};
use tracing::{
    Metadata, Subscriber,
    field::{Field, Visit},
    subscriber::Interest,
};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};
thread_local! {static WRITING:Cell<bool>=const{Cell::new(false)};}
struct Reentry;
impl Drop for Reentry {
    fn drop(&mut self) {
        WRITING.with(|v| v.set(false));
    }
}
pub(super) fn dispatch(context: Audio2Face3DContext) -> tracing::Dispatch {
    tracing::Dispatch::new(tracing_subscriber::registry().with(Bridge {
        context,
        _registration_guard: tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()),
    }))
}
struct Bridge {
    context: Audio2Face3DContext,
    // tracing-core uses the current thread to register callsites when only one
    // dispatcher exists. A scoped subscriber may be absent on that thread. Keep
    // a second inert registrar alive so interest is computed from all dispatchers.
    // Neither registrar is installed globally.
    _registration_guard: tracing::Dispatch,
}
fn level(value: &tracing::Level) -> LogLevel {
    match *value {
        tracing::Level::TRACE => LogLevel::Trace,
        tracing::Level::DEBUG => LogLevel::Debug,
        tracing::Level::INFO => LogLevel::Info,
        tracing::Level::WARN => LogLevel::Warn,
        tracing::Level::ERROR => LogLevel::Error,
    }
}
struct Text {
    value: String,
    limit: usize,
    fields: usize,
}
impl Text {
    fn new(limit: usize) -> Self {
        Self {
            value: String::new(),
            limit,
            fields: 0,
        }
    }
}
impl Write for Text {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let mut end = text.len().min(self.limit.saturating_sub(self.value.len()));
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.value.push_str(&text[..end]);
        Ok(())
    }
}
impl Visit for Text {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if self.fields >= 16 || self.value.len() >= self.limit {
            return;
        }
        self.fields += 1;
        let _ = write!(self, " {}={:?}", field.name(), value);
    }
}
#[derive(Clone)]
struct SpanText(String, usize);
impl<S> Layer<S> for Bridge
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }
    fn enabled(&self, metadata: &Metadata<'_>, _: Context<'_, S>) -> bool {
        !WRITING.with(Cell::get) && level(metadata.level()) >= self.context.logger().log_level()
    }
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: Context<'_, S>,
    ) {
        let mut text = Text::new(4096);
        let _ = write!(text, "{}", attrs.metadata().name());
        attrs.record(&mut text);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut()
                .insert(SpanText(text.value, text.fields));
        }
    }
    fn on_record(&self, id: &tracing::Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let old = span
                .extensions()
                .get::<SpanText>()
                .cloned()
                .unwrap_or(SpanText(String::new(), 0));
            let mut text = Text {
                value: old.0,
                limit: 4096,
                fields: old.1,
            };
            values.record(&mut text);
            span.extensions_mut()
                .replace(SpanText(text.value, text.fields));
        }
    }
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        if WRITING.with(|v| v.replace(true)) {
            return;
        }
        let _reentry = Reentry;
        let mut text = Text::new(16384);
        let _ = write!(text, "{}", event.metadata().target());
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                let saved = span.extensions().get::<SpanText>().cloned();
                if let Some(saved) = saved {
                    let _ = write!(text, " [{}]", saved.0);
                }
            }
        }
        event.record(&mut text);
        self.context
            .logger()
            .write_log(level(event.metadata().level()), text.value.into());
    }
}
