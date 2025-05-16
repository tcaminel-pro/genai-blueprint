# utilities to instrument chain execution
# WIP

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def get_telemetry_trace_provider() -> TracerProvider:
    endpoint = "http://0.0.0.0:6006/v1/traces"
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    # tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return trace_provider


LangChainInstrumentor().instrument(tracer_provider=get_telemetry_trace_provider())
