#include "motis/core/otel/tracer.h"

namespace motis {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::shared_ptr<opentelemetry::trace::Tracer> motis_tracer;

}  // namespace motis
