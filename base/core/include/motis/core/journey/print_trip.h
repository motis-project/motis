#include <ostream>

namespace motis {

struct schedule;
struct trip;
struct extern_trip;

std::ostream& operator<<(std::ostream&, extern_trip const&);

void print_trip(std::ostream&, schedule const&, trip const*,
                bool print_local_time);

}  // namespace motis