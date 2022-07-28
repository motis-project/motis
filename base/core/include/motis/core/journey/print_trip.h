#include <ctime>
#include <ostream>

#include "motis/core/schedule/time.h"

namespace motis {

struct schedule;
struct trip;
struct extern_trip;

std::ostream& print_time(std::ostream& out, std::time_t t, bool local_time);
std::ostream& print_time(std::ostream& out, schedule const& sched,
                         motis::time const t, bool local_time);

std::ostream& operator<<(std::ostream&, extern_trip const&);

void print_trip(std::ostream&, schedule const&, trip const*,
                bool print_local_time);

}  // namespace motis
