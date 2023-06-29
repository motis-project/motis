#include "motis/footpaths/transfer_updates.h"

#include <cmath>

#include "motis/core/common/logging.h"
#include "motis/footpaths/thread_pool.h"

#include "ppr/routing/search.h"

#include "utl/progress_tracker.h"

using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;

namespace motis::footpaths {

void update_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable tt,
    std::vector<transfer_requests> const& transfer_reqs) {
  std::ignore = rg;
  std::ignore = tt;

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(transfer_reqs.size());

  thread_pool pool{std::max(1U, std::thread::hardware_concurrency())};
  for (auto const& req : transfer_reqs) {
    pool.post([&, &req = req] {
      progress_tracker->increment();
      // TODO (Carsten) compute and update transfers
    });
  }
  pool.join();
  LOG(info) << "Profilebased transfers precomputed.";

  // routing_query rq = {};
  // find_routes_v2(rg, rq);
};

}  // namespace motis::footpaths