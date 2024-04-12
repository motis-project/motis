#include "motis/loader/hrd/model/repeat_service.h"

#include "utl/to_vec.h"

namespace motis::loader::hrd {

hrd_service::event update_event(hrd_service::event const& origin, int interval,
                                int repetition) {
  auto const new_time = origin.time_ != hrd_service::NOT_SET
                            ? origin.time_ + (interval * repetition)
                            : hrd_service::NOT_SET;
  return {new_time, origin.in_out_allowed_};
}

hrd_service create_repetition(hrd_service const& origin, int repetition) {
  return {origin.origin_,
          0,
          0,
          utl::to_vec(begin(origin.stops_), end(origin.stops_),
                      [&origin, &repetition](hrd_service::stop const& s) {
                        return hrd_service::stop{
                            s.eva_num_,
                            update_event(s.arr_, origin.interval_, repetition),
                            update_event(s.dep_, origin.interval_, repetition)};
                      }),
          origin.sections_,
          origin.traffic_days_,
          origin.initial_train_num_,
          origin.initial_admin_};
}

void expand_repetitions(std::vector<hrd_service>& services) {
  int const size = services.size();
  services.reserve(services.size() * (services[0].num_repetitions_ + 1));
  for (int service_idx = 0; service_idx < size; ++service_idx) {
    auto const& service = services[service_idx];
    for (int repetition = 1; repetition <= service.num_repetitions_;
         ++repetition) {
      services.push_back(create_repetition(service, repetition));
    }
  }
}

}  // namespace motis::loader::hrd
