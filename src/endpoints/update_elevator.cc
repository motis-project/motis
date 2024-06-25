#include "icc/endpoints/update_elevator.h"

#include "utl/helpers/algorithm.h"

namespace json = boost::json;

namespace icc::ep {

json::value update_elevator::operator()(json::value const& query) const {
  auto const& q = query.as_object();
  auto const id = q.at("id").to_number<std::int64_t>();
  auto const status = status_from_str(q.at("status").as_string());

  auto const e = e_.get();
  auto elevators = e->elevators_;
  auto const it =
      utl::find_if(elevators, [&](auto&& x) { return x.id_ == id; });
  if (it == end(elevators)) {
    return json::value{{"error", "id not found"}};
  }

  it->status_ = status;
  e_.set(
      shared_elevators::elevators{w_, elevator_nodes_, std::move(elevators)});

  return json::string{{"success", true}};
}

}  // namespace icc::ep