#include "motis/bikesharing/find_connections.h"

#include <vector>

#include "utl/get_or_create.h"

#include "motis/core/common/constants.h"

using namespace flatbuffers;
using namespace motis::module;

namespace motis::bikesharing {

constexpr uint64_t kSecondsPerHour = 3600;

using availability_bucket = BikesharingAvailability;
struct search_impl {
  struct bike_edge {
    int walk_duration_;
    int bike_duration_;
    std::vector<availability_bucket> availability_;

    persistable_terminal* from_;
    persistable_terminal* to_;

    std::string eva_nr_;
  };

  search_impl(database const& db, geo_index const& geo_index)
      : db_(db), geo_index_(geo_index) {}

  msg_ptr find_connections(BikesharingRequest const* req) {
    auto const edges = req->type() == Type::Type_Departure
                           ? find_departures(req)
                           : find_arrivals(req);
    mc_.create_and_finish(MsgContent_BikesharingResponse,
                          CreateBikesharingResponse(mc_, edges).Union());
    return make_msg(mc_);
  }

private:
  Offset<Vector<Offset<BikesharingEdge>>> find_departures(
      BikesharingRequest const* req) {
    auto begin =
        req->interval()->begin() - req->interval()->begin() % kSecondsPerHour;
    auto end = req->interval()->end();
    auto first_bucket = timestamp_to_bucket(begin);

    std::multimap<std::string, bike_edge> departures;
    foreach_terminal_in_walk_dist(
        req->position()->lat(), req->position()->lng(),
        [&, this](std::string const& id, int walk_dur) {
          auto const& from_t = load_terminal(id);
          for (auto const& reachable_t : *from_t->get()->reachable()) {
            auto to_t = load_terminal(reachable_t->id()->str());

            for (auto const& station : *to_t->get()->attached()) {
              // TODO(root) ajdust begin and end with walk_dur
              auto availability =
                  get_availability(from_t->get(), begin, end, first_bucket,
                                   req->availability_aggregator());
              bike_edge edge{walk_dur + station->duration(),
                             reachable_t->duration(),
                             availability,
                             from_t,
                             to_t,
                             station->id()->str()};
              departures.emplace(id, edge);
            }
          }
        });

    return serialize_edges(departures);
  }

  Offset<Vector<Offset<BikesharingEdge>>> find_arrivals(
      BikesharingRequest const* req) {
    auto begin =
        req->interval()->begin() - req->interval()->begin() % kSecondsPerHour;
    auto end = req->interval()->end() + MAX_TRAVEL_TIME_SECONDS;
    auto first_bucket = timestamp_to_bucket(begin);

    std::multimap<std::string, bike_edge> arrivals;
    foreach_terminal_in_walk_dist(
        req->position()->lat(), req->position()->lng(),
        [&, this](std::string const& id, int walk_dur) {
          auto const& to_t = load_terminal(id);
          for (auto const& reachable_t : *to_t->get()->reachable()) {
            auto from_t = load_terminal(reachable_t->id()->str());

            for (auto const& station : *from_t->get()->attached()) {
              auto availability =
                  get_availability(from_t->get(), begin, end, first_bucket,
                                   req->availability_aggregator());
              bike_edge edge{walk_dur + station->duration(),
                             reachable_t->duration(),
                             availability,
                             from_t,
                             to_t,
                             station->id()->str()};
              arrivals.emplace(id, edge);
            }
          }
        });

    return serialize_edges(arrivals);
  }

  template <typename F>
  void foreach_terminal_in_walk_dist(double lat, double lng, F func) const {
    for (const auto& t : geo_index_.get_terminals(lat, lng, MAX_WALK_DIST)) {
      func(t.id_, t.distance_ * LINEAR_DIST_APPROX / WALK_SPEED);
    }
  }

  persistable_terminal* load_terminal(std::string const& id) {
    return utl::get_or_create(
               terminals_, id,
               [&]() {
                 return std::make_unique<persistable_terminal>(db_.get(id));
               })
        .get();
  }

  static std::vector<availability_bucket> get_availability(
      Terminal const* term, uint64_t begin, uint64_t end, size_t bucket,
      AvailabilityAggregator aggr) {
    std::vector<availability_bucket> availability;
    for (auto t = begin; t < end; t += kSecondsPerHour) {
      double val = bikesharing::get_availability(
          term->availability()->Get(bucket), aggr);
      bucket = (bucket + 1) % kBucketCount;
      availability.emplace_back(t, t + kSecondsPerHour, val);
    }
    return availability;
  }

  Offset<Vector<Offset<BikesharingEdge>>> serialize_edges(
      std::multimap<std::string, bike_edge> const& edges) {
    std::vector<Offset<BikesharingEdge>> stored;
    for (auto const& pair : edges) {
      auto const& edge = pair.second;
      auto from = serialize_terminal(edge.from_);
      auto to = serialize_terminal(edge.to_);

      stored.push_back(CreateBikesharingEdge(
          mc_, from, to, mc_.CreateVectorOfStructs(edge.availability_),
          mc_.CreateString(edge.eva_nr_), edge.walk_duration_,
          edge.bike_duration_));
    }
    return mc_.CreateVector(stored);
  }

  Offset<BikesharingTerminal> serialize_terminal(
      persistable_terminal* terminal) {
    auto const* t = terminal->get();
    return utl::get_or_create(terminal_offsets_, t->id()->str(), [&]() {
      motis::Position pos(t->lat(), t->lng());
      return CreateBikesharingTerminal(mc_, mc_.CreateString(t->id()->str()),
                                       mc_.CreateString(t->name()->str()),
                                       &pos);
    });
  }

  database const& db_;
  geo_index const& geo_index_;

  message_creator mc_;
  std::map<std::string, std::unique_ptr<persistable_terminal>> terminals_;
  std::map<std::string, Offset<BikesharingTerminal>> terminal_offsets_;
};

msg_ptr find_connections(database const& db, geo_index const& index,
                         BikesharingRequest const* req) {
  search_impl impl(db, index);
  return impl.find_connections(req);
}

}  // namespace motis::bikesharing
