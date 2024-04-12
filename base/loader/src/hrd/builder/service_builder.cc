#include "motis/loader/hrd/builder/service_builder.h"

#include <sstream>

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#if defined(_WIN32) && defined(CreateService)
#undef CreateService
#endif

using namespace utl;
using namespace flatbuffers64;

namespace motis::loader::hrd {

service_builder::service_builder(track_rules track_rules)
    : track_rules_(std::move(track_rules)) {}

Offset<Vector<Offset<Section>>> create_sections(
    std::vector<hrd_service::section> const& sections, category_builder& cb,
    provider_builder& pb, line_builder& lb, attribute_builder& ab,
    bitfield_builder& bb, direction_builder& db, station_builder& sb,
    FlatBufferBuilder& fbb) {
  return fbb.CreateVector(utl::to_vec(
      begin(sections), end(sections), [&](hrd_service::section const& s) {
        return CreateSection(
            fbb, cb.get_or_create_category(s.category_[0], fbb),
            pb.get_or_create_provider(raw_to_int<uint64_t>(s.admin_), fbb),
            s.train_num_, lb.get_or_create_line(s.line_information_, fbb),
            ab.create_attributes(s.attributes_, bb, fbb),
            db.get_or_create_direction(s.directions_, sb, fbb));
      }));
}

void create_tracks(track_rule_key const& key, int time,
                   track_rules const& track_rules, bitfield_builder& bb,
                   std::vector<Offset<Track>>& tracks, FlatBufferBuilder& fbb) {
  auto dep_plr_it = track_rules.find(key);
  if (dep_plr_it == end(track_rules)) {
    return;
  }

  for (auto const& rule : dep_plr_it->second) {
    if (rule.time_ == TIME_NOT_SET || time % 1440 == rule.time_) {
      tracks.push_back(
          CreateTrack(fbb, bb.get_or_create_bitfield(rule.bitfield_num_, fbb),
                      rule.track_name_));
    }
  }
}

Offset<Vector<Offset<TrackRules>>> create_tracks(
    std::vector<hrd_service::section> const& sections,
    std::vector<hrd_service::stop> const& stops, track_rules const& track_rules,
    bitfield_builder& bb, FlatBufferBuilder& fbb) {
  struct stop_tracks {
    std::vector<Offset<Track>> dep_tracks_;
    std::vector<Offset<Track>> arr_tracks_;
  };

  std::vector<stop_tracks> stops_tracks(stops.size());
  for (auto i = 0UL; i < sections.size(); ++i) {
    int const section_index = i;
    int const from_stop_index = section_index;
    int const to_stop_index = from_stop_index + 1;

    auto const& section = sections[section_index];
    auto const& from_stop = stops[from_stop_index];
    auto const& to_stop = stops[to_stop_index];

    auto dep_event_key = std::make_tuple(from_stop.eva_num_, section.train_num_,
                                         raw_to_int<uint64_t>(section.admin_));
    auto arr_event_key = std::make_tuple(to_stop.eva_num_, section.train_num_,
                                         raw_to_int<uint64_t>(section.admin_));

    create_tracks(dep_event_key, from_stop.dep_.time_, track_rules, bb,
                  stops_tracks[from_stop_index].dep_tracks_, fbb);
    create_tracks(arr_event_key, to_stop.arr_.time_, track_rules, bb,
                  stops_tracks[to_stop_index].arr_tracks_, fbb);
  }

  return fbb.CreateVector(utl::to_vec(
      begin(stops_tracks), end(stops_tracks), [&](stop_tracks const& sp) {
        return CreateTrackRules(fbb, fbb.CreateVector(sp.arr_tracks_),
                                fbb.CreateVector(sp.dep_tracks_));
      }));
}

Offset<Vector<int32_t>> create_times(
    std::vector<hrd_service::stop> const& stops, FlatBufferBuilder& fbb) {
  std::vector<int32_t> times;
  for (auto const& stop : stops) {
    times.push_back(stop.arr_.time_);
    times.push_back(stop.dep_.time_);
  }
  return fbb.CreateVector(times);
}

Offset<Service> service_builder::create_service(
    hrd_service const& s, route_builder& rb, station_builder& sb,
    category_builder& cb, provider_builder& pb, line_builder& lb,
    attribute_builder& ab, bitfield_builder& bb, direction_builder& db,
    FlatBufferBuilder& fbb, bool is_rule_participant) {
  fbs_services_.push_back(CreateService(
      fbb, rb.get_or_create_route(s.stops_, sb, fbb),
      bb.get_or_create_bitfield(s.traffic_days_, fbb),
      create_sections(s.sections_, cb, pb, lb, ab, bb, db, sb, fbb),
      create_tracks(s.sections_, s.stops_, track_rules_, bb, fbb),
      create_times(s.stops_, fbb), rb.get_or_create_route(s.stops_, sb, fbb).o,
      CreateServiceDebugInfo(
          fbb,
          utl::get_or_create(
              filenames_, s.origin_.filename_,
              [&fbb, &s]() { return fbb.CreateString(s.origin_.filename_); }),
          s.origin_.line_number_from_, s.origin_.line_number_to_),
      static_cast<uint8_t>(is_rule_participant ? 1U : 0U) != 0U,
      s.initial_train_num_, 0 /* gtfs trip_id */, 0 /* gtfs seq_numbers */,
      ScheduleRelationship_SCHEDULED,
      fbb.CreateSharedString(s.initial_admin_.data(),
                             s.initial_admin_.length())));
  return fbs_services_.back();
}

}  // namespace motis::loader::hrd
