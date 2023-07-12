#include <cstdint>

#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/json/json.h"

#include "motis/ris/ribasis/common.h"
#include "motis/ris/ribasis/ribasis_formation_parser.h"

using namespace flatbuffers;
using namespace motis::json;

namespace motis::ris::ribasis::formation {

Offset<HalfTripId> parse_half_trip_id(context& ctx,
                                      rapidjson::Value const& fahrt) {
  auto const uuid =
      ctx.ris_.b_.CreateString(get_optional_str(fahrt, "fahrtID"));

  auto const& rel = get_obj(fahrt, "fahrtRelation");
  auto const start_time =
      get_schedule_timestamp(ctx.ris_, rel, "abfahrtZeitGeplant");
  auto const start_station_eva = get_str(rel, "startEvanummer");
  auto const train_nr =
      get_parsed_number<std::uint32_t>(rel, "startFahrtnummer");

  ctx.fahrtnummer_ = train_nr;
  ctx.gattung_ = get_str(rel, "startGattung");

  auto const empty_str = ctx.ris_.b_.CreateSharedString("");
  return CreateHalfTripId(
      ctx.ris_.b_,
      CreateTripId(ctx.ris_.b_, uuid,
                   ctx.ris_.b_.CreateSharedString(start_station_eva.data(),
                                                  start_station_eva.size()),
                   train_nr, start_time, empty_str, 0, empty_str),
      uuid);
}

Offset<StationInfo> parse_station(context& ctx, rapidjson::Value const& stop) {
  auto const eva = get_str(stop, "evanummer");
  auto const ds100 = get_optional_str(stop, "code");
  auto const name = get_str(stop, "bezeichnung");
  return utl::get_or_create(ctx.stations_, eva, [&]() {
    return CreateStationInfo(
        ctx.ris_.b_, ctx.ris_.b_.CreateSharedString(eva.data(), eva.size()),
        ctx.ris_.b_.CreateString(ds100), ctx.ris_.b_.CreateString(name));
  });
}

Offset<VehicleInfo> parse_vehicle(context& ctx, rapidjson::Value const& vi) {
  auto const uic =
      get_parsed_number<std::uint64_t>(vi, "fahrzeugnummer", true, true);
  auto const baureihe = get_str(vi, "fahrzeugbabr");
  auto const type = get_str(vi, "fahrzeugtyp");
  auto const order = get_str(vi, "ordnungsnummer");

  return CreateVehicleInfo(
      ctx.ris_.b_, uic,
      ctx.ris_.b_.CreateSharedString(baureihe.data(), baureihe.size()),
      ctx.ris_.b_.CreateSharedString(type.data(), type.size()),
      ctx.ris_.b_.CreateSharedString(order.data(), order.size()));
}

Offset<VehicleGroup> parse_vehicle_group(context& ctx,
                                         rapidjson::Value const& vg,
                                         Offset<String> default_dep_uuid) {
  auto const name = get_str(vg, "bezeichnung");
  auto const start_station =
      parse_station(ctx, get_obj(vg, "starthaltestelle"));
  auto const dest_station = parse_station(ctx, get_obj(vg, "zielhaltestelle"));

  auto trip_id = ctx.half_trip_id_;
  auto const& fahrt = get_value(vg, "fahrtAbweichend");
  if (fahrt.IsObject()) {
    trip_id = parse_half_trip_id(ctx, fahrt);
  }

  auto dep_uuid = default_dep_uuid;
  auto const& dep = get_value(vg, "abfahrtAbweichend");
  if (dep.IsObject()) {
    dep_uuid = ctx.ris_.b_.CreateString(get_optional_str(dep, "abfahrtID"));
  }

  return CreateVehicleGroup(
      ctx.ris_.b_, ctx.ris_.b_.CreateSharedString(name.data(), name.size()),
      start_station, dest_station, trip_id, dep_uuid,
      ctx.ris_.b_.CreateVector(
          utl::to_vec(get_array(vg, "allFahrzeug"),
                      [&](auto const& vi) { return parse_vehicle(ctx, vi); })));
}

Offset<TripFormationSection> parse_section(context& ctx,
                                           rapidjson::Value const& sec) {
  auto const& abfahrt = get_obj(sec, "abfahrt");
  auto const& dep = get_obj(abfahrt, "abfahrt");
  auto const dep_uuid = get_optional_str(dep, "abfahrtID");
  auto const& dep_rel = get_obj(dep, "abfahrtRelation");
  auto const dep_time =
      get_schedule_timestamp(ctx.ris_, dep_rel, "abfahrtZeitGeplant");
  auto const dep_station = parse_station(ctx, get_obj(abfahrt, "haltestelle"));

  auto const fbs_dep_uuid = ctx.ris_.b_.CreateString(dep_uuid);
  return CreateTripFormationSection(
      ctx.ris_.b_, fbs_dep_uuid, dep_station, dep_time,
      ctx.ris_.b_.CreateVector(utl::to_vec(
          get_array(abfahrt, "allFahrzeuggruppe"), [&](auto const& vg) {
            return parse_vehicle_group(ctx, vg, fbs_dep_uuid);
          })));
}

Offset<Vector<Offset<TripFormationSection>>> parse_sections(
    context& ctx, rapidjson::Value const& data) {
  auto const sections_data = get_array(data, "allFahrtabschnitt");
  return ctx.ris_.b_.CreateVector(
      utl::to_vec(sections_data, [&](auto const& sec) {
        utl::verify(sec.IsObject(), "invalid allFahrtabschnitt entry");
        return parse_section(ctx, sec);
      }));
}

void parse_ribasis_formation(ris_msg_context& ris_ctx,
                             rapidjson::Value const& data) {
  auto ctx = context{ris_ctx};
  ctx.half_trip_id_ = parse_half_trip_id(ctx, get_obj(data, "fahrt"));
  auto const sections = parse_sections(ctx, data);
  auto const formation_msg =
      CreateTripFormationMessage(ctx.ris_.b_, ctx.half_trip_id_, sections);
  ctx.ris_.b_.Finish(CreateRISMessage(
      ctx.ris_.b_, ctx.ris_.earliest_, ctx.ris_.latest_, ctx.ris_.timestamp_,
      RISMessageUnion_TripFormationMessage, formation_msg.Union()));
}

}  // namespace motis::ris::ribasis::formation
