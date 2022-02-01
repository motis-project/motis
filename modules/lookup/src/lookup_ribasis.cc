#include "motis/lookup/lookup_ribasis.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "date/date.h"

#include "utl/get_or_create.h"
#include "utl/to_set.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/pair.h"
#include "motis/string.h"

#include "motis/core/access/bfs.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

using namespace flatbuffers;
using namespace motis::ribasis;
namespace uu = boost::uuids;

namespace motis::lookup {

/* Current limitations:
 * - All UUIDs are randomly generated for every request and can
 *   only be used to parse the message (i.e. they are not stable and not stored)
 * - All lines with the same name have the same UUID
 * - All tracks with the same name (even at different stations) have the same
 *   UUID
 * - kategorie is always set to VORSCHAU
 * - fahrttyp is always set to PLANFAHRT
 * - zusatzhalt and bedarfshalt are always set to false
 * - RiBasisFahrtZuordnungstyp is always set to DURCHBINDUNG
 */

namespace {

constexpr auto const FULL_TIME_FORMAT = "%FT%T%Ez";
constexpr auto const DATE_FORMAT = "%F";

struct rib_ctx {
  rib_ctx(FlatBufferBuilder& fbb, schedule const& sched)
      : fbb_{fbb}, sched_{sched}, empty_string_{fbb.CreateString("")} {}

  Offset<String> timestamp(time const mt,
                           char const* format = FULL_TIME_FORMAT) {
    return fbb_.CreateString(date::format(
        format, date::sys_seconds{
                    std::chrono::seconds{motis_to_unixtime(sched_, mt)}}));
  }

  Offset<RiBasisHaltestelle> station(uint32_t const station_idx) {
    return utl::get_or_create(stations_, station_idx, [&]() {
      auto const& st = sched_.stations_.at(station_idx);
      auto const rl100 = st->external_ids_.empty()
                             ? empty_string_
                             : fbb_.CreateString(st->external_ids_.front());
      return CreateRiBasisHaltestelle(
          fbb_,
          rand_uuid(),  // haltestelleid
          fbb_.CreateString(st->name_),  // bezeichnung
          fbb_.CreateString(st->eva_nr_),  // evanummer
          rl100  // rl100
      );
    });
  }

  Offset<String> provider_id(provider const* p) {
    return utl::get_or_create(provider_ids_, p, [&]() {
      auto const id = rand_uuid();
      providers_.emplace_back(CreateRiBasisVerwaltung(
          fbb_, id,  // verwaltungid
          CreateRiBasisBetreiber(
              fbb_,
              p != nullptr ? fbb_.CreateString(p->full_name_)
                           : empty_string_,  // name
              p != nullptr ? fbb_.CreateString(p->short_name_)
                           : empty_string_  // code
              )));
      return id;
    });
  }

  Offset<String> category_id(uint32_t const family) {
    return utl::get_or_create(category_ids_, family, [&]() {
      auto const& cat = sched_.categories_.at(family);
      auto const id = rand_uuid();
      auto const name = fbb_.CreateString(cat->name_);
      categories_.emplace_back(CreateRiBasisGattung(fbb_,
                                                    id,  // gattungid
                                                    name,  // name
                                                    name  // code
                                                    ));
      return id;
    });
  }

  Offset<String> line_id(mcd::string const& line_identifier) {
    return utl::get_or_create(line_ids_, line_identifier, [&]() {
      auto const id = rand_uuid();
      lines_.emplace_back(CreateRiBasisLinie(
          fbb_,
          id,  // linieid
          fbb_.CreateString(line_identifier)  // name
          ));
      return id;
    });
  }

  Offset<RiBasisOrt> track(uint16_t const track_idx) {
    return utl::get_or_create(tracks_, track_idx, [&]() {
      auto const id = track_idx == 0 ? empty_string_ : rand_uuid();
      return CreateRiBasisOrt(
          fbb_,
          id,  // ortid
          fbb_.CreateString(sched_.tracks_.at(track_idx)),  // bezeichnung
          ribasis::RiBasisOrtTyp_GLEIS  // orttyp
      );
    });
  }

  Offset<String> trip_id(trip const* trp) {
    return utl::get_or_create(trip_ids_, trp, [&]() { return rand_uuid(); });
  }

  Offset<String> event_key(trip const* trp, ev_key const ev) {
    return utl::get_or_create(event_keys_, mcd::pair{trp, ev},
                              [&]() { return rand_uuid(); });
  }

  Offset<String> rand_uuid() {
    return fbb_.CreateString(uu::to_string(uuid_gen_()));
  }

  FlatBufferBuilder& fbb_;
  schedule const& sched_;
  uu::random_generator uuid_gen_{};

  Offset<String> empty_string_;

  mcd::hash_map<uint32_t /* station index */, Offset<RiBasisHaltestelle>>
      stations_;
  mcd::hash_map<trip const*, Offset<String>> trip_ids_;
  mcd::hash_map<mcd::pair<trip const*, ev_key>, Offset<String>> event_keys_;
  mcd::hash_map<uint16_t /* track index */, Offset<RiBasisOrt>> tracks_;

  mcd::hash_map<provider const*, Offset<String>> provider_ids_;
  std::vector<Offset<RiBasisVerwaltung>> providers_;

  mcd::hash_map<uint32_t /* family */, Offset<String>> category_ids_;
  std::vector<Offset<RiBasisGattung>> categories_;

  mcd::hash_map<mcd::string, Offset<String>> line_ids_;
  std::vector<Offset<RiBasisLinie>> lines_;
};

Offset<RiBasisFahrtRelation> trip_relation(rib_ctx& rc, trip const* trp) {
  // motis trip ids don't have provider + category information, so we
  // need to look it up for the first trip section (not ideal since this
  // may have changed...)
  auto const sections = motis::access::sections{trp};
  utl::verify(begin(sections) != end(sections),
              "lookup_ribasis_trip: trip has no sections");
  auto const first_section = *begin(sections);
  auto const& provider = first_section.lcon().full_con_->con_info_->provider_;
  return CreateRiBasisFahrtRelation(
      rc.fbb_,
      rc.fbb_.CreateSharedString(std::to_string(
          trp->id_.primary_.get_train_nr())),  // startfahrtnummer
      rc.timestamp(trp->id_.primary_.get_time()),  // startzeit
      provider != nullptr ? rc.fbb_.CreateString(provider->short_name_)
                          : rc.empty_string_,  // startverwaltung
      rc.fbb_.CreateString(
          rc.sched_.categories_
              .at(first_section.lcon().full_con_->con_info_->family_)
              ->name_),  // startgattung
      rc.fbb_.CreateString(trp->id_.secondary_.line_id_),  // startlinie
      rc.station(trp->id_.primary_.get_station_id()),  // starthaltestelle
      rc.timestamp(trp->id_.secondary_.target_time_),  // zielzeit
      rc.station(trp->id_.secondary_.target_station_id_)  // zielhaltestelle
  );
}

RiBasisZeitstatus convert_reason(timestamp_reason const tr) {
  switch (tr) {
    case timestamp_reason::SCHEDULE: return ribasis::RiBasisZeitstatus_FAHRPLAN;
    case timestamp_reason::IS: return ribasis::RiBasisZeitstatus_MELDUNG;
    case timestamp_reason::FORECAST:
    case timestamp_reason::PROPAGATION:
    case timestamp_reason::REPAIR: return ribasis::RiBasisZeitstatus_PROGNOSE;
    default: throw utl::fail("unsupported timestamp_reason");
  }
}

std::vector<trip const*> get_merged_trips(
    rib_ctx& rc, trip const* trp, motis::access::trip_section const& sec) {
  std::vector<trip const*> merged;
  for (auto const& mt : *rc.sched_.merged_trips_.at(sec.lcon().trips_)) {
    if (mt != trp) {
      merged.emplace_back(mt);
    }
  }
  return merged;
}

Offset<RiBasisAbfahrt> departure(rib_ctx& rc, trip const* trp,
                                 motis::access::trip_section const& sec) {
  auto const di = get_delay_info(rc.sched_, sec.ev_key_from());
  return CreateRiBasisAbfahrt(
      rc.fbb_, rc.event_key(trp, sec.ev_key_from()),  // abfahrtid
      rc.station(sec.from_station_id()),  // haltestelle
      sec.from_node()->is_in_allowed(),  // fahrgastwechsel
      rc.timestamp(di.get_schedule_time()),  // planabfahrtzeit
      rc.timestamp(sec.lcon().d_time_),  // abfahrtzeit
      convert_reason(di.get_reason()),  // abfahrtzeitstatus
      rc.track(
          get_schedule_track(rc.sched_, sec.ev_key_from())),  // planabfahrtort
      rc.track(sec.lcon().full_con_->d_track_),  // abfahrtort
      false,  // zusatzhalt
      false,  // bedarfshalt
      rc.fbb_.CreateVector(
          std::vector<
              Offset<RiBasisAbfahrtZuordnung>>{})  // allAbfahrtzuordnung
  );
}

Offset<RiBasisAnkunft> arrival(rib_ctx& rc, trip const* trp,
                               motis::access::trip_section const& sec) {
  auto const di = get_delay_info(rc.sched_, sec.ev_key_to());
  return CreateRiBasisAnkunft(
      rc.fbb_, rc.event_key(trp, sec.ev_key_to()),  // ankunftid
      rc.station(sec.to_station_id()),  // haltestelle
      sec.to_node()->is_out_allowed(),  // fahrgastwechsel
      rc.timestamp(di.get_schedule_time()),  // planankunftzeit
      rc.timestamp(sec.lcon().a_time_),  // ankunftzeit
      convert_reason(di.get_reason()),  // ankunftzeitstatus
      rc.track(
          get_schedule_track(rc.sched_, sec.ev_key_to())),  // planankunftort,
      rc.track(sec.lcon().full_con_->a_track_),  // ankunftort
      false,  // zusatzhalt
      false,  // bedarfshalt
      rc.fbb_.CreateVector(
          std::vector<
              Offset<RiBasisAnkunftZuordnung>>{})  // allAnkunftzuordnung
  );
}

Offset<Vector<Offset<RiBasisFahrtAbschnitt>>> trip_sections(rib_ctx& rc,
                                                            trip const* trp) {
  auto const secs = utl::to_vec(
      motis::access::sections{trp},
      [&](motis::access::trip_section const& sec) {
        auto const& lc = sec.lcon();
        auto const& ci = lc.full_con_->con_info_;
        return CreateRiBasisFahrtAbschnitt(
            rc.fbb_,
            rc.fbb_.CreateSharedString(std::to_string(output_train_nr(
                ci->train_nr_, ci->original_train_nr_))),  // fahrtnummer
            rc.fbb_.CreateSharedString(
                get_service_name(rc.sched_, ci)),  // fahrtbezeichnung
            rc.empty_string_,  // fahrtname,
            rc.provider_id(ci->provider_),  // verwaltungid
            rc.category_id(ci->family_),  // gattungid
            rc.line_id(ci->line_identifier_),  // lineid
            departure(rc, trp, sec),  // abfahrt
            arrival(rc, trp, sec),  // ankunft
            rc.fbb_.CreateVector(  // allVereinigtmit
                utl::to_vec(get_merged_trips(rc, trp, sec),
                            [&](trip const* merged_trp) {
                              return CreateRiBasisFormation(
                                  rc.fbb_,
                                  rc.trip_id(merged_trp),  // fahrtid
                                  rc.event_key(merged_trp,
                                               sec.ev_key_from()),  // abfahrtid
                                  rc.event_key(merged_trp,
                                               sec.ev_key_to()));  // ankunftid
                            })));
      });
  return rc.fbb_.CreateVector(secs);
}

Offset<RiBasisTrip> rib_trip(rib_ctx& rc, trip const* trp,
                             std::vector<edge const*> const& through_edges) {
  auto const start_day =
      rc.timestamp(trp->id_.primary_.get_time(), DATE_FORMAT);
  auto const now = std::chrono::system_clock::now();

  std::vector<Offset<RiBasisZubringerFahrtZuordnung>> through_in;
  std::vector<Offset<RiBasisAbbringerFahrtZuordnung>> through_out;
  auto const route_nodes = utl::to_set(
      access::stops{trp}, [](auto const& ts) { return ts.get_route_node(); });
  auto const lcon_idx = trp->lcon_idx_;
  for (auto const te : through_edges) {
    if (route_nodes.find(te->from_) != end(route_nodes)) {
      for (auto const& oe : te->to_->edges_) {
        if (oe.type() != edge::ROUTE_EDGE) {
          continue;
        }
        for (auto const& tt : *rc.sched_.merged_trips_.at(
                 oe.m_.route_edge_.conns_.at(lcon_idx).trips_)) {
          through_out.emplace_back(CreateRiBasisAbbringerFahrtZuordnung(
              rc.fbb_,
              rc.trip_id(tt),  // fahrtid
              rc.event_key(
                  tt, ev_key{&oe, lcon_idx, event_type::DEP}),  // abfahrtid
              ribasis::RiBasisFahrtZuordnungstyp_DURCHBINDUNG));  // typ
        }
      }
    }
    if (route_nodes.find(te->to_) != end(route_nodes)) {
      for (auto const& ie : te->from_->incoming_edges_) {
        if (ie->type() != edge::ROUTE_EDGE) {
          continue;
        }
        for (auto const& tt : *rc.sched_.merged_trips_.at(
                 ie->m_.route_edge_.conns_.at(lcon_idx).trips_)) {
          through_in.emplace_back(CreateRiBasisZubringerFahrtZuordnung(
              rc.fbb_,
              rc.trip_id(tt),  // fahrtid
              rc.event_key(tt, ev_key{cista::ptr_cast(ie), lcon_idx,
                                      event_type::ARR}),  // ankunftid
              ribasis::RiBasisFahrtZuordnungstyp_DURCHBINDUNG));  // typ
        }
      }
    }
  }

  return CreateRiBasisTrip(
      rc.fbb_, to_fbs(rc.sched_, rc.fbb_, trp),
      CreateRiBasisFahrt(
          rc.fbb_,
          CreateRiBasisMeta(
              rc.fbb_,
              rc.rand_uuid(),  // id
              rc.empty_string_,  // owner
              rc.fbb_.CreateString("RIPL"),  // format
              rc.fbb_.CreateString("v3"),  // version
              rc.fbb_.CreateVector(
                  std::vector<Offset<String>>{}),  // correlation
              rc.fbb_.CreateString(  // created
                  date::format(FULL_TIME_FORMAT, now)),
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch())
                  .count()  // sequence
              ),
          CreateRiBasisFahrtData(
              rc.fbb_,
              ribasis::RiBasisFahrtKategorie_VORSCHAU,  // kategorie
              start_day,  // planstarttag
              rc.trip_id(trp),  // fahrtid
              trip_relation(rc, trp),
              start_day,  // verkehrstag
              ribasis::RiBasisFahrtTyp_PLANFAHRT,  // fahrttyp
              rc.fbb_.CreateVector(rc.providers_),  // allVerwaltung
              rc.fbb_.CreateVector(rc.categories_),  // allGattung
              rc.fbb_.CreateVector(rc.lines_),  // allLinie
              trip_sections(rc, trp),  // allFahrtabschnitt
              rc.fbb_.CreateVector(through_in),  // allZubringerfahrtzuordnung
              rc.fbb_.CreateVector(  // allAbbringerfahrtzuordnung
                  through_out))));
}

std::pair<mcd::hash_set<trip const*>, std::vector<edge const*>>
trips_and_through_edges_in_route(schedule const& sched,
                                 trip const* requested_trip) {
  mcd::hash_set<trip const*> trips;
  std::vector<edge const*> through_edges;
  if (requested_trip->edges_->empty()) {
    return {trips, through_edges};
  }

  auto const first_dep = ev_key{requested_trip->edges_->front().get_edge(),
                                requested_trip->lcon_idx_, event_type::DEP};
  auto const lcon_idx = requested_trip->lcon_idx_;

  for (auto const& re : route_bfs(first_dep, bfs_direction::BOTH, true)) {
    auto const* e = re.get_edge();
    if (e->type() == edge::ROUTE_EDGE) {
      auto const& lcon = e->m_.route_edge_.conns_.at(lcon_idx);
      for (auto const& trp : *sched.merged_trips_.at(lcon.trips_)) {
        trips.insert(trp);
      }
    } else if (e->type() == edge::THROUGH_EDGE) {
      through_edges.emplace_back(e);
    }
  }
  return {trips, through_edges};
}

}  // namespace

Offset<LookupRiBasisResponse> lookup_ribasis(FlatBufferBuilder& fbb,
                                             schedule const& sched,
                                             LookupRiBasisRequest const* req) {
  auto const t = req->trip_id();
  auto requested_trp = get_trip(sched, t->station_id()->str(), t->train_nr(),
                                t->time(), t->target_station_id()->str(),
                                t->target_time(), t->line_id()->str());

  auto rc = rib_ctx{fbb, sched};
  auto const trips_and_through_edges =
      trips_and_through_edges_in_route(sched, requested_trp);
  auto const& trips = trips_and_through_edges.first;
  auto const& through_edges = trips_and_through_edges.second;

  return CreateLookupRiBasisResponse(
      fbb, fbb.CreateVector(utl::to_vec(trips, [&](trip const* trp) {
        return rib_trip(rc, trp, through_edges);
      })));
}

}  // namespace motis::lookup
