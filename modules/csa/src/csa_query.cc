#include "motis/csa/csa_query.h"

#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/module/context/motis_call.h"

#include "motis/csa/error.h"

using namespace motis::module;
using namespace motis::routing;

namespace motis::csa {

station_node const* get_station_node(schedule const& sched,
                                     InputStation const* input_station) {
  using guesser::StationGuesserResponse;

  std::string station_id;

  if (input_station->id()->Length() != 0) {
    station_id = input_station->id()->str();
  } else {
    module::message_creator b;
    b.create_and_finish(MsgContent_StationGuesserRequest,
                        guesser::CreateStationGuesserRequest(
                            b, 1, b.CreateString(input_station->name()->str()))
                            .Union(),
                        "/guesser");
    auto const msg = motis_call(make_msg(b))->val();
    auto const guesses = motis_content(StationGuesserResponse, msg)->guesses();
    if (guesses->size() == 0) {
      throw std::system_error(error::no_guess_for_station);
    }
    station_id = guesses->Get(0)->id()->str();
  }

  return motis::get_station_node(sched, station_id);
}

std::vector<station_id> get_metas(schedule const& sched,
                                  InputStation const* input_station,
                                  bool use_metas) {
  auto const node = get_station_node(sched, input_station);
  return use_metas ? utl::to_vec(sched.stations_[node->id_]->equivalent_,
                                 [](station const* st) { return st->index_; })
                   : std::vector<station_id>({node->id_});
}

csa_query::csa_query(schedule const& sched,
                     routing::RoutingRequest const* req) {
  utl::verify_ex(req->search_type() == SearchType_Default ||
                     req->search_type() == SearchType_Accessibility ||
                     req->search_type() == SearchType_DefaultPrice ||
                     req->search_type() == SearchType_DefaultPriceRegional,
                 std::system_error{error::search_type_not_supported});
  utl::verify_ex(req->use_start_footpaths(),
                 std::system_error{error::start_footpaths_no_disable});
  utl::verify_ex(req->via()->size() == 0U,
                 std::system_error{error::via_not_supported});
  utl::verify_ex(req->additional_edges()->size() == 0U,
                 std::system_error{error::additional_edges_not_supported});

  dir_ = req->search_dir() == SearchDir_Forward ? search_dir::FWD
                                                : search_dir::BWD;
  include_equivalent_ = req->include_equivalent();

  switch (req->start_type()) {
    case Start_OntripStationStart: {
      auto const start =
          reinterpret_cast<OntripStationStart const*>(req->start());
      verify_external_timestamp(sched, start->departure_time());
      meta_starts_ = get_metas(sched, start->station(), req->use_start_metas());
      meta_dests_ = get_metas(sched, req->destination(), req->use_dest_metas());
      search_interval_.begin_ =
          unix_to_motistime(sched, start->departure_time());
      break;
    }

    case Start_PretripStart: {
      auto const start = reinterpret_cast<PretripStart const*>(req->start());
      verify_external_timestamp(sched, start->interval()->begin());
      verify_external_timestamp(sched, start->interval()->end());
      meta_starts_ = get_metas(sched, start->station(), req->use_start_metas());
      meta_dests_ = get_metas(sched, req->destination(), req->use_dest_metas());
      search_interval_.begin_ =
          unix_to_motistime(sched, start->interval()->begin());
      search_interval_.end_ =
          unix_to_motistime(sched, start->interval()->end());
      min_connection_count_ = start->min_connection_count();
      extend_interval_earlier_ = start->extend_interval_earlier();
      extend_interval_later_ = start->extend_interval_later();
      break;
    }

    default: throw std::system_error{error::search_type_not_supported};
  }
}

}  // namespace motis::csa
