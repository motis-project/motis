#include "motis/itinerary_start_end.h"

#include "nigiri/types.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"
#include "motis-api/motis-api.h"
#include "motis/match_platforms.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "utl/verify.h"
#include "utl/visit.h"

namespace motis {
static api::Place suggestion_to_place(adr::suggestion const& s,
                                      api::Place const& p,
                                      adr::typeahead const& ta_,
                                      adr::formatter const& f_,
                                      adr_ext const* ae_,
                                      nigiri::timetable const* tt_,
                                      tag_lookup const* tags_,
                                      osr::ways const* w_,
                                      osr::platforms const* pl_,
                                      platform_matches_t const* matches_

) {
  auto modes = std::optional<std::vector<api::ModeEnum>>{};
  auto importance = std::optional<double>{};
  auto id = std::string{};
  auto level = std::optional<double>{};
  auto stop_pos = std::optional<geo::latlng>{};
  utl::visit(
      s.location_,
      [&](adr::place_idx_t const pi) {
        auto const type = ta_.place_type_[pi] == adr::amenity_category::kExtra
                              ? api::LocationTypeEnum::STOP
                              : api::LocationTypeEnum::PLACE;
        if (type == api::LocationTypeEnum::STOP) {
          if (tt_ != nullptr && tags_ != nullptr) {
            auto const l = nigiri::location_idx_t{
                static_cast<nigiri::location_idx_t::value_t>(
                    ta_.place_osm_ids_[pi][0])};
            stop_pos = tt_->locations_.coordinates_[l];
            level = get_level(w_, pl_, matches_, l);
            id = tags_->id(*tt_, l);
          } else {
            id = fmt::format("stop/{}", pi);
          }

          if (ae_ != nullptr) {
            auto const i = adr_extra_place_idx_t{
                static_cast<adr_extra_place_idx_t::value_t>(pi -
                                                            ta_.ext_start_)};
            modes = to_modes(ae_->place_clasz_[i], 5);
            importance = ae_->place_importance_[i];
          }
        } else {
          id = fmt::format("{}/{}",
                           ta_.place_is_way_[to_idx(pi)] ? "way" : "node",
                           ta_.place_osm_ids_[pi][0]);
        }
        return std::string{ta_.strings_[s.str_].view()};
      },
      [&](adr::address const addr) {
        if (addr.house_number_ != adr::address::kNoHouseNumber) {
          auto street = ta_.strings_[s.str_].view();
          auto house_number =
              ta_.strings_[ta_.house_numbers_[addr.street_][addr.house_number_]]
                  .view();
          return fmt::format("{} {}", street, house_number);
        }
        return std::string{ta_.strings_[s.str_].view()};
      });

  auto const country_code = s.get_country_code(ta_);

  return {
      .name_ = s.format(ta_, f_, country_code.value_or("DE")),
      .stopId_ = std::move(id),
      .parentId_ = std::nullopt,
      .importance_ = importance,
      .lat_ =
          stop_pos.transform([](geo::latlng const& pos) { return pos.lat_; })
              .value_or(s.coordinates_.as_latlng().lat_),
      .lon_ =
          stop_pos.transform([](geo::latlng const& pos) { return pos.lng_; })
              .value_or(s.coordinates_.as_latlng().lng_),
      .level_ = level,
      .tz_ =
          s.tz_ == adr::timezone_idx_t::invalid()
              ? std::nullopt
              : std::optional{std::string{ta_.timezone_names_[s.tz_].view()}},
      .arrival_ = p.arrival_,
      .departure_ = p.departure_,
      .scheduledArrival_ = p.scheduledArrival_,
      .scheduledDeparture_ = p.scheduledDeparture_,
      .scheduledTrack_ = p.scheduledTrack_,
      .track_ = p.track_,
      .stopCode_ = p.stopCode_,
      .description_ = p.description_,
      .vertexType_ = p.vertexType_,
      .pickupType_ = p.pickupType_,
      .dropoffType_ = p.dropoffType_,
      .cancelled_ = p.cancelled_,
      .alerts_ = p.alerts_,
      .flex_ = p.flex_,
      .flexId_ = p.flexId_,
      .flexStartPickupDropOffWindow_ = p.flexStartPickupDropOffWindow_,
      .flexEndPickupDropOffWindow_ = p.flexEndPickupDropOffWindow_,
      .modes_ = modes};
}

static api::Place enrich_place_with_lookup(api::Place const& p,

                                           adr::reverse const& r_,
                                           adr::typeahead const& ta_,
                                           adr::formatter const& f_,
                                           adr_ext const* ae_,
                                           nigiri::timetable const* tt_,
                                           tag_lookup const* tags_,
                                           osr::ways const* w_,
                                           osr::platforms const* pl_,
                                           platform_matches_t const* matches_) {
  if (p.vertexType_ != api::VertexTypeEnum::NORMAL) {
    return p;
  }

  auto s =
      r_.lookup(ta_, geo::latlng{.lat_ = p.lat_, .lng_ = p.lon_}, 1).front();
  return suggestion_to_place(s, p, ta_, f_, ae_, tt_, tags_, w_, pl_, matches_);
}

void resolve_itinerary_start_end(api::Itinerary& itinerary,
                                 adr::reverse const& r_,
                                 adr::typeahead const& ta_,
                                 adr::formatter const& f_,
                                 adr_ext const* ae_,
                                 nigiri::timetable const* tt_,
                                 tag_lookup const* tags_,
                                 osr::ways const* w_,
                                 osr::platforms const* pl_,
                                 platform_matches_t const* matches_) {
  utl::verify(!itinerary.legs_.empty(), "Itinerary contains at least one leg");

  auto& first_leg = itinerary.legs_.front();
  first_leg.from_ = enrich_place_with_lookup(first_leg.from_, r_, ta_, f_, ae_,
                                             tt_, tags_, w_, pl_, matches_);

  auto& last_leg = itinerary.legs_.back();
  last_leg.to_ = enrich_place_with_lookup(last_leg.to_, r_, ta_, f_, ae_, tt_,
                                          tags_, w_, pl_, matches_);
}
}  // namespace motis
