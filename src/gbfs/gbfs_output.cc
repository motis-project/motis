#include "motis/gbfs/gbfs_output.h"

#include "motis/gbfs/mode.h"
#include "motis/gbfs/osr_profile.h"
#include "motis/gbfs/routing_data.h"

namespace motis::gbfs {

gbfs_output::~gbfs_output() = default;

gbfs_output::gbfs_output(osr::ways const& w,
                         gbfs_routing_data& gbfs_rd,
                         gbfs_products_ref const prod_ref,
                         bool const ignore_rental_return_constraints)
    : w_{w},
      gbfs_rd_{gbfs_rd},
      provider_{*gbfs_rd_.data_->providers_.at(prod_ref.provider_)},
      products_{provider_.products_.at(prod_ref.products_)},
      prod_rd_{gbfs_rd_.get_products_routing_data(prod_ref)},
      sharing_data_{prod_rd_->get_sharing_data(
          w_.n_nodes(), ignore_rental_return_constraints)},
      rental_{
          .providerId_ = provider_.id_,
          .providerGroupId_ = provider_.group_id_,
          .systemId_ = provider_.sys_info_.id_,
          .systemName_ = provider_.sys_info_.name_,
          .url_ = provider_.sys_info_.url_,
          .color_ = provider_.color_,
          .formFactor_ = to_api_form_factor(products_.form_factor_),
          .propulsionType_ = to_api_propulsion_type(products_.propulsion_type_),
          .returnConstraint_ =
              to_api_return_constraint(products_.return_constraint_)} {}

api::ModeEnum gbfs_output::get_mode() const { return api::ModeEnum::RENTAL; }

bool gbfs_output::is_time_dependent() const { return false; }

transport_mode_t gbfs_output::get_cache_key() const {
  return gbfs_rd_.get_transport_mode({provider_.idx_, products_.idx_});
}

osr::search_profile gbfs_output::get_profile() const {
  return get_osr_profile(products_.form_factor_);
}

osr::sharing_data const* gbfs_output::get_sharing_data() const {
  return &sharing_data_;
}

void gbfs_output::annotate_leg(nigiri::lang_t const&,
                               osr::node_idx_t const from_node,
                               osr::node_idx_t const to_node,
                               api::Leg& leg) const {
  auto const from_additional_node = w_.is_additional_node(from_node);
  auto const to_additional_node = w_.is_additional_node(to_node);
  auto const is_rental =
      (leg.mode_ == api::ModeEnum::BIKE || leg.mode_ == api::ModeEnum::CAR) &&
      (from_additional_node || to_additional_node);
  if (!is_rental) {
    return;
  }

  leg.rental_ = rental_;
  leg.mode_ = api::ModeEnum::RENTAL;
  auto& ret = *leg.rental_;
  if (w_.is_additional_node(from_node)) {
    auto const& an = prod_rd_->compressed_.additional_nodes_.at(
        get_additional_node_idx(from_node));
    std::visit(
        utl::overloaded{
            [&](additional_node::station const& s) {
              auto const& st = provider_.stations_.at(s.id_);
              ret.fromStationName_ = st.info_.name_;
              ret.stationName_ = st.info_.name_;
              ret.rentalUriAndroid_ = st.info_.rental_uris_.android_;
              ret.rentalUriIOS_ = st.info_.rental_uris_.ios_;
              ret.rentalUriWeb_ = st.info_.rental_uris_.web_;
            },
            [&](additional_node::vehicle const& v) {
              auto const& vs = provider_.vehicle_status_.at(v.idx_);
              if (auto const st = provider_.stations_.find(vs.station_id_);
                  st != end(provider_.stations_)) {
                ret.fromStationName_ = st->second.info_.name_;
              }
              ret.rentalUriAndroid_ = vs.rental_uris_.android_;
              ret.rentalUriIOS_ = vs.rental_uris_.ios_;
              ret.rentalUriWeb_ = vs.rental_uris_.web_;
            }},
        an.data_);
  }
  if (w_.is_additional_node(to_node)) {
    auto const& an = prod_rd_->compressed_.additional_nodes_.at(
        get_additional_node_idx(to_node));
    std::visit(
        utl::overloaded{[&](additional_node::station const& s) {
                          auto const& st = provider_.stations_.at(s.id_);
                          ret.toStationName_ = st.info_.name_;
                          if (!ret.stationName_) {
                            ret.stationName_ = ret.toStationName_;
                          }
                        },
                        [&](additional_node::vehicle const& v) {
                          auto const& vs = provider_.vehicle_status_.at(v.idx_);
                          if (auto const st =
                                  provider_.stations_.find(vs.station_id_);
                              st != end(provider_.stations_)) {
                            ret.toStationName_ = st->second.info_.name_;
                          }
                        }},
        an.data_);
  }
}

api::Place gbfs_output::get_place(nigiri::lang_t const&,
                                  osr::node_idx_t const n,
                                  std::optional<std::string> const& tz) const {
  if (w_.is_additional_node(n)) {
    auto const pos = get_sharing_data()->get_additional_node_coordinates(n);
    return api::Place{.name_ = get_node_name(n),
                      .lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .tz_ = tz,
                      .vertexType_ = api::VertexTypeEnum::BIKESHARE};
  } else {
    auto const pos = w_.get_node_pos(n).as_latlng();
    return api::Place{.lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .tz_ = tz,
                      .vertexType_ = api::VertexTypeEnum::NORMAL};
  }
}

std::string gbfs_output::get_node_name(osr::node_idx_t const n) const {
  auto const& an =
      prod_rd_->compressed_.additional_nodes_.at(get_additional_node_idx(n));
  return std::visit(
      utl::overloaded{[&](additional_node::station const& s) {
                        return provider_.stations_.at(s.id_).info_.name_;
                      },
                      [&](additional_node::vehicle const& v) {
                        auto const& vs = provider_.vehicle_status_.at(v.idx_);
                        auto const it =
                            provider_.stations_.find(vs.station_id_);
                        return it == end(provider_.stations_)
                                   ? provider_.sys_info_.name_
                                   : it->second.info_.name_;
                      }},
      an.data_);
}

std::size_t gbfs_output::get_additional_node_idx(
    osr::node_idx_t const n) const {
  return to_idx(n) - sharing_data_.additional_node_offset_;
}

}  // namespace motis::gbfs
