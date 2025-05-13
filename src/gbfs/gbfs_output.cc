#include "motis/gbfs/gbfs_output.h"

#include "motis/gbfs/mode.h"
#include "motis/gbfs/routing_data.h"

namespace motis::gbfs {

bool is_additional_node(osr::ways const& w, osr::node_idx_t const n) {
  return n != osr::node_idx_t::invalid() && n >= w.n_nodes();
}

gbfs_output::~gbfs_output() = default;

gbfs_output::gbfs_output(osr::ways const& w,
                         gbfs_routing_data& gbfs_rd,
                         gbfs_products_ref const prod_ref)
    : w_{w},
      gbfs_rd_{gbfs_rd},
      provider_{*gbfs_rd_.data_->providers_.at(prod_ref.provider_)},
      products_{provider_.products_.at(prod_ref.products_)},
      prod_rd_{gbfs_rd_.get_products_routing_data(prod_ref)},
      sharing_data_{
          .start_allowed_ = &prod_rd_->start_allowed_,
          .end_allowed_ = &prod_rd_->end_allowed_,
          .through_allowed_ = &prod_rd_->through_allowed_,
          .additional_node_offset_ = w_.n_nodes(),
          .additional_edges_ = &prod_rd_->compressed_.additional_edges_},
      rental_{
          .systemId_ = provider_.sys_info_.id_,
          .systemName_ = provider_.sys_info_.name_,
          .url_ = provider_.sys_info_.url_,
          .formFactor_ = to_api_form_factor(products_.form_factor_),
          .propulsionType_ = to_api_propulsion_type(products_.propulsion_type_),
          .returnConstraint_ =
              to_api_return_constraint(products_.return_constraint_)} {}

transport_mode_t gbfs_output::get_cache_key(osr::search_profile) const {
  return gbfs_rd_.get_transport_mode({provider_.idx_, products_.idx_});
}

osr::sharing_data const* gbfs_output::get_sharing_data() const {
  return &sharing_data_;
}

api::VertexTypeEnum gbfs_output::get_vertex_type() const {
  return api::VertexTypeEnum::BIKESHARE;
}

void gbfs_output::annotate(osr::node_idx_t const from_node,
                           osr::node_idx_t const to_node,
                           api::Leg& leg) const {
  leg.rental_ = rental_;
  auto& ret = *leg.rental_;
  if (is_additional_node(w_, from_node)) {
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
  if (is_additional_node(w_, to_node)) {
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

geo::latlng gbfs_output::get_node_pos(osr::node_idx_t const n) const {
  return std::visit(
      utl::overloaded{[&](additional_node::station const& s) {
                        return provider_.stations_.at(s.id_).info_.pos_;
                      },
                      [&](additional_node::vehicle const& vehicle) {
                        return provider_.vehicle_status_.at(vehicle.idx_).pos_;
                      }},
      prod_rd_->compressed_.additional_nodes_.at(get_additional_node_idx(n))
          .data_);
}

std::string gbfs_output::get_node_name(osr::node_idx_t const n) const {
  if (!is_additional_node(w_, n)) {
    return provider_.sys_info_.name_;
  }
  auto const& an =
      prod_rd_->compressed_.additional_nodes_.at(get_additional_node_idx(n));
  return std::visit(
      utl::overloaded{[&](additional_node::station const& s) {
                        auto const& st = provider_.stations_.at(s.id_);
                        return st.info_.name_;
                      },
                      [&](additional_node::vehicle const& v) {
                        auto const& vs = provider_.vehicle_status_.at(v.idx_);
                        if (auto const st =
                                provider_.stations_.find(vs.station_id_);
                            st != end(provider_.stations_)) {
                          return st->second.info_.name_;
                        }
                        return provider_.sys_info_.name_;
                      }},
      an.data_);
}

std::size_t gbfs_output::get_additional_node_idx(
    osr::node_idx_t const n) const {
  return to_idx(n) - sharing_data_.additional_node_offset_;
}

}  // namespace motis::gbfs