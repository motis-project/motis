#include "./graph_builder_test.h"

#include "motis/core/common/date_time_util.h"

using std::get;

namespace motis::loader {

class loader_graph_builder_multiple_ice : public loader_graph_builder_test {
public:
  loader_graph_builder_multiple_ice()
      : loader_graph_builder_test("multiple-ice-files", "20151025", 3) {}
};

TEST_F(loader_graph_builder_multiple_ice, eva_num) {
  auto& stations = sched_->eva_to_station_;
  EXPECT_EQ("8000013", stations["8000013"]->eva_nr_);
  EXPECT_EQ("8000025", stations["8000025"]->eva_nr_);
  EXPECT_EQ("8000078", stations["8000078"]->eva_nr_);
  EXPECT_EQ("8000122", stations["8000122"]->eva_nr_);
  EXPECT_EQ("8000228", stations["8000228"]->eva_nr_);
  EXPECT_EQ("8000260", stations["8000260"]->eva_nr_);
  EXPECT_EQ("8000261", stations["8000261"]->eva_nr_);
  EXPECT_EQ("8000284", stations["8000284"]->eva_nr_);
  EXPECT_EQ("8001844", stations["8001844"]->eva_nr_);
  EXPECT_EQ("8004158", stations["8004158"]->eva_nr_);
  EXPECT_EQ("8010101", stations["8010101"]->eva_nr_);
  EXPECT_EQ("8010205", stations["8010205"]->eva_nr_);
  EXPECT_EQ("8010222", stations["8010222"]->eva_nr_);
  EXPECT_EQ("8010240", stations["8010240"]->eva_nr_);
  EXPECT_EQ("8010309", stations["8010309"]->eva_nr_);
  EXPECT_EQ("8011102", stations["8011102"]->eva_nr_);
  EXPECT_EQ("8011113", stations["8011113"]->eva_nr_);
  EXPECT_EQ("8011956", stations["8011956"]->eva_nr_);
  EXPECT_EQ("8098160", stations["8098160"]->eva_nr_);
}

TEST_F(loader_graph_builder_multiple_ice, simple_test) {
  auto& stations = sched_->eva_to_station_;
  ASSERT_EQ("Augsburg Hbf", stations["8000013"]->name_);
  ASSERT_EQ("Bamberg", stations["8000025"]->name_);
  ASSERT_EQ("Donauwörth", stations["8000078"]->name_);
  ASSERT_EQ("Treuchtlingen", stations["8000122"]->name_);
  ASSERT_EQ("Lichtenfels", stations["8000228"]->name_);
  ASSERT_EQ("Würzburg Hbf", stations["8000260"]->name_);
  ASSERT_EQ("München Hbf", stations["8000261"]->name_);
  ASSERT_EQ("Nürnberg Hbf", stations["8000284"]->name_);
  ASSERT_EQ("Erlangen", stations["8001844"]->name_);
  ASSERT_EQ("München-Pasing", stations["8004158"]->name_);
  ASSERT_EQ("Erfurt Hbf", stations["8010101"]->name_);
  ASSERT_EQ("Leipzig Hbf", stations["8010205"]->name_);
  ASSERT_EQ("Lutherstadt Wittenberg", stations["8010222"]->name_);
  ASSERT_EQ("Naumburg(Saale)Hbf", stations["8010240"]->name_);
  ASSERT_EQ("Saalfeld(Saale)", stations["8010309"]->name_);
  ASSERT_EQ("Berlin Gesundbrunnen", stations["8011102"]->name_);
  ASSERT_EQ("Berlin Südkreuz", stations["8011113"]->name_);
  ASSERT_EQ("Jena Paradies", stations["8011956"]->name_);
  ASSERT_EQ("Berlin Hbf (tief)", stations["8098160"]->name_);
}

TEST_F(loader_graph_builder_multiple_ice, coordinates) {
  auto& stations = sched_->eva_to_station_;

  ASSERT_FLOAT_EQ(48.3654410, stations["8000013"]->width_);
  ASSERT_FLOAT_EQ(49.9007590, stations["8000025"]->width_);
  ASSERT_FLOAT_EQ(48.7140260, stations["8000078"]->width_);
  ASSERT_FLOAT_EQ(48.9612670, stations["8000122"]->width_);
  ASSERT_FLOAT_EQ(50.1464520, stations["8000228"]->width_);
  ASSERT_FLOAT_EQ(49.8017960, stations["8000260"]->width_);
  ASSERT_FLOAT_EQ(48.1402320, stations["8000261"]->width_);
  ASSERT_FLOAT_EQ(49.4456160, stations["8000284"]->width_);
  ASSERT_FLOAT_EQ(49.5958950, stations["8001844"]->width_);
  ASSERT_FLOAT_EQ(48.1498960, stations["8004158"]->width_);
  ASSERT_FLOAT_EQ(50.9725510, stations["8010101"]->width_);
  ASSERT_FLOAT_EQ(51.3465490, stations["8010205"]->width_);
  ASSERT_FLOAT_EQ(51.8675310, stations["8010222"]->width_);
  ASSERT_FLOAT_EQ(51.1630710, stations["8010240"]->width_);
  ASSERT_FLOAT_EQ(50.6503160, stations["8010309"]->width_);
  ASSERT_FLOAT_EQ(52.5489630, stations["8011102"]->width_);
  ASSERT_FLOAT_EQ(52.4750470, stations["8011113"]->width_);
  ASSERT_FLOAT_EQ(50.9248560, stations["8011956"]->width_);
  ASSERT_FLOAT_EQ(52.5255920, stations["8098160"]->width_);

  ASSERT_FLOAT_EQ(10.8855700, stations["8000013"]->length_);
  ASSERT_FLOAT_EQ(10.8994890, stations["8000025"]->length_);
  ASSERT_FLOAT_EQ(10.7714430, stations["8000078"]->length_);
  ASSERT_FLOAT_EQ(10.9081590, stations["8000122"]->length_);
  ASSERT_FLOAT_EQ(11.0594720, stations["8000228"]->length_);
  ASSERT_FLOAT_EQ(9.93578000, stations["8000260"]->length_);
  ASSERT_FLOAT_EQ(11.5583350, stations["8000261"]->length_);
  ASSERT_FLOAT_EQ(11.0829890, stations["8000284"]->length_);
  ASSERT_FLOAT_EQ(11.0019080, stations["8001844"]->length_);
  ASSERT_FLOAT_EQ(11.4614850, stations["8004158"]->length_);
  ASSERT_FLOAT_EQ(11.0384990, stations["8010101"]->length_);
  ASSERT_FLOAT_EQ(12.3833360, stations["8010205"]->length_);
  ASSERT_FLOAT_EQ(12.6620150, stations["8010222"]->length_);
  ASSERT_FLOAT_EQ(11.7969840, stations["8010240"]->length_);
  ASSERT_FLOAT_EQ(11.3749870, stations["8010309"]->length_);
  ASSERT_FLOAT_EQ(13.3885130, stations["8011102"]->length_);
  ASSERT_FLOAT_EQ(13.3653190, stations["8011113"]->length_);
  ASSERT_FLOAT_EQ(11.5874610, stations["8011956"]->length_);
  ASSERT_FLOAT_EQ(13.3695450, stations["8098160"]->length_);
}

TEST_F(loader_graph_builder_multiple_ice, route_nodes) {
  EXPECT_EQ(1, sched_->route_index_to_first_route_node_.size());

  for (auto const& first_route_node :
       sched_->route_index_to_first_route_node_) {
    ASSERT_TRUE(first_route_node->is_route_node());

    auto station_id = first_route_node->get_station()->id_;
    auto station_eva = sched_->stations_[station_id]->eva_nr_;

    EXPECT_EQ("8000284", station_eva);

    ASSERT_EQ(1, first_route_node->incoming_edges_.size());
    ASSERT_EQ(2, first_route_node->edges_.size());

    ASSERT_EQ(first_route_node->incoming_edges_[0]->from_,
              first_route_node->get_station());
    ASSERT_EQ(first_route_node->get_station(), first_route_node->edges_[0].to_);

    ASSERT_EQ(edge_type::ENTER_EDGE,
              first_route_node->incoming_edges_[0]->type());
    ASSERT_EQ(edge_type::ROUTE_EDGE, first_route_node->edges_[1].type());
    ASSERT_EQ(edge_type::EXIT_EDGE, first_route_node->edges_[0].type());

    auto next_route_node = first_route_node->edges_[1].to_;
    ASSERT_TRUE(next_route_node->is_route_node());

    ASSERT_EQ("8000260",
              sched_->stations_[next_route_node->get_station()->id_]->eva_nr_);

    // [M]otis [T]ime [O]ffset (1*MINUTES_A_DAY - GMT+1)
    auto const MTO = SCHEDULE_OFFSET_MINUTES - 60;
    ASSERT_TRUE(first_route_node->edges_[1].is_route_edge());
    ASSERT_EQ(1, first_route_node->edges_[1].m_.route_edge_.conns_.size());
    auto& lcon = first_route_node->edges_[1].m_.route_edge_.conns_;
    ASSERT_EQ(time{19 * 60 + 3 + MTO},
              lcon[0].event_time(event_type::DEP, SCHEDULE_OFFSET_DAYS));
    ASSERT_EQ(time{19 * 60 + 58 + MTO},
              lcon[0].event_time(event_type::ARR, SCHEDULE_OFFSET_DAYS));

    auto conns = get_connections(first_route_node, motis_time(1903));
    ASSERT_EQ(8, static_cast<int>(conns.size()));

    auto& stations = sched_->stations_;
    EXPECT_EQ("8000284",
              stations[get<2>(conns[0])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8000260",
              stations[get<2>(conns[1])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8010101",
              stations[get<2>(conns[2])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8010240",
              stations[get<2>(conns[3])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8010205",
              stations[get<2>(conns[4])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8010222",
              stations[get<2>(conns[5])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8011113",
              stations[get<2>(conns[6])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8098160",
              stations[get<2>(conns[7])->get_station()->id_]->eva_nr_);
    EXPECT_EQ("8011102",
              stations[get<3>(conns[7])->get_station()->id_]->eva_nr_);

    EXPECT_EQ(motis_time(1903),
              get<light_connection const*>(conns[0])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[0])));
    EXPECT_EQ(motis_time(2000),
              get<light_connection const*>(conns[1])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[1])));
    EXPECT_EQ(motis_time(2155),
              get<light_connection const*>(conns[2])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[2])));
    EXPECT_EQ(motis_time(2234),
              get<light_connection const*>(conns[3])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[3])));
    EXPECT_EQ(motis_time(2318),
              get<light_connection const*>(conns[4])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[4])));
    EXPECT_EQ(motis_time(2349),
              get<light_connection const*>(conns[5])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[5])));
    EXPECT_EQ(motis_time(2425),
              get<light_connection const*>(conns[6])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[6])));
    EXPECT_EQ(motis_time(2433),
              get<light_connection const*>(conns[7])->event_time(
                  event_type::DEP, get<day_idx_t>(conns[7])));

    EXPECT_EQ(motis_time(1958), get<0>(conns[0])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[0])));
    EXPECT_EQ(motis_time(2153), get<0>(conns[1])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[1])));
    EXPECT_EQ(motis_time(2232), get<0>(conns[2])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[2])));
    EXPECT_EQ(motis_time(2308), get<0>(conns[3])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[3])));
    EXPECT_EQ(motis_time(2347), get<0>(conns[4])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[4])));
    EXPECT_EQ(motis_time(2423), get<0>(conns[5])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[5])));
    EXPECT_EQ(motis_time(2430), get<0>(conns[6])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[6])));
    EXPECT_EQ(motis_time(2438), get<0>(conns[7])->event_time(
                                    event_type::ARR, get<day_idx_t>(conns[7])));

    ASSERT_TRUE(std::all_of(begin(conns), end(conns), [&](auto const& con) {
      auto fc = std::get<0>(con)->full_con_;
      return fc->con_info_->attributes_.size() == 1 &&
             fc->con_info_->train_nr_ == 1000 &&
             sched_->categories_[fc->con_info_->category_]->name_ == "ICE";
    }));

    for (auto const& c : conns) {
      auto fc = std::get<0>(c)->full_con_;
      ASSERT_EQ("---", fc->con_info_->provider_->short_name_);
      ASSERT_EQ("DB AG", fc->con_info_->provider_->long_name_);
      ASSERT_EQ("Deutsche Bahn AG", fc->con_info_->provider_->full_name_);
    }

    auto const& tracks = sched_->tracks_;
    auto const day = day_idx_t{0U};

    auto const get_track = [&](size_t const idx, event_type const ev_type) {
      return tracks
          .at(get<light_connection const*>(conns.at(idx))
                  ->full_con_->get_track(ev_type))
          .get_info(day)
          ->str();
    };

    EXPECT_EQ("6", get_track(0, event_type::DEP));
    EXPECT_EQ("", get_track(1, event_type::DEP));
    EXPECT_EQ("", get_track(2, event_type::DEP));
    EXPECT_EQ("1", get_track(3, event_type::DEP));
    EXPECT_EQ("", get_track(4, event_type::DEP));
    EXPECT_EQ("3", get_track(5, event_type::DEP));
    EXPECT_EQ("8", get_track(6, event_type::DEP));
    EXPECT_EQ("7", get_track(7, event_type::DEP));

    EXPECT_EQ("", get_track(0, event_type::ARR));
    EXPECT_EQ("", get_track(1, event_type::ARR));
    EXPECT_EQ("1", get_track(2, event_type::ARR));
    EXPECT_EQ("", get_track(3, event_type::ARR));
    EXPECT_EQ("3", get_track(4, event_type::ARR));
    EXPECT_EQ("8", get_track(5, event_type::ARR));
    EXPECT_EQ("7", get_track(6, event_type::ARR));
    EXPECT_EQ("6", get_track(7, event_type::ARR));

    auto const bt_info = get<0>(conns[0])->full_con_->con_info_->attributes_[0];
    auto const sn_info = get<0>(conns[4])->full_con_->con_info_->attributes_[0];
    auto const bt = bt_info.attr_;
    auto const sn = sn_info.attr_;
    EXPECT_TRUE(bt_info.traffic_days_->test(get<day_idx_t>(conns[0])));
    EXPECT_TRUE(bt_info.traffic_days_->test(get<day_idx_t>(conns[4])));
    EXPECT_EQ("BT", bt_info.attr_->code_);
    EXPECT_EQ("SN", sn_info.attr_->code_);
    EXPECT_EQ("Bordbistro", bt_info.attr_->text_);
    EXPECT_EQ("SnackPoint/Imbiss im Zug", sn_info.attr_->text_);

    auto const get_attr = [&](size_t const i) {
      auto const attr = get<light_connection const*>(conns[i])
                            ->full_con_->con_info_->attributes_.at(0);
      EXPECT_TRUE(attr.traffic_days_->test(get<day_idx_t>(conns[i])));
      return attr.attr_;
    };

    EXPECT_EQ(bt, get_attr(0));
    EXPECT_EQ(bt, get_attr(1));
    EXPECT_EQ(bt, get_attr(2));
    EXPECT_EQ(bt, get_attr(3));
    EXPECT_EQ(sn, get_attr(4));
    EXPECT_EQ(sn, get_attr(5));
    EXPECT_EQ(sn, get_attr(6));
    EXPECT_EQ(sn, get_attr(7));

    auto const from_node_6 = std::get<2>(conns[6]);
    EXPECT_EQ(2, from_node_6->incoming_edges_.size());
    EXPECT_TRUE(std::any_of(begin(from_node_6->incoming_edges_),
                            end(from_node_6->incoming_edges_),
                            [&](edge const* e) {
                              return e->type() == edge_type::INVALID_EDGE &&
                                     e->from_ == from_node_6->get_station();
                            }));
    EXPECT_TRUE(std::any_of(begin(from_node_6->incoming_edges_),
                            end(from_node_6->incoming_edges_),
                            [](edge const* e) { return e->is_route_edge(); }));

    auto const from_node_5 = std::get<2>(conns[5]);
    EXPECT_TRUE(std::none_of(
        begin(from_node_5->incoming_edges_), end(from_node_5->incoming_edges_),
        [](edge const* e) { return e->type() == edge_type::INVALID_EDGE; }));
  }
}

}  // namespace motis::loader
