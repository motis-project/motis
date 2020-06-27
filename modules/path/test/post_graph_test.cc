#include "gtest/gtest.h"

#include "utl/repeat_n.h"
#include "utl/to_vec.h"

#include "motis/path/prepare/post/build_post_graph.h"
#include "motis/path/prepare/post/color_to_seq_seg_index.h"
#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/post/post_processor.h"
#include "motis/path/prepare/post/post_serializer.h"

namespace mp = motis::path;

#define CHECK_AP(rpp, a, b)                          \
  if ((rpp).second) {                                \
    EXPECT_EQ((a), (rpp).first->from_->id_.osm_id_); \
    EXPECT_EQ((b), (rpp).first->to_->id_.osm_id_);   \
  } else {                                           \
    EXPECT_EQ((b), (rpp).first->from_->id_.osm_id_); \
    EXPECT_EQ((a), (rpp).first->to_->id_.osm_id_);   \
  }

mp::station_seq make_resolved_cls_seq(
    mcd::vector<motis::service_class> const& classes,
    mcd::vector<mcd::vector<int64_t>> const& paths) {
  return mp::station_seq{
      utl::repeat_n<mcd::string, mcd::vector<mcd::string>>(mcd::string{},
                                                           paths.size() + 1),
      utl::repeat_n<mcd::string, mcd::vector<mcd::string>>(mcd::string{},
                                                           paths.size() + 1),
      utl::repeat_n<geo::latlng, mcd::vector<geo::latlng>>(geo::latlng{},
                                                           paths.size() + 1),
      classes,
      mcd::to_vec(paths,
                  [](auto const& p) -> mp::osm_path {
                    return {mcd ::to_vec(p,
                                         [](double e) {
                                           return geo::latlng{e, e};
                                         }),
                            p};
                  }),
      {},
      0.F};
}

mp::station_seq make_resolved_seq(
    mcd::vector<mcd::vector<int64_t>> const& paths) {
  return make_resolved_cls_seq({motis::service_class::OTHER}, paths);
}

TEST(post_graph, simple) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{
          make_resolved_seq({{0, 1, 2}, {2, 3, 4, 5}})}));

  ASSERT_EQ(6, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(2, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(1));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(4, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 2, 5);
  }
}

TEST(post_graph, split_fwd) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3}}),
                                   make_resolved_seq({{0, 1, 2, 4, 5}})}));

  ASSERT_EQ(6, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(2, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);

    EXPECT_EQ(2, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 2, 3);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(2, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);

    EXPECT_EQ(3, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 2, 5);
  }
}

TEST(post_graph, split_bwd) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{2, 1, 0}}),
                                   make_resolved_seq({{3, 1, 0}})}));

  ASSERT_EQ(4, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(2, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 2, 1);

    EXPECT_EQ(2, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 0);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(2, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 3, 1);

    EXPECT_EQ(2, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 0);
  }
}

TEST(post_graph, reverse) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2}}),
                                   make_resolved_seq({{2, 1, 0}})}));

  ASSERT_EQ(3, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 2, 0);
  }
}

TEST(post_graph, reverse_partial) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3, 4, 5, 6}}),
                                   make_resolved_seq({{7, 4, 3, 2, 8}})}));

  ASSERT_EQ(9, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);

    EXPECT_EQ(3, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 2, 4);

    EXPECT_EQ(3, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 4, 6);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 7, 4);

    EXPECT_EQ(3, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 4, 2);

    EXPECT_EQ(2, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 2, 8);
  }
}

TEST(post_graph, inside) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3, 4, 5, 6}}),
                                   make_resolved_seq({{2, 3, 4}})}));

  ASSERT_EQ(7, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);

    EXPECT_EQ(3, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 2, 4);

    EXPECT_EQ(3, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 4, 6);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 2, 4);
  }
}

TEST(post_graph, inside_reverse) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3, 4, 5, 6}}),
                                   make_resolved_seq({{4, 3, 2}})}));

  ASSERT_EQ(7, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);

    EXPECT_EQ(3, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 2, 4);

    EXPECT_EQ(3, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 4, 6);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 4, 2);
  }
}

TEST(post_graph, cross) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2}}),
                                   make_resolved_seq({{3, 1, 4}})}));

  ASSERT_EQ(5, g.nodes_.size());
  ASSERT_EQ(2, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());
  ASSERT_EQ(1, g.segment_ids_.at(1).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 2);
  }
  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(1).at(0));
    ASSERT_EQ(1, rp.size());

    EXPECT_EQ(3, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 3, 4);
  }
}

TEST(post_graph, self_loop_short) {
  // actually, this does not make that much sense in reality:
  // on the map 1 - 2 - 1 would just be a spike
  // -> adjacent nodes are connected by direct liness
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 1, 3}})}));

  ASSERT_EQ(4, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(4, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 1);

    EXPECT_EQ(2, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 2);

    EXPECT_EQ(2, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 2, 1);

    EXPECT_EQ(2, rp.at(3).first->path_.size());
    CHECK_AP(rp.at(3), 1, 3);
  }
}

TEST(post_graph, self_loop_long) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3, 1, 4}})}));

  ASSERT_EQ(5, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 1);

    EXPECT_EQ(4, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 1);

    EXPECT_EQ(2, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 1, 4);
  }
}

TEST(post_graph, self_loop_short_suffix) {
  // makes no sense, should still work
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 1}})}));

  ASSERT_EQ(3, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(3, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 1);

    EXPECT_EQ(2, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 2);

    EXPECT_EQ(2, rp.at(2).first->path_.size());
    CHECK_AP(rp.at(2), 2, 1);
  }
}

TEST(post_graph, self_loop_long_suffix) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 3, 1}})}));

  ASSERT_EQ(4, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(2, rp.size());

    EXPECT_EQ(2, rp.at(0).first->path_.size());
    CHECK_AP(rp.at(0), 0, 1);

    EXPECT_EQ(4, rp.at(1).first->path_.size());
    CHECK_AP(rp.at(1), 1, 1);
  }
}

TEST(post_graph, self_loop_full_short) {
  // makes no sense, should still work
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 0}})}));

  ASSERT_EQ(2, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(0, rp.size());
  }
}

TEST(post_graph, self_loop_start_long) {
  // makes no sense, should still work
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 1, 2, 0}})}));

  ASSERT_EQ(3, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(0, rp.size());
  }
}

TEST(post_graph, no_path) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{0, 0}})}));

  ASSERT_EQ(1, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(0, rp.size());
  }
}

TEST(post_graph, invalid_path) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{make_resolved_seq({{}})}));

  ASSERT_EQ(0, g.nodes_.size());
  ASSERT_EQ(1, g.segment_ids_.size());
  ASSERT_EQ(1, g.segment_ids_.at(0).size());

  mp::post_process(g);

  {
    auto const rp = mp::reconstruct_path(g.segment_ids_.at(0).at(0));
    ASSERT_EQ(0, rp.size());
  }
}

TEST(post_graph, reverse_path_min_class) {
  auto g = mp::build_post_graph(mcd::make_unique<mcd::vector<mp::station_seq>>(
      mcd::vector<mp::station_seq>{
          make_resolved_cls_seq(
              {motis::service_class::RE, motis::service_class::RB},
              {{0, 1, 2}}),
          make_resolved_cls_seq({motis::service_class::S}, {{2, 1, 0}})}));

  mp::post_process(g);
  ASSERT_EQ(1, g.atomic_paths_.size());

  auto const& ap = *g.atomic_paths_.front();

  mp::color_to_seq_seg_index index{g};

  auto const& [seq_segs, classes] = index.resolve_atomic_path(ap);

  EXPECT_EQ((std::vector<mp::seq_seg>{{0, 0}, {1, 0}}), seq_segs);
  EXPECT_EQ((mcd::vector<motis::service_class>{motis::service_class::RE,
                                               motis::service_class::RB,
                                               motis::service_class::S}),
            classes);
}
