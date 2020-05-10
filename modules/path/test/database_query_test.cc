#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "utl/equal_ranges_linear.h"
#include "utl/repeat_n.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "tiles/db/tile_index.h"
#include "tiles/feature/feature.h"
#include "tiles/fixed/convert.h"
#include "tiles/get_tile.h"

#include "motis/path/path_database.h"
#include "motis/path/path_database_query.h"
#include "motis/path/prepare/db_builder.h"

namespace mp = motis::path;
namespace mm = motis::module;

namespace geo {
void PrintTo(latlng const& ll, std::ostream* os) {
  auto old_p = os->precision();
  os->precision(11);
  *os << "(" << ll.lat_ << ", " << ll.lng_ << ")";
  os->precision(old_p);
}
}  // namespace geo

struct path_database_query_test : public ::testing::Test {

  void SetUp() override {
    // use boost filesystem for unique test case file names
    db_fname_ =
        boost::filesystem::unique_path("pathdb_test_%%%%-%%%%-%%%%-%%%%")
            .string();
    std::cout << db_fname_ << std::endl;
  }

  void TearDown() override {
    boost::filesystem::remove(db_fname_ + ".mdb");
    boost::filesystem::remove(db_fname_ + ".mdb-lock");
    boost::filesystem::remove(db_fname_ + ".pck");
  }

  std::string db_fname() const { return db_fname_ + ".mdb"; }

  static constexpr auto const xFactor = 128;

  static geo::latlng fixed_x_to_latlng(tiles::fixed_coord_t x) {
    return tiles::fixed_to_latlng(tiles::fixed_xy{x * xFactor, 0ul});
  }

  // simulates rounding/floating point errors which occur during a db roundtrip
  static geo::latlng fixed_x_to_latlng_db(tiles::fixed_coord_t x) {
    return tiles::fixed_to_latlng(  // deserialize from db
        tiles::latlng_to_fixed(  // serialize for db
            tiles::fixed_to_latlng(tiles::fixed_xy{x * xFactor, 0ul})));
  }

  static geo::latlng get_nth_coord(flatbuffers::Vector<double> const* vec,
                                   size_t n) {
    utl::verify(vec->size() >= n * 2 + 1, "cant get nth (size={}, n={})",
                vec->size(), n);
    return geo::latlng(vec->Get(n * 2), vec->Get(n * 2 + 1));
  }

  static std::pair<int64_t, uint64_t> add_feature(
      mp::db_builder& builder, std::vector<tiles::fixed_coord_t> geometry) {
    auto const line =
        utl::to_vec(geometry, [](auto x) { return fixed_x_to_latlng(x); });
    return builder.add_feature(line, {}, {0}, false);
  }

  static void add_seq(mp::db_builder& builder, size_t idx,
                      std::vector<std::vector<int64_t>> feature_ids,
                      std::vector<std::vector<uint64_t>> hints) {
    auto hints_rle = utl::to_vec(hints, [&](auto const& h) {
      std::vector<uint64_t> rle;
      utl::equal_ranges_linear(
          h, [](auto const& lhs, auto const& rhs) { return lhs == rhs; },
          [&](auto lb, auto ub) {
            rle.emplace_back(*lb);
            rle.emplace_back(std::distance(lb, ub));
          });
      return rle;
    });

    mp::resolved_station_seq seq;
    seq.station_ids_ = utl::repeat_n(std::string{}, feature_ids.size() + 1);
    seq.classes_ = {0};

    auto boxes = utl::repeat_n(geo::box{}, feature_ids.size());

    builder.add_seq(idx, seq, boxes, feature_ids, hints_rle);
  }

  std::string db_fname_;
};

TEST_F(path_database_query_test, simple) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1, 2, 3});

  add_seq(*builder, 0UL, {{id1}}, {{h1}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);
  auto const render_ctx = tiles::make_render_ctx(*db->db_handle_);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db, render_ctx);
  EXPECT_EQ(1, q.subqueries_.size());

  mm::message_creator mc;
  mc.create_and_finish(motis::MsgContent_PathSeqResponse,
                       q.write_sequence(mc, *db, 0UL).Union());
  auto msg = mm::make_msg(mc);

  using motis::MsgContent_PathSeqResponse;
  using motis::path::PathSeqResponse;
  auto const* resp = motis_content(PathSeqResponse, msg);

  ASSERT_EQ(1, resp->segments()->size());

  auto const* coordinates = resp->segments()->Get(0)->coordinates();
  ASSERT_EQ(8, coordinates->size());
  EXPECT_EQ(fixed_x_to_latlng_db(0), get_nth_coord(coordinates, 0));
  EXPECT_EQ(fixed_x_to_latlng_db(1), get_nth_coord(coordinates, 1));
  EXPECT_EQ(fixed_x_to_latlng_db(2), get_nth_coord(coordinates, 2));
  EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 3));
}
