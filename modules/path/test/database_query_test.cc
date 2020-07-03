#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "utl/equal_ranges_linear.h"
#include "utl/progress_tracker.h"
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
#include "geo/polyline_format.h"

namespace m = motis;
namespace mp = m::path;
namespace mm = m::module;

namespace geo {
// NOLINTNEXTLINE
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

    utl::activate_progress_tracker("query_test");
  }

  void TearDown() override {
    utl::get_global_progress_trackers().clear();

    boost::filesystem::remove(db_fname_ + ".mdb");
    boost::filesystem::remove(db_fname_ + ".mdb-lock");
    boost::filesystem::remove(db_fname_ + ".pck");
  }

  std::string db_fname() const { return db_fname_ + ".mdb"; }

  static constexpr auto const xFactor = 1024;

  static geo::latlng fixed_x_to_latlng(tiles::fixed_coord_t x) {
    return tiles::fixed_to_latlng(tiles::fixed_xy{x * xFactor, 0UL});
  }

  // simulates rounding/floating point errors which occur during a db roundtrip
  static geo::latlng fixed_x_to_latlng_db(tiles::fixed_coord_t x) {
    return tiles::fixed_to_latlng(  // deserialize from db
        tiles::latlng_to_fixed(  // serialize for db
            tiles::fixed_to_latlng(tiles::fixed_xy{x * xFactor, 0UL})));
  }

  static std::vector<geo::latlng> fixed_x_to_line_db(
      std::vector<tiles::fixed_coord_t> const& xs) {
    return utl::to_vec(xs, [](auto x) { return fixed_x_to_latlng_db(x); });
  }

  static geo::latlng get_nth_coord(flatbuffers::Vector<double> const* vec,
                                   size_t n) {
    utl::verify(vec->size() >= n * 2 + 1, "cant get nth (size={}, n={})",
                vec->size(), n);
    return geo::latlng(vec->Get(n * 2), vec->Get(n * 2 + 1));
  }

  static std::pair<int64_t, uint64_t> add_feature(
      mp::db_builder& builder,
      std::vector<tiles::fixed_coord_t> const& geometry) {
    auto const line =
        utl::to_vec(geometry, [](auto x) { return fixed_x_to_latlng(x); });

    // don't inline this: MSVC miscompiles(?!) in release mode
    mcd::vector<m::service_class> classes;
    classes.push_back(m::service_class::OTHER);
    return builder.add_feature(line, {}, classes, false, 0.);
  }

  static void add_seq(mp::db_builder& builder, size_t idx,
                      std::vector<std::vector<int64_t>> const& feature_ids,
                      std::vector<std::vector<uint64_t>> const& hints) {
    add_seq(builder, idx, feature_ids, hints,
            utl::repeat_n<tiles::fixed_coord_t>(0, feature_ids.size()));
  }

  static void add_seq(mp::db_builder& builder, size_t idx,
                      std::vector<std::vector<int64_t>> const& feature_ids,
                      std::vector<std::vector<uint64_t>> const& hints,
                      std::vector<tiles::fixed_coord_t> const& fallbacks) {
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

    mp::station_seq seq;
    seq.station_ids_ = utl::repeat_n<mcd::string, mcd::vector<mcd::string>>(
        mcd::string{}, feature_ids.size() + 1);
    seq.classes_ = {motis::service_class::OTHER};
    seq.paths_ = mcd::to_vec(fallbacks, [](tiles::fixed_coord_t x) {
      return mp::osm_path{{fixed_x_to_latlng(x)}, {-1}};
    });

    auto boxes = utl::repeat_n(geo::box{}, feature_ids.size());

    builder.add_seq(idx, seq, boxes, feature_ids, hints_rle);
  }

  static std::pair<mm::msg_ptr, mp::PathSeqResponse const*> get_response(
      mp::path_database const& db, mp::path_database_query& q,
      size_t const index) {
    mm::message_creator mc;
    mc.create_and_finish(m::MsgContent_PathSeqResponse,
                         q.write_sequence(mc, db, index).Union());
    auto msg = mm::make_msg(mc);

    using m::MsgContent_PathSeqResponse;
    using mp::PathSeqResponse;
    auto const* resp = motis_content(PathSeqResponse, msg);
    return std::make_pair(std::move(msg), resp);
  }

  static std::pair<mm::msg_ptr, mp::PathByTripIdBatchResponse const*> get_batch(
      mp::path_database_query& q) {
    mm::message_creator mc;
    mc.create_and_finish(m::MsgContent_PathByTripIdBatchResponse,
                         q.write_batch(mc).Union());
    auto msg = mm::make_msg(mc);

    using m::MsgContent_PathByTripIdBatchResponse;
    using mp::PathByTripIdBatchResponse;
    auto const* resp = motis_content(PathByTripIdBatchResponse, msg);
    return std::make_pair(std::move(msg), resp);
  }

  static ptrdiff_t get_batch_path(int const line,
                                  mp::PathByTripIdBatchResponse const* resp,
                                  std::vector<geo::latlng> const& p0) {
    auto const pred = [&](auto const* p) {
      auto const decoded =
          geo::decode_polyline<6>(std::string_view{p->data(), p->size()});

      if (decoded.size() != p0.size()) {
        return false;
      }
      for (auto i = 0ULL; i < p0.size(); ++i) {
        if (std::abs(p0[i].lat_ - decoded[i].lat_) > 1e-6 ||
            std::abs(p0[i].lng_ - decoded[i].lng_) > 1e-6) {
          return false;
        }
      }
      return true;
    };

    auto begin = std::begin(*resp->polylines());
    auto end = std::end(*resp->polylines());
    auto it = std::find_if(begin, end, pred);
    utl::verify(it != end, "get_batch_path: path missing (line {})", line);
    utl::verify(std::none_of(std::next(it), end, pred),
                "get_batch_path: path duplicate (line {}", line);
    return std::distance(begin, it);
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

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_response(*db, q, 0UL);

  ASSERT_EQ(1, resp->segments()->size());

  auto const* coordinates = resp->segments()->Get(0)->coordinates();
  ASSERT_EQ(8, coordinates->size());
  EXPECT_EQ(fixed_x_to_latlng_db(0), get_nth_coord(coordinates, 0));
  EXPECT_EQ(fixed_x_to_latlng_db(1), get_nth_coord(coordinates, 1));
  EXPECT_EQ(fixed_x_to_latlng_db(2), get_nth_coord(coordinates, 2));
  EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 3));
}

TEST_F(path_database_query_test, two_features_in_segment) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1, 2, 3});
  auto [id2, h2] = add_feature(*builder, {3, 4, 5, 6});

  add_seq(*builder, 0UL, {{id1, id2}}, {{h1, h2}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_response(*db, q, 0UL);

  ASSERT_EQ(1, resp->segments()->size());

  auto const* coordinates = resp->segments()->Get(0)->coordinates();
  ASSERT_EQ(14, coordinates->size());
  EXPECT_EQ(fixed_x_to_latlng_db(0), get_nth_coord(coordinates, 0));
  EXPECT_EQ(fixed_x_to_latlng_db(1), get_nth_coord(coordinates, 1));
  EXPECT_EQ(fixed_x_to_latlng_db(2), get_nth_coord(coordinates, 2));
  EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 3));
  EXPECT_EQ(fixed_x_to_latlng_db(4), get_nth_coord(coordinates, 4));
  EXPECT_EQ(fixed_x_to_latlng_db(5), get_nth_coord(coordinates, 5));
  EXPECT_EQ(fixed_x_to_latlng_db(6), get_nth_coord(coordinates, 6));
}

TEST_F(path_database_query_test, two_segments) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1, 2, 3});
  auto [id2, h2] = add_feature(*builder, {3, 4, 5, 6});

  add_seq(*builder, 0UL, {{id1}, {id2}}, {{h1}, {h2}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_response(*db, q, 0UL);

  ASSERT_EQ(2, resp->segments()->size());
  {
    auto const* coordinates = resp->segments()->Get(0)->coordinates();
    ASSERT_EQ(8, coordinates->size());
    EXPECT_EQ(fixed_x_to_latlng_db(0), get_nth_coord(coordinates, 0));
    EXPECT_EQ(fixed_x_to_latlng_db(1), get_nth_coord(coordinates, 1));
    EXPECT_EQ(fixed_x_to_latlng_db(2), get_nth_coord(coordinates, 2));
    EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 3));
  }
  {
    auto const* coordinates = resp->segments()->Get(1)->coordinates();
    ASSERT_EQ(8, coordinates->size());
    EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 0));
    EXPECT_EQ(fixed_x_to_latlng_db(4), get_nth_coord(coordinates, 1));
    EXPECT_EQ(fixed_x_to_latlng_db(5), get_nth_coord(coordinates, 2));
    EXPECT_EQ(fixed_x_to_latlng_db(6), get_nth_coord(coordinates, 3));
  }
}

// empty segments occur if start and end of a segment is identical
// eg. repeated stations or identical stop positions for close stations
TEST_F(path_database_query_test, empty_segments) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id, h] = add_feature(*builder, {2, 3});

  add_seq(*builder, 0UL, {{}, {id}}, {{}, {h}}, {11, 12});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_response(*db, q, 0UL);
  ASSERT_EQ(2, resp->segments()->size());
  {
    auto const* coordinates = resp->segments()->Get(0)->coordinates();
    ASSERT_EQ(4, coordinates->size());
    EXPECT_EQ(fixed_x_to_latlng_db(11), get_nth_coord(coordinates, 0));
    EXPECT_EQ(fixed_x_to_latlng_db(11), get_nth_coord(coordinates, 1));
  }
  {
    auto const* coordinates = resp->segments()->Get(1)->coordinates();
    ASSERT_EQ(4, coordinates->size());
    EXPECT_EQ(fixed_x_to_latlng_db(2), get_nth_coord(coordinates, 0));
    EXPECT_EQ(fixed_x_to_latlng_db(3), get_nth_coord(coordinates, 1));
  }
}

TEST_F(path_database_query_test, batch_base) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});
  auto [id2, h2] = add_feature(*builder, {2, 3});
  auto [id3, h3] = add_feature(*builder, {4, 5});

  add_seq(*builder, 0UL, {{id1}, {id2}}, {{h1}, {h2}});
  add_seq(*builder, 1UL, {{-id2}, {id3}}, {{h2}, {h3}});
  add_seq(*builder, 2UL, {{-id2}}, {{h2}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q{20};
  q.add_sequence(0UL);
  q.add_sequence(1UL);
  q.add_sequence(2UL);
  EXPECT_EQ(3, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  EXPECT_EQ(0, resp->extras()->size());
  ASSERT_EQ(4, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());

  auto const p1 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({0, 1}));
  auto const p2 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({3, 2}));
  EXPECT_EQ(1, p2);  // reused first -> lower index, shorter json
  auto const p3 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({4, 5}));

  ASSERT_EQ(5, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p1, resp->segments()->Get(0)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(1)->indices()->size());
  EXPECT_EQ(-p2, resp->segments()->Get(1)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(2)->indices()->size());
  EXPECT_EQ(p2, resp->segments()->Get(2)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(3)->indices()->size());
  EXPECT_EQ(p3, resp->segments()->Get(3)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(4)->indices()->size());
  EXPECT_EQ(p2, resp->segments()->Get(4)->indices()->Get(0));
}

TEST_F(path_database_query_test, batch_empty) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});

  add_seq(*builder, 0UL, {{-id1}, {}}, {{h1}, {}}, {11, 12});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  EXPECT_EQ(0, resp->extras()->size());
  ASSERT_EQ(3, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());

  auto const p1 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({1, 0}));
  auto const p2 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({12, 12}));

  ASSERT_EQ(2, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p1, resp->segments()->Get(0)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(1)->indices()->size());
  EXPECT_EQ(p2, resp->segments()->Get(1)->indices()->Get(0));
}

TEST_F(path_database_query_test, batch_concat) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});
  auto [id2, h2] = add_feature(*builder, {2, 3});
  auto [id3, h3] = add_feature(*builder, {4, 5});
  auto [id4, h4] = add_feature(*builder, {6, 7});
  auto [id5, h5] = add_feature(*builder, {8, 9});
  auto [id6, h6] = add_feature(*builder, {10, 11});
  auto [id7, h7] = add_feature(*builder, {12, 13});
  auto [id8, h8] = add_feature(*builder, {14, 15});

  add_seq(*builder, 0UL, {{id1, id2}}, {{h1, h2}});  // fwd + fwd
  add_seq(*builder, 1UL, {{id3, -id4}}, {{h3, h4}});  // fwd + bwd
  add_seq(*builder, 2UL, {{-id5, id6}}, {{h5, h6}});  // bwd + fwd
  add_seq(*builder, 3UL, {{-id7, -id8}}, {{h7, h8}});  // bwd + bwd

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  q.add_sequence(1UL);
  q.add_sequence(2UL);
  q.add_sequence(3UL);
  EXPECT_EQ(4, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  EXPECT_EQ(0, resp->extras()->size());
  ASSERT_EQ(5, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());

  auto const p0 =
      get_batch_path(__LINE__, resp, fixed_x_to_line_db({0, 1, 2, 3}));
  auto const p1 =
      get_batch_path(__LINE__, resp, fixed_x_to_line_db({4, 5, 7, 6}));
  auto const p2 =
      get_batch_path(__LINE__, resp, fixed_x_to_line_db({9, 8, 10, 11}));
  auto const p3 =
      get_batch_path(__LINE__, resp, fixed_x_to_line_db({13, 12, 15, 14}));

  ASSERT_EQ(4, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p0, resp->segments()->Get(0)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(1)->indices()->size());
  EXPECT_EQ(p1, resp->segments()->Get(1)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(2)->indices()->size());
  EXPECT_EQ(p2, resp->segments()->Get(2)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(3)->indices()->size());
  EXPECT_EQ(p3, resp->segments()->Get(3)->indices()->Get(0));
}

TEST_F(path_database_query_test, batch_reverse_single) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});

  add_seq(*builder, 0UL, {{-id1}}, {{h1}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL);
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  ASSERT_EQ(2, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());
  auto const p0 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({1, 0}));

  ASSERT_EQ(1, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p0, resp->segments()->Get(0)->indices()->Get(0));
}

TEST_F(path_database_query_test, batch_partial_sequence) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});
  auto [id2, h2] = add_feature(*builder, {2, 3});
  auto [id3, h3] = add_feature(*builder, {4, 5});

  add_seq(*builder, 0UL, {{id1}, {id2}, {id3}}, {{h1}, {h2}, {h3}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_sequence(0UL, {1, 2});
  q.add_sequence(0UL, {0});
  EXPECT_EQ(2, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(1, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  EXPECT_EQ(0, resp->extras()->size());
  ASSERT_EQ(4, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());
  auto const p0 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({0, 1}));
  auto const p1 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({2, 3}));
  auto const p2 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({4, 5}));

  ASSERT_EQ(3, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p1, resp->segments()->Get(0)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(1)->indices()->size());
  EXPECT_EQ(p2, resp->segments()->Get(1)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(2)->indices()->size());
  EXPECT_EQ(p0, resp->segments()->Get(2)->indices()->Get(0));
}

TEST_F(path_database_query_test, batch_extra) {
  auto builder = std::make_unique<mp::db_builder>(db_fname());

  auto [id1, h1] = add_feature(*builder, {0, 1});

  add_seq(*builder, 0UL, {{id1}}, {{h1}});

  builder->finish();
  builder.reset();

  auto db = mp::make_path_database(db_fname(), true);

  mp::path_database_query q;
  q.add_extra({{fixed_x_to_latlng(10), fixed_x_to_latlng(11)},
               {fixed_x_to_latlng(12), fixed_x_to_latlng(13)}});
  EXPECT_EQ(1, q.sequences_.size());

  q.execute(*db);
  EXPECT_EQ(0, q.subqueries_.size());

  auto const& [msg, resp] = get_batch(q);

  ASSERT_EQ(3, resp->polylines()->size());
  EXPECT_EQ(0, resp->polylines()->Get(0)->size());
  auto const p0 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({10, 11}));
  auto const p1 = get_batch_path(__LINE__, resp, fixed_x_to_line_db({12, 13}));

  ASSERT_EQ(2, resp->segments()->size());

  ASSERT_EQ(1, resp->segments()->Get(0)->indices()->size());
  EXPECT_EQ(p0, resp->segments()->Get(0)->indices()->Get(0));

  ASSERT_EQ(1, resp->segments()->Get(1)->indices()->size());
  EXPECT_EQ(p1, resp->segments()->Get(1)->indices()->Get(0));

  ASSERT_EQ(2, resp->extras()->size());
  EXPECT_EQ(1, resp->extras()->Get(0));
  EXPECT_EQ(2, resp->extras()->Get(1));
}
