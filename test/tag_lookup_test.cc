#include "gtest/gtest.h"

#include "boost/url/url.hpp"

#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/tag_lookup.h"

#include "./test_case.h"

using namespace std::string_view_literals;
using namespace osr;

TEST(motis, tag_lookup) {
  auto [d, _] = get_test_case<test_case::FFM_tag_lookup>();
  auto const rt = d.rt_;
  auto const rtt = rt->rtt_.get();
  EXPECT_TRUE(
      d.tags_->get_trip(*d.tt_, rtt, "20190501_01:15_test_S3 ").first.valid());
  EXPECT_TRUE(
      d.tags_->get_trip(*d.tt_, rtt, "20190501_01:05_test_Ü4").first.valid());
  EXPECT_TRUE(d.tags_->get_trip(*d.tt_, rtt, "20190501_00:35_test_+ICE_&A")
                  .first.valid());
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_DA 10"));
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_+FFM_HÄUPT_&U"));
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_DA 10"));
  auto u = boost::urls::url{
      "/api",
  };
  u.params({true, false, false}).append({"encoded", "a+& b"});
  u.params({false, false, false}).append({"unencoded", "a+& b"});
  {
    std::stringstream buffer;
    buffer << u;
    EXPECT_EQ("/api?encoded=a%2B%26+b&unencoded=a+%26%20b", buffer.str());
  }
  {
    std::stringstream buffer;
    buffer << u.encoded_params();
    EXPECT_EQ("encoded=a%2B%26+b&unencoded=a+%26%20b", buffer.str());
  }
}
