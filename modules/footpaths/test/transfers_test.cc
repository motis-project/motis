#include "gtest/gtest.h"

#include "motis/footpaths/transfers.h"

/* merge transfer requests */

TEST(footpaths, merge_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  auto should_treq_k = transfer_request_keys{};
  auto is_treq_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_treq_k, should_treq_k);
}

TEST(footpaths, merge_right_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};

  auto treq_k_b = transfer_request_keys{};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_left_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};
  treq_k_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.from_nloc_key_ = key64_t{26};
  treq_k_a.profile_ = key8_t{1};
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};

  auto treq_k_b = transfer_request_keys{};
  treq_k_b.from_nloc_key_ = key64_t{26};
  treq_k_b.profile_ = key8_t{1};
  treq_k_b.to_nloc_keys_ = {key64_t{3}, key64_t{4}, key64_t{5}};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.from_nloc_key_ = key64_t{26};
  should_treq_k.profile_ = key8_t{1};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5},
                                 key64_t{4}};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_id_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.from_nloc_key_ = key64_t{26};
  treq_k_a.profile_ = key8_t{1};
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};

  auto treq_k_b = transfer_request_keys{};
  treq_k_b.from_nloc_key_ = key64_t{26};
  treq_k_b.profile_ = key8_t{1};
  treq_k_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.from_nloc_key_ = key64_t{26};
  should_treq_k.profile_ = key8_t{1};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3},
                                 key64_t{5}};

  auto is_treq_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_treq_k, should_treq_k);
}

TEST(footpaths, merge_different_from_nloc_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.from_nloc_key_ = key64_t{1};

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

TEST(footpaths, merge_different_profiles) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.profile_ = key8_t{1};

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

/* merge transfer results */
TEST(transfer_result, merge_empty_transfer_results) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};

  auto should_tres = transfer_result{};
  auto is_tres = merge(tres_a, tres_b);

  ASSERT_EQ(is_tres, should_tres);
}

TEST(transfer_result, merge_right_empty_transfer_result) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  tres_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}};
  tres_a.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0}};

  auto tres_b = transfer_result{};

  auto should_tres = transfer_result{};
  should_tres.to_nloc_keys_ = {key64_t{1}, key64_t{2}};
  should_tres.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0}};

  auto is_tres = merge(tres_a, tres_b);

  ASSERT_EQ(is_tres, should_tres);
}

TEST(transfer_result, merge_left_empty_transfer_result) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};
  tres_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}};
  tres_b.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0}};

  auto should_tres = transfer_result{};
  should_tres.to_nloc_keys_ = {key64_t{1}, key64_t{2}};
  should_tres.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0}};

  auto is_tres = merge(tres_a, tres_b);

  ASSERT_EQ(is_tres, should_tres);
}

TEST(transfer_result, merge_transfer_result) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  tres_a.from_nloc_key_ = key64_t{26};
  tres_a.profile_ = key8_t{1};
  tres_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  tres_a.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0},
                   transfer_info{{}, 3.0}, transfer_info{{}, 5.0}};

  auto tres_b = transfer_result{};
  tres_b.from_nloc_key_ = key64_t{26};
  tres_b.profile_ = key8_t{1};
  tres_b.to_nloc_keys_ = {key64_t{3}, key64_t{4}, key64_t{5}};
  tres_b.infos_ = {transfer_info{{}, 3.0}, transfer_info{{}, 4.0},
                   transfer_info{{}, 5.0}};

  auto should_tres = transfer_result{};
  should_tres.from_nloc_key_ = key64_t{26};
  should_tres.profile_ = key8_t{1};
  should_tres.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5},
                               key64_t{4}};
  should_tres.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0},
                        transfer_info{{}, 3.0}, transfer_info{{}, 5.0},
                        transfer_info{{}, 4.0}};

  auto is_tres = merge(tres_a, tres_b);

  ASSERT_EQ(is_tres, should_tres);
}

TEST(transfer_Result, merge_id_transfer_results) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  tres_a.from_nloc_key_ = key64_t{26};
  tres_a.profile_ = key8_t{1};
  tres_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  tres_a.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0},
                   transfer_info{{}, 3.0}, transfer_info{{}, 5.0}};

  auto tres_b = transfer_result{};
  tres_b.from_nloc_key_ = key64_t{26};
  tres_b.profile_ = key8_t{1};
  tres_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  tres_b.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0},
                   transfer_info{{}, 3.0}, transfer_info{{}, 5.0}};

  auto should_tres = transfer_result{};
  should_tres.from_nloc_key_ = key64_t{26};
  should_tres.profile_ = key8_t{1};
  should_tres.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  should_tres.infos_ = {transfer_info{{}, 1.0}, transfer_info{{}, 2.0},
                        transfer_info{{}, 3.0}, transfer_info{{}, 5.0}};

  auto is_tres = merge(tres_a, tres_b);

  ASSERT_EQ(is_tres, should_tres);
}

TEST(transfer_result, merge_different_from_nloc_keys) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};

  tres_a.from_nloc_key_ = key64_t{1};

  ASSERT_ANY_THROW(merge(tres_a, tres_b));
}

TEST(transfer_result, merge_different_profiles) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};

  tres_a.profile_ = key8_t{1};

  ASSERT_ANY_THROW(merge(tres_a, tres_b));
}

TEST(transfer_result, merge_different_to_nloc_keys_a) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};

  tres_a.to_nloc_keys_ = {key64_t{1}};

  ASSERT_ANY_THROW(merge(tres_a, tres_b));
}

TEST(transfer_result, merge_different_to_nloc_keys_b) {
  using namespace motis::footpaths;

  auto tres_a = transfer_result{};
  auto tres_b = transfer_result{};

  tres_b.to_nloc_keys_ = {key64_t{1}};

  ASSERT_ANY_THROW(merge(tres_a, tres_b));
}
