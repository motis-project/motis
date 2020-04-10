#include "motis/loader/wzr_loader.h"
#include "motis/core/schedule/category.h"
#include "motis/loader/util.h"

#include "gtest/gtest.h"

namespace motis::loader {

constexpr auto const wzr_classes_path =
    "base/loader/test_resources/wzr/wzr_classes.csv";
constexpr auto const wzr_matrix_path =
    "base/loader/test_resources/wzr/wzr_matrix.txt";

TEST(loader_wzr, matrix) {
  auto waiting_time_rules =
      load_waiting_time_rules(wzr_classes_path, wzr_matrix_path, {});

  ASSERT_TRUE(waiting_time_rules.waiting_time(1, 1) == 1);
  ASSERT_TRUE(waiting_time_rules.waiting_time(1, 2) == 2);
  ASSERT_TRUE(waiting_time_rules.waiting_time(1, 3) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(1, 4) == 4);
  ASSERT_TRUE(waiting_time_rules.waiting_time(1, 5) == 5);

  ASSERT_TRUE(waiting_time_rules.waiting_time(2, 1) == 10);
  ASSERT_TRUE(waiting_time_rules.waiting_time(2, 2) == 10);
  ASSERT_TRUE(waiting_time_rules.waiting_time(2, 3) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(2, 4) == 1);
  ASSERT_TRUE(waiting_time_rules.waiting_time(2, 5) == 1);

  ASSERT_TRUE(waiting_time_rules.waiting_time(3, 1) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(3, 2) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(3, 3) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(3, 4) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(3, 5) == 0);

  ASSERT_TRUE(waiting_time_rules.waiting_time(4, 1) == 4);
  ASSERT_TRUE(waiting_time_rules.waiting_time(4, 2) == 1);
  ASSERT_TRUE(waiting_time_rules.waiting_time(4, 3) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(4, 4) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(4, 5) == 0);

  ASSERT_TRUE(waiting_time_rules.waiting_time(5, 1) == 5);
  ASSERT_TRUE(waiting_time_rules.waiting_time(5, 2) == 6);
  ASSERT_TRUE(waiting_time_rules.waiting_time(5, 3) == 0);
  ASSERT_TRUE(waiting_time_rules.waiting_time(5, 4) == 1);
  ASSERT_TRUE(waiting_time_rules.waiting_time(5, 5) == 5);
}

TEST(loader_wzr, family_to_category_assignment) {
  auto c = mcd::make_unique<category>(category{"IC", 0});
  mcd::vector<mcd::unique_ptr<category>> category_ptrs;
  category_ptrs.emplace_back(std::move(c));

  auto waiting_time_rules =
      load_waiting_time_rules(wzr_classes_path, wzr_matrix_path, category_ptrs);
  ASSERT_TRUE(waiting_time_rules.waiting_time_category(0) == 1);
}

TEST(loader_wzr, train_class_waits_for_other_trains) {
  auto waiting_time_rules =
      load_waiting_time_rules(wzr_classes_path, wzr_matrix_path, {});

  ASSERT_TRUE(waiting_time_rules.waits_for_other_trains(1));
  ASSERT_TRUE(!waiting_time_rules.waits_for_other_trains(3));
}

TEST(loader_wzr, other_trains_wait_for_train_class) {
  auto waiting_time_rules =
      load_waiting_time_rules(wzr_classes_path, wzr_matrix_path, {});

  ASSERT_TRUE(waiting_time_rules.other_trains_wait_for(1));
  ASSERT_TRUE(!waiting_time_rules.other_trains_wait_for(3));
}

}  // namespace motis::loader
