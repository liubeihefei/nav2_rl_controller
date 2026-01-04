#include "gtest/gtest.h"
#include "nav2_rl_controller/rl_controller.hpp"

using namespace nav2_rl_controller;

TEST(HistoryBufferTest, AssembleObservationWaitsUntilFull)
{
  RLController controller;

  // 调用 history_length 次应始终返回空（未满）
  for (size_t i = 0; i < 50; ++i) {
    auto v = controller.assembleObservationPublic();
    EXPECT_TRUE(v.empty());
  }

  // 第 51 次（有 50 帧历史）应返回完整输入
  auto vec = controller.assembleObservationPublic();
  const size_t expected = 51u * 25u;
  ASSERT_EQ(vec.size(), expected);
  // 在无 costmap/pose 时，obs 扇区值与最小距离为 0，最后一维 last_action 为 0
  EXPECT_FLOAT_EQ(vec[0], 0.0f);
  EXPECT_FLOAT_EQ(vec[expected - 1], 0.0f);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}