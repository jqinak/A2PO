def a2_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
  print(data_source.shape)
  print(solution_str)
  print(ground_truth)
  print(extra_info)
  return len(solution_str)/100