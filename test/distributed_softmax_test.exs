defmodule Softmax.DistributedTest do
  use ExUnit.Case
  doctest Softmax.Distributed

  test "computes softmax" do
    x = [0.1, 0.9, 0.7, 0.2, 0.89, 0.12, 0.91, 0.31, 0.65, 0.92, 0.75,
         0.76, 0.21, 0.56, 0.45, 0.99, 0.88, 0.77, -0.2, -0.3]
    logits = Softmax.Distributed.softmax(x)
    {argmax, _} = Enum.zip(0..length(logits) - 1, logits)
                |> Enum.max_by(fn {_, x} -> x end)
    assert Enum.at(x, argmax) == 0.99
  end
end
