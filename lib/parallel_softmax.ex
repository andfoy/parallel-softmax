defmodule Softmax.Parallel do
  @moduledoc """
  Documentation for ParallelSoftmax.
  """

  def pmap(collection, func) do
    collection
    |> Enum.map(&(Task.async(fn -> func.(&1) end)))
    |> Enum.map(&Task.await/1)
  end

  @spec merge({number(), number()}, {number(), number()}) :: {number(), float()}
  def merge({m1, d1}, {m2, d2}) do
    m3 = max(m1, m2)
    d3 = d1 * :math.exp(m1 - m3) + d2 * :math.exp(m2 - m3)
    {m3, d3}
  end

  @spec normalize([number()]) :: {number(), number()}
  def normalize(x) when length(x) == 1 do
    [x1|_] = x
    {x1, 1}
  end

  def normalize(x) when length(x) > 1 do
    {left, right} = Enum.split(x, Integer.floor_div(length(x), 2))
    pid = Task.async(fn -> normalize(right) end)
    left_norm = normalize(left)
    right_norm = Task.await(pid)
    merge(left_norm, right_norm)
  end

  def rescale(x, mv, dv) do
    pmap(x, fn elem -> :math.exp(elem - mv) / dv end)
  end

  @spec softmax([number()]) :: [number()]
  def softmax(x) do
    {mv, dv} = normalize(x)
    rescale(x, mv, dv)
  end
end
