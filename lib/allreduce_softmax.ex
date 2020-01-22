defmodule Softmax.AllReduce do
  @world_size 3

  def start(rank, vector, world_size, caller) do
    name = String.to_atom("node_#{inspect(rank)}")
    :logger.debug("Rank #{inspect(rank)} - Got: #{inspect(vector)}")

    Process.register(self(), name)

    normalizer =
      vector
      |> Matrex.to_list()
      |> Enum.map(fn x -> {x, 1} end)
      |> Enum.reduce(fn x, acc ->
        :logger.debug("Rank: #{inspect(rank)} - Reducing...")
        merge(x, acc)
      end)

    Barrier.synchronize(caller)

    :logger.debug("Rank #{inspect(rank)} - Normalizer: #{inspect(normalizer)}")

    Utils.synchronize(rank, world_size)
    :logger.debug("Rank #{inspect(rank)} - Synchronized!")

    normalizer = distribute_normalizer(normalizer, rank, world_size)
    :logger.debug("Rank #{inspect(rank)} - Full Normalizer: #{inspect(normalizer)}")

    result = rescale(vector, normalizer)
    send(caller, {:result, rank, result})
    Process.unregister(name)
  end

  @spec merge({number(), number()}, {number(), number()}) :: {number(), float()}
  def merge({m1, d1}, {m2, d2}) do
    m3 = max(m1, m2)
    d3 = d1 * :math.exp(m1 - m3) + d2 * :math.exp(m2 - m3)
    {m3, d3}
  end

  def rescale(vector, {mv, dv}) do
    vector
    |> Matrex.subtract(mv)
    |> Matrex.apply(:exp)
    |> Matrex.divide(dv)
  end

  def distribute_normalizer(normalizer, rank, world_size) do
    ranks = 0..(world_size - 1)

    ranks
    |> Enum.map(fn
      ^rank -> :skip
      any_rank -> send_normalizer_to_rank(rank, any_rank, normalizer)
    end)

    reduce_normalizer(normalizer, rank, world_size - 1)
  end

  def send_normalizer_to_rank(from_rank, to_rank, normalizer) do
    rank_name = String.to_atom("node_#{inspect(to_rank)}")
    send({rank_name, node()}, {:normalizer, from_rank, normalizer})
  end

  def reduce_normalizer(normalizer, rank, 0) do
    :logger.debug("Rank #{inspect(rank)} - All normalizers reduced!")
    normalizer
  end

  def reduce_normalizer(this_normalizer, rank, count) do
    :logger.debug("Rank #{inspect(rank)} - #{inspect(count)} normalizers remaining")

    new_normalizer =
      receive do
        {:normalizer, _, that_normalizer} ->
          merge(this_normalizer, that_normalizer)
      end

    reduce_normalizer(new_normalizer, rank, count - 1)
  end

  def softmax(%Matrex{} = vector) do
    inputs =
      vector
      |> Utils.split_matrix(@world_size, :cols)

    pids =
      0..(@world_size - 1)
      |> Enum.zip(inputs)
      |> Enum.map(fn {rank, input} ->
        spawn_link(__MODULE__, :start, [rank, input, @world_size, self()])
      end)

    :logger.debug("PIDs: #{inspect pids}")

    Barrier.synchronize(pids)

    wait_for_results(pids)
  end

  defp wait_for_results(pids) do
    acc =
      0..(length(pids) - 1)
      |> Enum.map(fn elem -> {elem, %{}} end)
      |> Map.new()

    wait_for_results(pids, acc)
  end

  defp wait_for_results([], acc) do
    acc
  end

  defp wait_for_results([_ | pids], acc) do
    acc =
      receive do
        {:result, rank, result} ->
          %{acc | rank => result}
      end

    wait_for_results(pids, acc)
  end
end
