defmodule Softmax.Distributed do
  @master_rank 0
  @world_size 6

  def start(rank, list) do
    name = String.to_atom("node_#{inspect(rank)}")
    Process.register(self(), name)
    send({:orchestrator, node()}, {:alive, rank})
    wait_for_nodes()
    node_normalization(list, rank)
  end

  def wait_for_nodes() do
    receive do
      :ok -> :ok
    end

  end

  def node_normalization(list, rank) do
    receive do
      {:compute, level, report_to} ->
        norm = compute_normalization(list, level, rank)
        send(report_to, {:result, rank, norm})
    end

    node_softmax(list, rank)
  end

  def compute_normalization(list, level, rank) do
    next_level = Integer.floor_div(level, 2)
    case level > 1 do
      true ->
        case rank == @world_size - 1 do
          false ->
            next_rank = rank + next_level
            name = String.to_atom("node_#{inspect(next_rank)}")
            send(
              {name, node()},
              {:compute, next_level, self()}
            )

            norm = compute_normalization(list, next_level, rank)
            case next_rank >= @world_size do
              false ->
                receive do
                  {:result, _, next_norm} ->
                    Softmax.Parallel.merge(norm, next_norm)
                end
              true -> norm
            end

          _ ->
            Softmax.Parallel.normalize(list)
        end

      false ->
        Softmax.Parallel.normalize(list)
    end
  end

  def node_softmax(list, @master_rank) do
    receive do
      {:result, _, norm} ->
        node_ranks = 1..(@world_size - 1)
        pairs = Enum.map(node_ranks, fn _ -> nil end)
        empty_map = Enum.into(Enum.zip(node_ranks, pairs), %{})
        pending = Enum.into(node_ranks, MapSet.new())
        for n <- node_ranks,
            do: send({String.to_atom("node_#{inspect(n)}"), node()}, {:norm, norm, self()})
        {mv, dv} = norm
        values = Softmax.Parallel.rescale(list, mv, dv)
        pairs = Map.put(empty_map, 0, values)
        result = gather_list(pairs, pending)
        send({:orchestrator, node()}, {:result, result})
    end
  end

  def node_softmax(list, rank) do
    receive do
      {:norm, {mv, dv}, master} ->
        values = Softmax.Parallel.rescale(list, mv, dv)
        send(master, {:values, rank, values})
    end
  end

  def gather_list(chunk_map, pending) do
    case MapSet.size(pending) > 0 do
      true ->
        receive do
          {:values, rank, values} ->
            chunk_map = Map.put(chunk_map, rank, values)
            pending = MapSet.delete(pending, rank)
            gather_list(chunk_map, pending)
        end
      false ->
        chunk_map
        |> Map.values()
        |> Enum.flat_map(fn x -> x end)
    end
  end

  def are_ranks_ready(node_set) do
    case MapSet.size(node_set) > 0 do
      true ->
        receive do
          {:alive, rank} ->
            node_set = MapSet.delete(node_set, rank)
            are_ranks_ready(node_set)
        end
      false -> :ok
    end
  end

  def softmax(x) do
    Process.register(self(), :orchestrator)
    chunk_size = Integer.floor_div(length(x), @world_size)
    chunks = Enum.chunk_every(x, chunk_size)
    idx = 0..length(chunks)-1
    chunk_map = Enum.into(Enum.zip(idx, chunks), %{})
    chunks = case Map.size(chunk_map) == @world_size do
      true -> Map.values(chunk_map)
      false ->
        rem = Enum.flat_map(@world_size..Map.size(chunk_map)-1,
                            fn n -> Map.get(chunk_map, n)  end)
        {_, chunk_map} = Map.get_and_update(
          chunk_map, @world_size-1, fn value -> {value, Enum.concat(value, rem)} end)
        Map.values(chunk_map)
    end
    level = :math.pow(2, :math.ceil(:math.log(@world_size)/:math.log(2)))
          |> round
    ranks = 0..(@world_size-1)
    pids = Enum.map(Enum.zip(ranks, chunks),
                    fn {rank, list} -> spawn_link(__MODULE__, :start, [rank, list]) end)
    :ok = are_ranks_ready(MapSet.new(ranks))
    for pid <- pids, do: send(pid, :ok)
    send({:node_0, node()}, {:compute, level, {:node_0, node()}})
    result = receive do
      {:result, result} -> result
    end
    Process.unregister(:orchestrator)
    result
  end
end
