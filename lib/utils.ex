defmodule Utils do
  def synchronize(rank, world_size) do
    0..(world_size - 1)
    |> Enum.map(fn
      ^rank ->
        :skip

      r ->
        send_sync_message(r)
    end)

    wait_for_sync(world_size - 1)
  end

  defp send_sync_message(rank) do
    name = String.to_atom("node_#{inspect(rank)}")
    send({name, node()}, :sync)
  end

  defp wait_for_sync(0) do
    :ok
  end

  defp wait_for_sync(to_answer) do
    receive do
      :sync -> :ok
    end

    wait_for_sync(to_answer - 1)
  end

  defp compress_range(r) do
    case r do
      [_ | [_ | _]] ->
        first_element = Enum.at(r, 0)
        last_element = Enum.at(r, length(r) - 1)
        [first_element, last_element]

      r ->
        r
    end
  end

  def split_matrix(mat, n) do
    split_matrix(mat, n, :rows)
  end

  def split_matrix(mat, n, :rows) do
    {h, _} = mat[:size]
    chunks = Integer.floor_div(h, n)

    1..h
    |> Enum.chunk_every(chunks)
    |> Enum.map(&compress_range/1)
    |> Enum.map(fn
      [s_idx, e_idx] -> mat[s_idx..e_idx]
      [idx] -> mat[idx]
    end)
  end

  def split_matrix(mat, n, :cols) do
    {h, w} = mat[:size]
    chunks = Integer.floor_div(w, n)
    1..w
    |> Enum.chunk_every(chunks)
    |> Enum.map(&compress_range/1)
    |> Enum.map(fn
        [s_idx, e_idx] -> Matrex.submatrix(mat, 1..h, s_idx..e_idx)
        [idx] -> Matrex.column(mat, idx)
      end)
  end
end
